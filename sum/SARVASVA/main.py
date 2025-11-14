"""
Main Application: PDF Document Processor with ML/DL Classification and Summarization
"""
import json
import logging
from pdf_indexer import PDFIndexer
from line_classifier import LineClassifier
from hierarchy_builder import HierarchyBuilder
from content_mapper import ContentMapper
from summarizer import ContentSummarizer
from quality_validator import QualityValidator
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFDocumentProcessor:
    """
    Main processor that combines all components:
    1. Indexing: Breaks down PDF by page and line number
    2. Categorization: Classifies lines as heading/subheading/content
    3. Content Overview: Maps headings to start/end positions
    4. Summarization: Summarizes content when requested
    """
    
    def __init__(self, pdf_path: str, classifier_model: Optional[str] = None,
                 summarizer_model: Optional[str] = None):
        """
        Initialize the PDF document processor
        
        Args:
            pdf_path: Path to the PDF file
            classifier_model: Optional Hugging Face model for classification
            summarizer_model: Optional Hugging Face model for summarization
        """
        self.pdf_path = pdf_path
        self.indexer = PDFIndexer(pdf_path)
        self.classifier = LineClassifier(classifier_model) if classifier_model else LineClassifier()
        self.hierarchy_builder = HierarchyBuilder()
        self.content_mapper = None  # Will be initialized after indexing
        self.summarizer = ContentSummarizer(summarizer_model) if summarizer_model else ContentSummarizer()
        self.validator = QualityValidator()
        
        self.indexed_lines: List[Dict] = []
        self.classified_lines: List[Dict] = []
        self.hierarchy: List[Dict] = []
        self.content_overview: List[Dict] = []
    
    def process_document(self) -> Dict:
        """
        Process the entire document through all stages
        
        Returns:
            Dictionary containing all processing results
        """
        print("📄 Step 1: Indexing document by page and line number...")
        self.indexed_lines = self.indexer.index_document()
        print(f"   ✓ Indexed {len(self.indexed_lines)} lines")
        
        print("\n🤖 Step 2: Classifying lines (heading/subheading/content)...")
        self.classified_lines = self.classifier.classify_lines(self.indexed_lines)
        print(f"   ✓ Classified {len(self.classified_lines)} lines")
        
        # Count classifications
        heading_count = sum(1 for line in self.classified_lines if line.get('classification') == 'heading')
        subheading_count = sum(1 for line in self.classified_lines if line.get('classification') == 'subheading')
        content_count = sum(1 for line in self.classified_lines if line.get('classification') == 'content')
        print(f"   - Headings: {heading_count}, Subheadings: {subheading_count}, Content: {content_count}")
        
        print("\n🏗️  Step 3: Building hierarchical structure...")
        self.hierarchy = self.hierarchy_builder.build_hierarchy(self.classified_lines)
        print(f"   ✓ Built hierarchy with {len(self.hierarchy)} headings")
        
        # Initialize content mapper after indexing (pass classified lines to filter headings/subheadings)
        self.content_mapper = ContentMapper(self.indexer, self.classified_lines)
        
        print("\n📋 Step 4: Creating content overview...")
        self.content_overview = self.hierarchy_builder.get_content_overview()
        print(f"   ✓ Created content overview with {len(self.content_overview)} entries")
        
        # Step 5: Quality validation
        print("\n✅ Step 5: Validating quality...")
        hierarchy_validation = self.validator.validate_hierarchy(self.hierarchy)
        coverage_validation = self.validator.validate_content_coverage(self.indexed_lines, self.hierarchy)
        
        # Log detected headings with confidence
        logger.info("\n📊 Detected Headings:")
        for heading in self.hierarchy:
            self.validator.log_heading_detection(heading)
            for subheading in heading.get('subheadings', []):
                self.validator.log_heading_detection(subheading)
        
        validation_report = self.validator.get_validation_report()
        print(f"   ✓ Validation complete: {validation_report['hierarchy_issues']} hierarchy issues, "
              f"{validation_report['content_gaps']} content gaps")
        
        return {
            'indexed_lines': len(self.indexed_lines),
            'classified_lines': len(self.classified_lines),
            'headings': len(self.hierarchy),
            'content_overview': self.content_overview,
            'validation': validation_report
        }
    
    def get_content_overview(self) -> List[Dict]:
        """
        Get the content overview (table of contents)
        
        Returns:
            List of headings with their positions and subheadings
        """
        return self.content_overview
    
    def summarize_subheading(self, subheading_text: str) -> Optional[str]:
        """
        Get summary for a specific subheading
        
        Args:
            subheading_text: Text of the subheading to summarize
            
        Returns:
            Summary text or None if subheading not found
        """
        subheading = self.hierarchy_builder.find_subheading(subheading_text)
        
        if not subheading:
            return None
        
        # Find the next subheading to set proper boundaries
        next_subheading = None
        for h in self.hierarchy:
            subheadings = h.get('subheadings', [])
            for i, sub in enumerate(subheadings):
                if sub['text'] == subheading_text:
                    # Check if there's a next subheading in the same heading
                    if i + 1 < len(subheadings):
                        next_subheading = subheadings[i + 1]
                    break
            if next_subheading:
                break
        
        # Get content for the subheading with proper boundaries (now returns dict)
        content_data = self.content_mapper.get_content_for_subheading(subheading, next_subheading)
        
        if not content_data or not content_data.get('text') or len(content_data.get('text', '').strip()) < 20:
            # Try to get more context by looking at the heading's content
            heading = None
            for h in self.hierarchy:
                for sub in h.get('subheadings', []):
                    if sub['text'] == subheading_text:
                        heading = h
                        break
                if heading:
                    break
            
            if heading:
                # Get content from the parent heading
                heading_content = self.content_mapper.get_content_for_heading(heading)
                if heading_content and len(heading_content.strip()) >= 10:
                    # Use heading content as fallback
                    content_data = {'text': heading_content, 'metadata': {}}
                else:
                    return "No content available for this subheading. The subheading may be a label without associated content."
            else:
                return "No content available for this subheading. The subheading may be a label without associated content."
        
        # Validate content before summarizing
        content_text = content_data.get('text', '')
        if not content_text or len(content_text.strip()) < 30:
            return "No sufficient content available for this subheading. The subheading may not have associated paragraph content."
        
        # Make sure content doesn't start with the subheading itself
        content_cleaned = content_text.strip()
        if content_cleaned.lower().startswith(subheading_text.lower()):
            # Remove the subheading from the start
            content_cleaned = content_cleaned[len(subheading_text):].strip()
            if content_cleaned.startswith(':'):
                content_cleaned = content_cleaned[1:].strip()
        
        # Summarize the content (now returns dict with metadata)
        logger.info(f"Summarizing content for: {subheading_text}")
        summary_result = self.summarizer.summarize_subheading(content_data, subheading_text, content_cleaned)
        
        # Extract summary text from result
        if isinstance(summary_result, dict):
            summary = summary_result.get('summary', '')
            method = summary_result.get('method', 'unknown')
            verified = summary_result.get('verified', False)
            
            # Log summary generation info
            logger.info(f"   Method: {method}, Verified: {verified}")
            if summary_result.get('metadata'):
                metadata = summary_result['metadata']
                logger.info(f"   Source: Page {metadata.get('start_page')}-{metadata.get('end_page')}, "
                          f"Lines {metadata.get('start_line')}-{metadata.get('end_line')}")
            
            # Validate summary against source
            validation = self.validator.validate_summary(summary, content_cleaned, subheading_text)
            logger.info(f"   Summary validation: {validation['verification_score']:.2f} "
                      f"({validation['verified_sentences']}/{validation['total_sentences']} sentences verified)")
        else:
            summary = summary_result
        
        # Final validation: summary should not be the same as subheading
        if summary and summary.strip().lower() == subheading_text.lower():
            # If summary is just the subheading, extract sentences from content
            sentences = [s.strip() for s in content_cleaned.split('.') if len(s.strip()) > 30]
            if sentences:
                summary = '. '.join(sentences[:2]) + '.' if len(sentences) >= 2 else sentences[0] + '.'
            else:
                summary = content_cleaned[:200] + "..."
        
        return summary
    
    def summarize_heading(self, heading_text: str) -> Optional[str]:
        """
        Get summary for a specific heading
        
        Args:
            heading_text: Text of the heading to summarize
            
        Returns:
            Summary text or None if heading not found
        """
        heading = self.hierarchy_builder.find_heading(heading_text)
        
        if not heading:
            return None
        
        # Get content for the heading
        content = self.content_mapper.get_content_for_heading(heading)
        
        if not content or len(content.strip()) < 30:
            return "No sufficient content available for this heading."
        
        # Make sure content doesn't start with the heading itself
        content_cleaned = content.strip()
        if content_cleaned.lower().startswith(heading_text.lower()):
            # Remove the heading from the start
            content_cleaned = content_cleaned[len(heading_text):].strip()
            if content_cleaned.startswith(':'):
                content_cleaned = content_cleaned[1:].strip()
        
        # Summarize the content
        print(f"\n📝 Summarizing content for: {heading_text}")
        summary = self.summarizer.summarize(content_cleaned, max_length=300, min_length=100)
        
        # Final validation: summary should not be the same as heading
        if summary and summary.strip().lower() == heading_text.lower():
            # If summary is just the heading, extract sentences from content
            sentences = [s.strip() for s in content_cleaned.split('.') if len(s.strip()) > 30]
            if sentences:
                summary = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else '. '.join(sentences) + '.'
            else:
                summary = content_cleaned[:300] + "..."
        
        return summary
    
    def print_content_overview(self):
        """Print a formatted content overview"""
        print("\n" + "="*80)
        print("CONTENT OVERVIEW (Table of Contents)")
        print("="*80)
        
        for i, entry in enumerate(self.content_overview, 1):
            print(f"\n{i}. {entry['heading']}")
            print(f"   Page {entry['page']}, Line {entry['line']}")
            print(f"   Range: Page {entry['start_page']}-{entry['end_page']}, "
                  f"Lines {entry['start_line']}-{entry['end_line']}")
            
            if entry['subheadings']:
                print(f"   Subheadings ({len(entry['subheadings'])}):")
                for j, subheading in enumerate(entry['subheadings'], 1):
                    print(f"      {i}.{j} {subheading['subheading']}")
                    print(f"         Page {subheading['page']}, Line {subheading['line']}")
                    print(f"         Range: Page {subheading['start_page']}-{subheading['end_page']}, "
                          f"Lines {subheading['start_line']}-{subheading['end_line']}")
    
    def save_results(self, output_file: str = "processing_results.json"):
        """
        Save processing results to a JSON file
        
        Args:
            output_file: Path to output JSON file
        """
        results = {
            'pdf_path': self.pdf_path,
            'statistics': {
                'total_lines': len(self.indexed_lines),
                'headings': len(self.hierarchy),
                'total_subheadings': sum(len(h.get('subheadings', [])) for h in self.hierarchy)
            },
            'content_overview': self.content_overview,
            'hierarchy': self.hierarchy
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {output_file}")
    
    def close(self):
        """Close the PDF document"""
        self.indexer.close()


def main():
    """Example usage of the PDF Document Processor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path>")
        print("Example: python main.py document.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    # Initialize processor
    processor = PDFDocumentProcessor(pdf_path)
    
    try:
        # Process document
        results = processor.process_document()
        
        # Print content overview
        processor.print_content_overview()
        
        # Save results
        processor.save_results()
        
        # Example: Summarize a subheading (if available)
        content_overview = processor.get_content_overview()
        if content_overview and content_overview[0].get('subheadings'):
            first_subheading = content_overview[0]['subheadings'][0]
            summary = processor.summarize_subheading(first_subheading['subheading'])
            if summary:
                print(f"\n📄 Summary for '{first_subheading['subheading']}':")
                print(f"   {summary}")
        
    except Exception as e:
        print(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.close()


if __name__ == "__main__":
    main()

