"""
Main Application: PDF Document Processor with ML/DL Classification and Summarization
"""
import json
import logging
from .pdf_indexer import PDFIndexer
from .line_classifier import LineClassifier
from .hierarchy_builder import HierarchyBuilder
from .content_mapper import ContentMapper
from .summarizer import ContentSummarizer
from .quality_validator import QualityValidator
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
        logger.info("📄 Step 1: Indexing document by page and line number...")
        self.indexed_lines = self.indexer.index_document()
        logger.info(f"   ✓ Indexed {len(self.indexed_lines)} lines")
        
        logger.info("\n🤖 Step 2: Classifying lines (heading/subheading/content)...")
        self.classified_lines = self.classifier.classify_lines(self.indexed_lines)
        logger.info(f"   ✓ Classified {len(self.classified_lines)} lines")
        
        # Count classifications
        heading_count = sum(1 for line in self.classified_lines if line.get('classification') == 'heading')
        subheading_count = sum(1 for line in self.classified_lines if line.get('classification') == 'subheading')
        content_count = sum(1 for line in self.classified_lines if line.get('classification') == 'content')
        logger.info(f"   - Headings: {heading_count}, Subheadings: {subheading_count}, Content: {content_count}")
        
        logger.info("\n🏗️  Step 3: Building hierarchical structure...")
        self.hierarchy = self.hierarchy_builder.build_hierarchy(self.classified_lines)
        logger.info(f"   ✓ Built hierarchy with {len(self.hierarchy)} headings")
        
        # Initialize content mapper after indexing (pass classified lines to filter headings/subheadings)
        self.content_mapper = ContentMapper(self.indexer, self.classified_lines)
        
        logger.info("\n📋 Step 4: Creating content overview...")
        self.content_overview = self.hierarchy_builder.get_content_overview()
        logger.info(f"   ✓ Created content overview with {len(self.content_overview)} entries")
        
        # Step 5: Quality validation
        logger.info("\n✅ Step 5: Validating quality...")
        hierarchy_validation = self.validator.validate_hierarchy(self.hierarchy)
        coverage_validation = self.validator.validate_content_coverage(self.indexed_lines, self.hierarchy)
        
        # Log detected headings with confidence
        logger.info("\n📊 Detected Headings:")
        for heading in self.hierarchy:
            self.validator.log_heading_detection(heading)
            for subheading in heading.get('subheadings', []):
                self.validator.log_heading_detection(subheading)
        
        validation_report = self.validator.get_validation_report()
        logger.info(f"   ✓ Validation complete: {validation_report['hierarchy_issues']} hierarchy issues, "
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
    
    def get_all_text(self) -> str:
        """
        Get all text content from the PDF (excluding headings/subheadings structure)
        Uses multiple methods to ensure comprehensive text extraction
        
        Returns:
            Combined text content from all content lines
        """
        try:
            all_content_parts = []
            
            # Method 1: Get from classified content lines (preferred - filters out headings)
            if hasattr(self, 'classified_lines') and self.classified_lines:
                content_lines = [
                    line.get('text', '') 
                    for line in self.classified_lines 
                    if line.get('classification') == 'content' and line.get('text', '').strip()
                ]
                if content_lines:
                    all_content_parts.extend(content_lines)
                    logger.info(f"Extracted {len(content_lines)} content lines from classified_lines")
            
            # Method 2: Get from content mapper using all headings (more comprehensive)
            if hasattr(self, 'content_mapper') and self.content_mapper and hasattr(self, 'hierarchy') and self.hierarchy:
                try:
                    for heading in self.hierarchy:
                        heading_content = self.content_mapper.get_content_for_heading(heading)
                        if heading_content and len(heading_content.strip()) > 20:
                            # Split into sentences to avoid duplicates
                            sentences = heading_content.split('.')
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if sentence and len(sentence) > 20:
                                    # Check if not already added
                                    if sentence not in all_content_parts:
                                        all_content_parts.append(sentence)
                    logger.info(f"Extracted content from {len(self.hierarchy)} headings via content_mapper")
                except Exception as mapper_err:
                    logger.warning(f"Error getting content from mapper: {mapper_err}")
            
            # Method 3: Fallback to indexed lines if needed
            if not all_content_parts and hasattr(self, 'indexed_lines') and self.indexed_lines:
                logger.warning("Using indexed_lines as fallback")
                content_lines = [
                    line.get('text', '') 
                    for line in self.indexed_lines 
                    if line.get('text', '').strip() and len(line.get('text', '').strip()) > 10
                ]
                all_content_parts.extend(content_lines)
            
            # Combine all content, removing duplicates while preserving order
            if all_content_parts:
                # Join with spaces and clean up
                all_text = ' '.join(all_content_parts)
                # Remove excessive whitespace
                import re
                all_text = re.sub(r'\s+', ' ', all_text).strip()
                logger.info(f"Total extracted text: {len(all_text)} characters from {len(all_content_parts)} parts")
                return all_text
            
            return ""
            
        except Exception as e:
            logger.error(f"Error in get_all_text: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Fallback: try to get text from indexed lines
            try:
                if hasattr(self, 'indexed_lines') and self.indexed_lines:
                    content_lines = [line.get('text', '') for line in self.indexed_lines if line.get('text', '').strip()]
                    return ' '.join(content_lines)
            except:
                pass
            return ""
    
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
        logger.info(f"\n📝 Summarizing content for: {heading_text}")
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
    
    def close(self):
        """Close the PDF document"""
        self.indexer.close()
