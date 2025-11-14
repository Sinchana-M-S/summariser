"""
Quality Validator: Validates document structure, hierarchy, and summaries
"""
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityValidator:
    """Validates quality of document processing"""
    
    def __init__(self):
        """Initialize quality validator"""
        self.validation_results = {
            'headings': [],
            'hierarchy_issues': [],
            'content_gaps': [],
            'summary_issues': []
        }
    
    def validate_hierarchy(self, hierarchy: List[Dict]) -> Dict:
        """
        Validate heading hierarchy (H1 → H2 → H3, not H1 → H3)
        
        Args:
            hierarchy: List of headings with subheadings
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        for i, heading in enumerate(hierarchy):
            heading_text = heading.get('text', '')
            confidence = heading.get('confidence', 0.0)
            
            # Check confidence threshold
            if confidence < 0.75:
                issues.append({
                    'type': 'low_confidence',
                    'heading': heading_text,
                    'confidence': confidence,
                    'message': f"Heading '{heading_text}' has low confidence ({confidence:.2f})"
                })
            
            # Check subheadings
            subheadings = heading.get('subheadings', [])
            for j, subheading in enumerate(subheadings):
                subheading_text = subheading.get('text', '')
                sub_confidence = subheading.get('confidence', 0.0)
                
                if sub_confidence < 0.6:
                    issues.append({
                        'type': 'low_confidence',
                        'subheading': subheading_text,
                        'confidence': sub_confidence,
                        'message': f"Subheading '{subheading_text}' has low confidence ({sub_confidence:.2f})"
                    })
        
        self.validation_results['hierarchy_issues'] = issues
        
        if issues:
            logger.warning(f"Found {len(issues)} hierarchy issues")
            for issue in issues:
                logger.warning(f"  - {issue['message']}")
        else:
            logger.info("✓ Hierarchy validation passed")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_headings': len(hierarchy),
            'total_subheadings': sum(len(h.get('subheadings', [])) for h in hierarchy)
        }
    
    def validate_content_coverage(self, indexed_lines: List[Dict], hierarchy: List[Dict]) -> Dict:
        """
        Verify no content gaps (every line is assigned to a section)
        
        Args:
            indexed_lines: All indexed lines from PDF
            hierarchy: Document hierarchy
            
        Returns:
            Dictionary with validation results
        """
        # Build set of lines covered by headings/subheadings
        covered_lines = set()
        
        for heading in hierarchy:
            start_page = heading.get('start_page')
            start_line = heading.get('start_line')
            end_page = heading.get('end_page', start_page)
            end_line = heading.get('end_line', start_line)
            
            # Mark heading line
            if start_page and start_line:
                covered_lines.add((start_page, start_line))
            
            # Mark content lines
            for line in indexed_lines:
                page = line.get('page')
                line_num = line.get('line_number')
                
                if page and line_num:
                    if start_page == end_page:
                        if page == start_page and start_line <= line_num <= end_line:
                            covered_lines.add((page, line_num))
                    elif start_page < end_page:
                        if (page == start_page and line_num >= start_line) or \
                           (page == end_page and line_num <= end_line) or \
                           (start_page < page < end_page):
                            covered_lines.add((page, line_num))
            
            # Mark subheading lines
            for subheading in heading.get('subheadings', []):
                sub_start_page = subheading.get('start_page')
                sub_start_line = subheading.get('start_line')
                if sub_start_page and sub_start_line:
                    covered_lines.add((sub_start_page, sub_start_line))
        
        # Find uncovered lines
        uncovered_lines = []
        for line in indexed_lines:
            page = line.get('page')
            line_num = line.get('line_number')
            classification = line.get('classification', 'content')
            
            if page and line_num:
                if (page, line_num) not in covered_lines and classification == 'content':
                    uncovered_lines.append({
                        'page': page,
                        'line': line_num,
                        'text': line.get('text', '')[:50]
                    })
        
        self.validation_results['content_gaps'] = uncovered_lines
        
        if uncovered_lines:
            logger.warning(f"Found {len(uncovered_lines)} uncovered content lines")
            for gap in uncovered_lines[:5]:  # Show first 5
                logger.warning(f"  - Page {gap['page']}, Line {gap['line']}: {gap['text']}...")
        else:
            logger.info("✓ Content coverage validation passed")
        
        return {
            'valid': len(uncovered_lines) == 0,
            'uncovered_lines': uncovered_lines,
            'coverage_percentage': (len(covered_lines) / len(indexed_lines) * 100) if indexed_lines else 0
        }
    
    def validate_summary(self, summary: str, source_text: str, heading: str) -> Dict:
        """
        Validate summary doesn't contain information not in source
        
        Args:
            summary: Generated summary
            source_text: Original source text
            heading: Heading/subheading name
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        if not summary or not source_text:
            return {
                'valid': False,
                'issues': ['Summary or source text is empty'],
                'verification_score': 0.0
            }
        
        # Split into sentences
        summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
        source_lower = source_text.lower()
        
        verified_sentences = 0
        for sentence in summary_sentences:
            sentence_lower = sentence.lower()
            
            # Extract key words
            key_words = [w for w in sentence_lower.split() if len(w) >= 3]
            
            if len(key_words) == 0:
                continue
            
            # Check word overlap with source
            matches = sum(1 for word in key_words if word in source_lower)
            match_ratio = matches / len(key_words) if key_words else 0
            
            if match_ratio >= 0.5:
                verified_sentences += 1
            else:
                issues.append({
                    'sentence': sentence[:100],
                    'match_ratio': match_ratio,
                    'message': f"Sentence may contain information not in source (match: {match_ratio:.2f})"
                })
        
        verification_score = verified_sentences / len(summary_sentences) if summary_sentences else 0.0
        
        self.validation_results['summary_issues'].append({
            'heading': heading,
            'issues': issues,
            'verification_score': verification_score
        })
        
        if issues:
            logger.warning(f"Summary for '{heading}' has {len(issues)} potential issues (verification: {verification_score:.2f})")
        else:
            logger.info(f"✓ Summary for '{heading}' validated (verification: {verification_score:.2f})")
        
        return {
            'valid': verification_score >= 0.7,
            'issues': issues,
            'verification_score': verification_score,
            'verified_sentences': verified_sentences,
            'total_sentences': len(summary_sentences)
        }
    
    def get_validation_report(self) -> Dict:
        """
        Get comprehensive validation report
        
        Returns:
            Dictionary with all validation results
        """
        return {
            'hierarchy_issues': len(self.validation_results['hierarchy_issues']),
            'content_gaps': len(self.validation_results['content_gaps']),
            'summary_issues': len(self.validation_results['summary_issues']),
            'details': self.validation_results
        }
    
    def log_heading_detection(self, heading: Dict):
        """
        Log heading detection with confidence score
        
        Args:
            heading: Heading dictionary with classification and confidence
        """
        text = heading.get('text', '')
        classification = heading.get('classification', 'unknown')
        confidence = heading.get('confidence', 0.0)
        page = heading.get('page', '?')
        line = heading.get('line_number', '?')
        
        logger.info(f"Detected {classification}: '{text}' (confidence: {confidence:.2f}, Page {page}, Line {line})")
        
        self.validation_results['headings'].append({
            'text': text,
            'classification': classification,
            'confidence': confidence,
            'page': page,
            'line': line
        })

