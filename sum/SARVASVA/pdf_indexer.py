"""
PDF Indexer: Breaks down PDF documents by page and line number with font features
"""
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional
import re


class PDFIndexer:
    """Indexes PDF documents by page and line number with enhanced font features"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF indexer
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.indexed_lines: List[Dict] = []
    
    def _extract_font_features(self, page: fitz.Page, line_text: str, line_num: int) -> Dict:
        """
        Extract font features for a line from the PDF
        
        Args:
            page: PyMuPDF page object
            line_text: The text of the line
            line_num: Line number on the page
            
        Returns:
            Dictionary with font features
        """
        features = {
            'font_size': 0.0,
            'is_bold': False,
            'font_name': '',
            'spacing_before': 0.0,
            'spacing_after': 0.0,
            'indentation': 0.0,
            'line_height': 0.0,
        }
        
        try:
            # Get text blocks with formatting information
            blocks = page.get_text("dict")
            
            # Find the block containing this line
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line_block in block["lines"]:
                        for span in line_block.get("spans", []):
                            span_text = span.get("text", "").strip()
                            if line_text.strip() in span_text or span_text in line_text:
                                # Extract font features
                                features['font_size'] = span.get("size", 0.0)
                                features['font_name'] = span.get("font", "")
                                features['is_bold'] = "bold" in features['font_name'].lower() or span.get("flags", 0) & 16
                                
                                # Get bounding box for spacing and indentation
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                if bbox:
                                    features['indentation'] = bbox[0]  # x0 position
                                    features['line_height'] = bbox[3] - bbox[1]  # height
                                
                                return features
        except Exception as e:
            # If font extraction fails, use defaults
            pass
        
        return features
    
    def index_document(self) -> List[Dict]:
        """
        Index the entire document by page and line number with font features
        
        Returns:
            List of dictionaries containing page, line_number, text, and font features
        """
        self.indexed_lines = []
        previous_line_features = None
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Split text into lines
            lines = text.split('\n')
            
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if line:  # Only index non-empty lines
                    # Extract font features
                    font_features = self._extract_font_features(page, line, line_num)
                    
                    # Calculate spacing
                    spacing_before = 0.0
                    if previous_line_features:
                        spacing_before = abs(font_features.get('line_height', 0) - previous_line_features.get('line_height', 0))
                    
                    line_data = {
                        'page': page_num + 1,  # 1-indexed
                        'line_number': line_num,
                        'text': line,
                        'original_text': line,
                        'font_size': font_features.get('font_size', 0.0),
                        'is_bold': font_features.get('is_bold', False),
                        'font_name': font_features.get('font_name', ''),
                        'indentation': font_features.get('indentation', 0.0),
                        'line_height': font_features.get('line_height', 0.0),
                        'spacing_before': spacing_before,
                    }
                    
                    self.indexed_lines.append(line_data)
                    previous_line_features = font_features
        
        return self.indexed_lines
    
    def get_line_by_index(self, page: int, line_number: int) -> Dict:
        """
        Get a specific line by page and line number
        
        Args:
            page: Page number (1-indexed)
            line_number: Line number on the page
            
        Returns:
            Dictionary with line information
        """
        for line in self.indexed_lines:
            if line['page'] == page and line['line_number'] == line_number:
                return line
        return None
    
    def get_lines_in_range(self, start_page: int, start_line: int, 
                          end_page: int, end_line: int) -> List[str]:
        """
        Get all lines in a specified range
        
        Args:
            start_page: Starting page number
            start_line: Starting line number
            end_page: Ending page number
            end_line: Ending line number
            
        Returns:
            List of text lines in the range
        """
        content = []
        
        # Handle None values
        if start_page is None or start_line is None:
            return []
        if end_page is None:
            end_page = start_page
        if end_line is None:
            end_line = start_line
        
        for line in self.indexed_lines:
            page = line['page']
            line_num = line['line_number']
            
            # Check if line is in range
            in_range = False
            
            if page == start_page == end_page:
                # Same page: check if line is between start and end
                in_range = start_line <= line_num <= end_line
            elif page == start_page:
                # Starting page: line must be >= start_line
                in_range = line_num >= start_line
            elif page == end_page:
                # Ending page: line must be <= end_line
                in_range = line_num <= end_line
            elif start_page < page < end_page:
                # Middle pages: include all lines
                in_range = True
            
            if in_range:
                content.append(line['text'])
        
        return content
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()

