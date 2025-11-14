"""
Hierarchy Builder: Automatically builds structured hierarchy of headings and subheadings
"""
from typing import List, Dict, Optional


class HierarchyBuilder:
    """Builds hierarchical structure from classified lines"""
    
    def __init__(self):
        self.hierarchy: List[Dict] = []
    
    def build_hierarchy(self, classified_lines: List[Dict]) -> List[Dict]:
        """
        Build hierarchical structure from classified lines
        
        Args:
            classified_lines: List of lines with classification
            
        Returns:
            List of hierarchical structure with headings and subheadings
        """
        self.hierarchy = []
        current_heading = None
        current_subheading = None
        
        for i, line in enumerate(classified_lines):
            previous_line = classified_lines[i - 1] if i > 0 else None
            classification = line.get('classification', 'content')
            
            if classification == 'heading':
                # Close previous subheading if it exists
                if current_subheading and previous_line:
                    current_subheading['end_page'] = previous_line['page']
                    current_subheading['end_line'] = previous_line['line_number']
                current_subheading = None

                # Close previous heading if it exists
                if current_heading and previous_line:
                    current_heading['end_page'] = previous_line['page']
                    current_heading['end_line'] = previous_line['line_number']

                # Start a new heading
                current_heading = {
                    'type': 'heading',
                    'text': line['text'],
                    'page': line['page'],
                    'line_number': line['line_number'],
                    'start_page': line['page'],
                    'start_line': line['line_number'],
                    'end_page': None,
                    'end_line': None,
                    'subheadings': [],
                    'content_start_page': line['page'],
                    'content_start_line': line['line_number'] + 1,
                }
                current_subheading = None
                self.hierarchy.append(current_heading)
            
            elif classification == 'subheading':
                # Start a new subheading under current heading
                if current_heading is None:
                    # If no heading exists, treat as heading
                    current_heading = {
                        'type': 'heading',
                        'text': line['text'],
                        'page': line['page'],
                        'line_number': line['line_number'],
                        'start_page': line['page'],
                        'start_line': line['line_number'],
                        'end_page': None,
                        'end_line': None,
                        'subheadings': [],
                        'content_start_page': line['page'],
                        'content_start_line': line['line_number'] + 1,
                    }
                    self.hierarchy.append(current_heading)
                else:
                    # Close previous subheading if exists
                    if current_subheading:
                        # If previous line exists and is content, use it as end
                        if previous_line and previous_line.get('classification') == 'content':
                            current_subheading['end_page'] = previous_line['page']
                            current_subheading['end_line'] = previous_line['line_number']
                        else:
                            # No content between subheadings, end at the line before this subheading
                            current_subheading['end_page'] = line['page']
                            current_subheading['end_line'] = max(line['line_number'] - 1, current_subheading['start_line'])
                    
                    # Create new subheading
                    current_subheading = {
                        'type': 'subheading',
                        'text': line['text'],
                        'page': line['page'],
                        'line_number': line['line_number'],
                        'start_page': line['page'],
                        'start_line': line['line_number'],
                        'end_page': None,
                        'end_line': None,
                        'content_start_page': line['page'],
                        'content_start_line': line['line_number'] + 1,
                    }
                    current_heading['subheadings'].append(current_subheading)
            
            elif classification == 'content':
                # Content line - update end positions
                if current_subheading:
                    current_subheading['end_page'] = line['page']
                    current_subheading['end_line'] = line['line_number']
                elif current_heading:
                    current_heading['end_page'] = line['page']
                    current_heading['end_line'] = line['line_number']
        
        # Close any open headings/subheadings
        if current_subheading and current_subheading['end_page'] is None:
            if classified_lines:
                last_line = classified_lines[-1]
                current_subheading['end_page'] = last_line['page']
                current_subheading['end_line'] = last_line['line_number']
        
        if current_heading and current_heading['end_page'] is None:
            if classified_lines:
                last_line = classified_lines[-1]
                current_heading['end_page'] = last_line['page']
                current_heading['end_line'] = last_line['line_number']
        
        return self.hierarchy
    
    def get_content_overview(self) -> List[Dict]:
        """
        Get content overview (table of contents structure)
        
        Returns:
            List of headings with their positions and subheadings
        """
        overview = []
        
        for heading in self.hierarchy:
            heading_info = {
                'heading': heading['text'],
                'page': heading['page'],
                'line': heading['line_number'],
                'start_page': heading['start_page'],
                'start_line': heading['start_line'],
                'end_page': heading['end_page'],
                'end_line': heading['end_line'],
                'subheadings': []
            }
            
            for subheading in heading.get('subheadings', []):
                subheading_info = {
                    'subheading': subheading['text'],
                    'page': subheading['page'],
                    'line': subheading['line_number'],
                    'start_page': subheading['start_page'],
                    'start_line': subheading['start_line'],
                    'end_page': subheading['end_page'],
                    'end_line': subheading['end_line'],
                }
                heading_info['subheadings'].append(subheading_info)
            
            overview.append(heading_info)
        
        return overview
    
    def find_subheading(self, subheading_text: str) -> Optional[Dict]:
        """
        Find a subheading by text
        
        Args:
            subheading_text: Text of the subheading to find
            
        Returns:
            Subheading dictionary or None
        """
        for heading in self.hierarchy:
            for subheading in heading.get('subheadings', []):
                if subheading['text'] == subheading_text:
                    return subheading
        return None
    
    def find_heading(self, heading_text: str) -> Optional[Dict]:
        """
        Find a heading by text
        
        Args:
            heading_text: Text of the heading to find
            
        Returns:
            Heading dictionary or None
        """
        for heading in self.hierarchy:
            if heading['text'] == heading_text:
                return heading
        return None

