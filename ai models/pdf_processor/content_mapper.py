"""
Content Mapper: Maps headings and subheadings to their start and end positions
"""
from typing import List, Dict, Optional
import re
from .pdf_indexer import PDFIndexer


class ContentMapper:
    """Maps content to positions in the document"""
    
    def __init__(self, pdf_indexer: PDFIndexer, classified_lines: Optional[List[Dict]] = None):
        """
        Initialize content mapper
        
        Args:
            pdf_indexer: PDFIndexer instance
            classified_lines: Optional list of classified lines to filter out headings/subheadings
        """
        self.pdf_indexer = pdf_indexer
        self.classified_lines = classified_lines or []
    
    def get_content_for_subheading(self, subheading: Dict, next_subheading: Optional[Dict] = None) -> Dict:
        """
        Get EXACT content text for a subheading with metadata
        
        Args:
            subheading: Subheading dictionary with start/end positions
            next_subheading: Optional next subheading to stop at
            
        Returns:
            Dictionary with 'text', 'start_page', 'start_line', 'end_page', 'end_line'
        """
        start_page = subheading.get('content_start_page', subheading.get('start_page'))
        start_line = subheading.get('content_start_line', subheading.get('start_line', 1) + 1)
        end_page = subheading.get('end_page')
        end_line = subheading.get('end_line')
        
        # Make sure we have valid start positions
        if start_page is None or start_line is None:
            start_page = subheading.get('start_page')
            start_line = subheading.get('start_line', 1) + 1
            if start_page is None:
                return {
                    'text': '',
                    'start_page': None,
                    'start_line': None,
                    'end_page': None,
                    'end_line': None
                }
        
        # If end positions are not set, use next subheading as boundary
        if end_page is None or end_line is None or \
           (start_page > end_page) or (start_page == end_page and start_line > end_line):
            # If we have a next subheading, stop before it
            if next_subheading:
                end_page = next_subheading.get('start_page', start_page)
                end_line = next_subheading.get('start_line', start_line) - 1
                if end_line < start_line:
                    end_line = start_line + 50  # Fallback: reasonable limit
            else:
                # Look ahead only a reasonable amount (50 lines max)
                end_page = start_page
                end_line = min(start_line + 50, 9999)
        
        # Get EXACT lines with metadata
        exact_lines = []
        for line in self.pdf_indexer.indexed_lines:
            page = line['page']
            line_num = line['line_number']
            
            # Check if line is in range
            in_range = False
            if page == start_page == end_page:
                in_range = start_line <= line_num <= end_line
            elif page == start_page:
                in_range = line_num >= start_line
            elif page == end_page:
                in_range = line_num <= end_line
            elif start_page < page < end_page:
                in_range = True
            
            if in_range:
                exact_lines.append(line)
        
        lines = [line['text'] for line in exact_lines]
        
        # Create a map of text to classification for quick lookup (optimized)
        text_classification_map = {}
        if self.classified_lines:
            for cl_line in self.classified_lines:
                text_key = cl_line.get('text', '').strip().lower()
                text_classification_map[text_key] = cl_line.get('classification', 'content')
        
        # Filter out empty lines, headings, subheadings, and lines that look like headings/subheadings
        content_lines = []
        social_media_keywords = [
            'share your thoughts', 'post a video', 'facebook', 'twitter', 'instagram',
            'comment section', 'comment below', 'share your video', 'share a video',
            'we would like to hear', 'tell us about', 'post on', 'follow us',
            'like and share', 'subscribe', 'click here', 'visit our', 'follow our',
            'comment', 'comments', 'share your', 'post your', 'upload your'
        ]
        
        for line_text in lines:
            line_stripped = line_text.strip()
            line_lower = line_stripped.lower()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # STRICT FILTER: Exclude social media prompts and irrelevant content
            if any(keyword in line_lower for keyword in social_media_keywords):
                break  # Stop extraction - we've hit irrelevant content
            
            # Filter out URLs, emails
            if '@' in line_stripped or 'http' in line_lower or 'www.' in line_lower:
                continue
            
            # Filter out copyright, page numbers in short lines
            if len(line_stripped.split()) <= 5:
                if any(phrase in line_lower for phrase in ['copyright', '©', 'page', 'all rights reserved']):
                    continue
            
            # Find the classification for this line (optimized lookup)
            text_key = line_lower
            line_classification = text_classification_map.get(text_key, 'content')
            
            # Stop if we hit another heading or subheading (boundary reached)
            if line_classification in ('heading', 'subheading'):
                break  # Stop extraction, we've hit a boundary
            
            # Also check if this line looks like a subheading pattern (numbered list)
            if re.match(r'^\d+[\.\)]\s*[A-Z]', line_stripped) and len(line_stripped) <= 30:
                # Likely another subheading, stop here
                break
            
            # Skip very short lines (likely headings/subheadings)
            if len(line_stripped) <= 5:
                continue
            
            # Skip lines that are just numbers, single characters, or number patterns
            if line_stripped.isdigit() or len(line_stripped) == 1:
                continue
            
            # Skip lines that match numbered list patterns (e.g., "1.", "2.", "3.")
            if re.match(r'^\d+[\.\)]\s*$', line_stripped) or re.match(r'^\d+[\.\)]\s*[A-Z]', line_stripped):
                # This might be a subheading, skip if it's very short
                if len(line_stripped) <= 15:
                    continue
            
            # Skip lines that are all uppercase and short (likely headings)
            if line_stripped.isupper() and len(line_stripped) <= 15:
                continue
            
            # Skip lines that look like names (2-4 words, title case, no punctuation)
            words = line_stripped.split()
            if 2 <= len(words) <= 4:
                if all(word[0].isupper() if word else False for word in words):
                    if not any(p in line_stripped for p in ['.', '!', '?', ',', ':', ';', '-']):
                        if len(line_stripped) < 30:
                            continue  # Likely a name, skip it
            
            # Skip lines that look like titles (title case, short, no punctuation)
            if len(words) <= 5 and line_stripped[0].isupper() and not line_stripped.endswith(('.', '!', '?', ':', ';')):
                # Might be a heading, but include if it's long enough
                if len(line_stripped) < 30:
                    continue
            
            # Prefer lines that look like actual content (have punctuation, are longer)
            # Must be substantial content (at least 20 chars, preferably with punctuation)
            if len(line_stripped) >= 20:
                # STRICT: Only include lines that are clearly sentences/content
                # Must have punctuation OR be long enough to be a sentence
                has_punctuation = any(p in line_stripped for p in ['.', ',', ';', '!', '?'])
                is_long_enough = len(line_stripped) >= 40
                
                # Exclude if it's just a heading-like phrase
                if has_punctuation or is_long_enough:
                    # Make sure it's not just a heading with a period
                    if has_punctuation or len(line_stripped) >= 50:
                        content_lines.append(line_text)
        
        # If we found content, return it (limit to reasonable amount)
        if content_lines:
            # Take only the first substantial content (stop at next subheading/heading)
            final_content = []
            for line in content_lines:
                # Stop if we hit another heading/subheading
                line_stripped = line.strip().lower()
                line_classification = text_classification_map.get(line_stripped, 'content')
                if line_classification in ('heading', 'subheading'):
                    break
                
                # DOUBLE CHECK: Make sure this line is not a heading/subheading pattern
                line_original = line.strip()
                words = line_original.split()
                
                # Skip if it looks like a heading (short, title case, no punctuation)
                if len(words) <= 5 and line_original[0].isupper() and not any(p in line_original for p in ['.', ',', ';', '!', '?']):
                    if len(line_original) < 40:
                        continue  # Skip potential heading
                
                # Only add if it's substantial content (sentence-like)
                if len(line_original) >= 20 and ('.' in line_original or ',' in line_original or len(line_original) >= 40):
                    final_content.append(line)
                
                # Limit to reasonable amount (about 20 lines of content for better summaries)
                if len(final_content) >= 20:
                    break
            
            # Ensure we have enough content
            if final_content:
                content_text = '\n'.join(final_content)
                # Remove any trailing numbers (like "18" at end of lines)
                content_text = re.sub(r'\s+\d+\s*$', '', content_text, flags=re.MULTILINE)
                # Clean up multiple spaces
                content_text = re.sub(r'\s+', ' ', content_text)
                
                # Get actual end positions from extracted lines
                actual_end_page = exact_lines[-1]['page'] if exact_lines else end_page
                actual_end_line = exact_lines[-1]['line_number'] if exact_lines else end_line
                
                return {
                    'text': content_text,
                    'start_page': start_page,
                    'start_line': start_line,
                    'end_page': actual_end_page,
                    'end_line': actual_end_line,
                    'original_lines': final_content  # Preserve original text
                }
        
        return {
            'text': '',
            'start_page': start_page,
            'start_line': start_line,
            'end_page': end_page,
            'end_line': end_line,
            'original_lines': []
        }
    
    def get_content_for_heading(self, heading: Dict) -> str:
        """
        Get content text for a heading (includes all subheadings and content)
        
        Args:
            heading: Heading dictionary with start/end positions
            
        Returns:
            Content text as a single string
        """
        start_page = heading.get('content_start_page', heading.get('start_page'))
        start_line = heading.get('content_start_line', heading.get('start_line', 1) + 1)
        end_page = heading.get('end_page')
        end_line = heading.get('end_line')
        
        # If end positions are not set, use start as fallback
        if end_page is None or end_line is None:
            end_page = heading.get('start_page')
            end_line = heading.get('start_line', 1)
        
        # Make sure we have valid start positions
        if start_page is None or start_line is None:
            return ""
        
        # Ensure start is before end
        if (start_page > end_page) or (start_page == end_page and start_line > end_line):
            end_page = start_page
            end_line = min(start_line + 100, 9999)  # Reasonable limit
        
        lines = self.pdf_indexer.get_lines_in_range(
            start_page, start_line, end_page, end_line
        )
        
        # Create a map of text to classification for quick lookup (optimized)
        text_classification_map = {}
        if self.classified_lines:
            for cl_line in self.classified_lines:
                text_key = cl_line.get('text', '').strip().lower()
                text_classification_map[text_key] = cl_line.get('classification', 'content')
        
        # Filter out empty lines, headings, and subheadings
        content_lines = []
        for line_text in lines:
            line_stripped = line_text.strip()
            if not line_stripped:
                continue
            
            # Find the classification for this line (optimized lookup)
            text_key = line_stripped.lower()
            line_classification = text_classification_map.get(text_key, 'content')
            
            # Skip if it's classified as heading or subheading
            if line_classification in ('heading', 'subheading'):
                continue
            
            # Prefer actual content (longer lines with punctuation)
            if len(line_stripped) >= 15 or any(p in line_stripped for p in ['.', ',', ';', ':', '!', '?']):
                content_lines.append(line_text)
        
        return '\n'.join(content_lines)
    
    def get_content_range(self, start_page: int, start_line: int, 
                         end_page: int, end_line: int) -> str:
        """
        Get content in a specific range
        
        Args:
            start_page: Starting page number
            start_line: Starting line number
            end_page: Ending page number
            end_line: Ending line number
            
        Returns:
            Content text as a single string
        """
        lines = self.pdf_indexer.get_lines_in_range(
            start_page, start_line, end_page, end_line
        )
        return '\n'.join(lines)

