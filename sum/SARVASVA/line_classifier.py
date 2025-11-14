"""
Line Classifier: Uses Hugging Face transformers to classify lines as 
heading, subheading, or content
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Optional, Tuple
import numpy as np
import re


class LineClassifier:
    """Classifies lines as heading, subheading, or content using Hugging Face models"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the classifier
        
        Args:
            model_name: Hugging Face model name for classification
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        # For a custom classifier, you would fine-tune this model
        # For now, we'll use a pre-trained model and add a classification head
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model (in production, you'd load a fine-tuned model)
        # For demonstration, we'll use a simple heuristic + model approach
        self.model = None
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize or load the classification model"""
        # In a real scenario, you would:
        # 1. Fine-tune a model on labeled heading/subheading/content data
        # 2. Save and load the fine-tuned model
        
        # For now, we'll use a hybrid approach with heuristics + embeddings
        try:
            # Try to load a fine-tuned model if it exists
            # self.model = AutoModelForSequenceClassification.from_pretrained("path/to/fine-tuned-model")
            pass
        except:
            # Use heuristic-based classification with ML features
            pass
    
    def extract_features(self, text: str, context: Optional[Dict] = None, line_data: Optional[Dict] = None) -> Dict:
        """
        Extract enhanced features from a line of text for classification
        
        Args:
            text: The line of text
            context: Optional context (previous lines, formatting, etc.)
            line_data: Optional line data with font features from PDF
            
        Returns:
            Dictionary of features
        """
        words = text.split()
        word_count = len(words)
        
        features = {
            'length': len(text),
            'word_count': word_count,
            'is_uppercase': text.isupper(),
            'is_title_case': text.istitle(),
            'starts_with_number': bool(re.match(r'^\d+', text)),
            'has_special_chars': bool(re.search(r'[^\w\s]', text)),
            'ends_with_colon': text.strip().endswith(':'),
            'is_short': word_count <= 5,
            'is_very_short': word_count <= 3,
            'is_single_word': word_count == 1,
            'has_numbering': bool(re.match(r'^(\d+[\.\)]|\d+\.\d+|[a-z][\.\)])\s+', text, re.IGNORECASE)),
        }
        
        # Add font features if available from PDF
        if line_data:
            features.update({
                'font_size': line_data.get('font_size', 0.0),
                'is_bold': line_data.get('is_bold', False),
                'indentation': line_data.get('indentation', 0.0),
                'spacing_before': line_data.get('spacing_before', 0.0),
                'line_height': line_data.get('line_height', 0.0),
            })
            
            # Normalize font size (compare to average)
            avg_font_size = context.get('avg_font_size', 12.0) if context else 12.0
            features['font_size_ratio'] = features['font_size'] / avg_font_size if avg_font_size > 0 else 1.0
            features['is_larger_font'] = features['font_size'] > avg_font_size * 1.1
        else:
            features.update({
                'font_size': 0.0,
                'is_bold': False,
                'indentation': 0.0,
                'spacing_before': 0.0,
                'line_height': 0.0,
                'font_size_ratio': 1.0,
                'is_larger_font': False,
            })
        
        # Add context features if available
        if context:
            features.update({
                'previous_line_type': context.get('previous_type', 'unknown'),
                'line_position': context.get('line_position', 0),
                'avg_font_size': context.get('avg_font_size', 12.0),
            })
        
        return features
    
    def classify_line(self, text: str, context: Optional[Dict] = None, line_data: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Classify a single line as 'heading', 'subheading', or 'content' with confidence score
        
        Args:
            text: The line of text to classify
            context: Optional context information
            line_data: Optional line data with font features from PDF
            
        Returns:
            Tuple of (classification, confidence_score)
        """
        if not text.strip():
            return ('content', 1.0)
        
        features = self.extract_features(text, context, line_data)
        
        # Heuristic-based classification (can be replaced with trained model)
        score_heading = 0
        score_subheading = 0
        score_content = 0
        
        # Heading indicators (balanced - not too lenient, not too strict)
        text_stripped = text.strip()
        
        # STRICT FILTERS: Exclude names, social media prompts, OCR errors, etc.
        
        # Filter out bullet points and list items
        if text_stripped.startswith(('•', '○', '▪', '-', '*')) or re.match(r'^\d+[\.\)]\s+[a-z]', text_stripped):
            score_content += 5
            return ('content', 1.0)
        
        # CRITICAL: Filter out OCR errors and malformed text
        # Lines with numbers mixed in text (like "statistically30") are NOT headings
        if re.search(r'[a-zA-Z]+\d+[a-zA-Z]*|\d+[a-zA-Z]+\d*', text_stripped):
            # Has number mixed with letters (OCR error) - exclude
            score_content += 10
            return ('content', 1.0)
        
        # Filter out lines that start with lowercase (incomplete sentences, not headings)
        if text_stripped and text_stripped[0].islower():
            score_content += 8
            return ('content', 1.0)
        
        # Filter out lines that are sentence fragments (end with period but are incomplete)
        if text_stripped.endswith('.') and len(text_stripped.split()) < 8:
            # Short line ending with period - likely content fragment, not heading
            score_content += 5
        
        # Filter out social media prompts and irrelevant content
        social_media_keywords = [
            'share your thoughts', 'post a video', 'facebook', 'twitter', 'instagram',
            'comment section', 'comment below', 'share your video', 'share a video',
            'we would like to hear', 'tell us about', 'post on', 'follow us',
            'like and share', 'subscribe', 'click here', 'visit our'
        ]
        text_lower = text_stripped.lower()
        if any(keyword in text_lower for keyword in social_media_keywords):
            score_content += 10
            return ('content', 1.0)
        
        # Filter out names (common pattern: "First Last" or "First Middle Last")
        # Names are typically 2-4 words, title case, no punctuation
        words = text_stripped.split()
        if 2 <= len(words) <= 4:
            # Check if it looks like a name (all words start with capital, no punctuation)
            if all(word[0].isupper() if word else False for word in words):
                # Check if it ends with common name patterns or is just capitalized words
                if not any(p in text_stripped for p in ['.', '!', '?', ',', ':', ';', '-']):
                    # Additional check: if previous line was content and this is isolated, likely a name
                    if context and context.get('previous_type') == 'content':
                        # Very likely a name (author, teacher, etc.) - exclude
                        score_content += 8
                        return ('content', 1.0)
                    # If it's very short and title case with no context, might be name
                    if len(text_stripped) < 25:
                        score_content += 5
        
        # Filter out email addresses, URLs
        if '@' in text_stripped or 'http' in text_lower or 'www.' in text_lower:
            score_content += 10
            return ('content', 1.0)
        
        # Filter out copyright notices, page numbers, etc.
        if any(phrase in text_lower for phrase in ['copyright', '©', 'page', 'all rights reserved']):
            if len(words) <= 5:
                score_content += 10
                return ('content', 1.0)
        
        # Filter out incomplete sentences that look like content fragments
        # Headings should be complete phrases, not sentence fragments
        if text_stripped.endswith('.') and features['word_count'] >= 5 and features['word_count'] <= 10:
            # Medium length line ending with period - likely content, not heading
            score_content += 4
        
        # Filter out lines that are clearly content (have periods, commas, are long)
        if text_stripped.endswith('.') and features['word_count'] >= 8:
            # Likely a sentence/paragraph, not a heading
            score_content += 3
        
        # Strong heading indicators - ONLY for proper headings
        # Headings must start with capital letter (not lowercase)
        if text_stripped and text_stripped[0].isupper():
            # Font-based indicators (from PDF)
            if features.get('is_bold', False):
                score_heading += 3  # Bold text is often a heading
            if features.get('is_larger_font', False):
                score_heading += 2  # Larger font suggests heading
            if features.get('font_size_ratio', 1.0) > 1.2:
                score_heading += 2  # Significantly larger font
            
            if features['is_uppercase'] and features['word_count'] <= 10:
                score_heading += 4
            # Title case headings (like "What is Research?", "Objectives", "Motivations")
            if features['is_title_case'] and 1 <= features['word_count'] <= 8:
                # Single word titles (like "Objectives", "Motivations") are valid headings
                if features['word_count'] == 1:
                    score_heading += 5  # Strong indicator for single-word headings
                elif 2 <= features['word_count'] <= 6:
                    score_heading += 4  # Strong for short title case phrases
                else:
                    score_heading += 3
            # Headings ending with colon (like "Selecting the Problem:")
            if features['ends_with_colon'] and features['word_count'] <= 8:
                score_heading += 4
            # Short title case or uppercase (like "Objectives", "What is Research?")
            if features['is_short'] and (features['is_title_case'] or features['is_uppercase']) and features['word_count'] <= 6:
                score_heading += 4
            # Question headings (like "What is Research?")
            if text_stripped.endswith('?') and 2 <= features['word_count'] <= 8:
                score_heading += 5  # Very strong for question headings
            
            # Spacing indicators (headings often have more space before them)
            if features.get('spacing_before', 0) > 5.0:
                score_heading += 1
        # Numbered headings (usually subheadings)
        if features['starts_with_number'] and features['word_count'] <= 6:
            score_heading += 1
            score_subheading += 3  # More likely a subheading
        
        # Additional heading patterns (more selective)
        # Only count if line starts with capital (already checked above)
        if text_stripped and text_stripped[0].isupper():
            # Lines without ending punctuation (often headings) - but must be short
            if not text_stripped.endswith(('.', '!', '?', ',', ';', ':')) and 1 <= features['word_count'] <= 8:
                # Single words or short phrases without punctuation are likely headings
                if features['word_count'] == 1:
                    score_heading += 3  # Single word headings like "Objectives"
                elif 2 <= features['word_count'] <= 6:
                    score_heading += 2
                else:
                    score_heading += 1
        
        # Subheading indicators
        if features['starts_with_number']:
            score_subheading += 3
        # Comparison-style headings (e.g., "Descriptive vs. Analytical:") are subheadings
        if ' vs. ' in text_stripped or ' vs ' in text_stripped:
            score_subheading += 4
            score_heading -= 2  # Reduce heading score
        # Lines ending with ":" after a heading are likely subheadings
        if features['ends_with_colon'] and features['word_count'] <= 6:
            # Check if this looks like a comparison or category
            if ' vs. ' in text_stripped or ' vs ' in text_stripped or features['word_count'] <= 4:
                score_subheading += 3
        if features['is_title_case'] and not features['is_uppercase']:
            score_subheading += 2
        if 2 <= features['word_count'] <= 12:
            score_subheading += 1
        
        # Content indicators (more aggressive to filter out false headings)
        # Clearly a paragraph or sentence
        if features['word_count'] >= 12:  # Lowered threshold to catch more content
            score_content += 4
        # Sentences that end with period (likely content, not heading)
        if text_stripped.endswith('.') and features['word_count'] >= 6:
            score_content += 3
        # Lowercase start (usually content)
        if text_stripped and text_stripped[0].islower() and features['word_count'] >= 5:
            score_content += 3
        # Has commas (usually content, not heading)
        if ',' in text_stripped and features['word_count'] >= 6:
            score_content += 2
        # Lines that start with "The", "A", "An", "This", "These" (usually content)
        if re.match(r'^(The|A|An|This|These|That|Those)\s+', text_stripped, re.IGNORECASE) and features['word_count'] >= 5:
            score_content += 2
        # Lines that are too long to be headings
        if features['word_count'] > 10:
            score_content += 2
        
        # Context-based adjustments
        if context:
            prev_type = context.get('previous_type')
            line_pos = context.get('line_position', 0)
            
            # If previous was content and this looks like heading, it's likely a heading
            # But only if it's short and title case
            if prev_type == 'content' and (features['is_title_case'] or features['is_uppercase']) and features['word_count'] <= 8:
                score_heading += 3  # Strong boost for headings after content
            
            # If previous was heading, next title case line is likely subheading
            if prev_type == 'heading':
                if features['is_title_case']:
                    # If it ends with ":" and is short, it's definitely a subheading
                    if text_stripped.endswith(':') and features['word_count'] <= 6:
                        score_subheading += 4  # Very strong subheading indicator
                        score_heading = 0  # Definitely not a heading
                    elif features['word_count'] <= 8:
                        score_subheading += 3
                        score_heading += 1  # Could also be heading, but prefer subheading
                # If previous was heading and this is long, it's content
                if features['word_count'] >= 8:
                    score_content += 2
            elif prev_type == 'subheading':
                score_content += 1
                # If we're still in subheading context, next title case might be another subheading
                if features['is_title_case'] and text_stripped.endswith(':') and features['word_count'] <= 6:
                    score_subheading += 2
        
        # Determine classification
        scores = {
            'heading': score_heading,
            'subheading': score_subheading,
            'content': score_content
        }
        
        # Calculate confidence score (normalize to 0-1)
        total_score = sum(scores.values())
        if total_score > 0:
            max_score = max(scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.0
        else:
            confidence = 0.0
        
        classification = max(scores, key=scores.get)
        
        # STRICT threshold - require higher minimum score for headings
        # Headings must have strong evidence (score >= 4) AND confidence > 0.75
        if classification == 'heading' and (scores['heading'] < 4 or confidence < 0.75):
            # Not confident enough, mark as content
            classification = 'content'
            confidence = 1.0 - confidence  # Invert confidence for content
        elif classification == 'subheading' and (scores['subheading'] < 3 or confidence < 0.6):
            classification = 'content'
            confidence = 1.0 - confidence
        elif max(scores.values()) == 0:
            classification = 'content'
            confidence = 1.0
        
        # FINAL CHECK: If line starts with lowercase, it CANNOT be a heading
        if text_stripped and text_stripped[0].islower() and classification == 'heading':
            classification = 'content'
            confidence = 1.0
        
        # FINAL CHECK: If line has OCR errors (numbers mixed with text), it CANNOT be a heading
        if re.search(r'[a-zA-Z]+\d+[a-zA-Z]*|\d+[a-zA-Z]+\d*', text_stripped) and classification == 'heading':
            classification = 'content'
            confidence = 1.0
        
        # Special case: if heading and subheading scores are close, use context
        if classification == 'subheading' and score_heading >= score_subheading - 1:
            # If previous was heading, prefer subheading
            if context and context.get('previous_type') == 'heading':
                classification = 'subheading'
            elif score_heading >= 3:
                classification = 'heading'
        # If it's a comparison-style line, force subheading
        if ' vs. ' in text_stripped or ' vs ' in text_stripped:
            if score_subheading >= 2:
                classification = 'subheading'
        
        # Final check: if content score is very high, it's definitely content
        if score_content >= 5 and classification in ('heading', 'subheading'):
            classification = 'content'
            confidence = 1.0
        
        return (classification, confidence)
    
    def classify_lines(self, indexed_lines: List[Dict]) -> List[Dict]:
        """
        Classify all indexed lines with enhanced features and confidence scores
        
        Args:
            indexed_lines: List of indexed lines from PDFIndexer (with font features)
            
        Returns:
            List of lines with added 'classification' and 'confidence' fields
        """
        classified_lines = []
        previous_type = None
        
        # Calculate average font size for normalization
        font_sizes = [line.get('font_size', 12.0) for line in indexed_lines if line.get('font_size', 0) > 0]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
        
        for i, line in enumerate(indexed_lines):
            context = {
                'previous_type': previous_type,
                'line_position': i,
                'avg_font_size': avg_font_size,
            }
            
            # Extract line data (font features) from indexed line
            line_data = {
                'font_size': line.get('font_size', 0.0),
                'is_bold': line.get('is_bold', False),
                'indentation': line.get('indentation', 0.0),
                'spacing_before': line.get('spacing_before', 0.0),
                'line_height': line.get('line_height', 0.0),
            }
            
            classification, confidence = self.classify_line(line['text'], context, line_data)
            line['classification'] = classification
            line['confidence'] = confidence
            
            classified_lines.append(line)
            previous_type = classification
        
        # Post-processing: Validate hierarchy and fix issues
        classified_lines = self._validate_hierarchy(classified_lines)
        
        return classified_lines
    
    def _validate_hierarchy(self, classified_lines: List[Dict]) -> List[Dict]:
        """
        Validate heading hierarchy and fix issues
        
        Args:
            classified_lines: List of classified lines
            
        Returns:
            Validated list of classified lines
        """
        # Track hierarchy levels
        current_heading_level = 0
        
        for i, line in enumerate(classified_lines):
            classification = line.get('classification', 'content')
            confidence = line.get('confidence', 0.0)
            
            # Only accept headings with high confidence
            if classification == 'heading' and confidence < 0.75:
                # Downgrade to content if confidence is too low
                line['classification'] = 'content'
                line['confidence'] = 1.0 - confidence
            
            # Validate hierarchy (H1 → H2 → H3, not H1 → H3)
            if classification == 'heading':
                current_heading_level = 1
            elif classification == 'subheading':
                if current_heading_level == 0:
                    # Subheading without a heading - might be a heading instead
                    if confidence > 0.6:
                        line['classification'] = 'heading'
                        current_heading_level = 1
                    else:
                        line['classification'] = 'content'
                else:
                    current_heading_level = 2
        
        # Post-processing: Catch missed headings and fix subheadings
        # Look for patterns that suggest headings but were classified as content
        for i, line in enumerate(classified_lines):
            text = line['text'].strip()
            text_lower = text.lower()
            current_class = line.get('classification', 'content')
            
            # Skip if it's a bullet point or list item
            if text.startswith(('•', '○', '▪', '-', '*')) or re.match(r'^\d+[\.\)]\s+[a-z]', text):
                continue
            
            # STRICT: Skip social media prompts, names, URLs
            social_media_keywords = [
                'share your thoughts', 'post a video', 'facebook', 'twitter', 'instagram',
                'comment section', 'comment below', 'share your video', 'share a video',
                'we would like to hear', 'tell us about', 'post on', 'follow us',
                'like and share', 'subscribe', 'click here', 'visit our'
            ]
            if any(keyword in text_lower for keyword in social_media_keywords):
                line['classification'] = 'content'
                continue
            
            if '@' in text or 'http' in text_lower or 'www.' in text_lower:
                line['classification'] = 'content'
                continue
            
            # Check previous classification for context
            prev_class = classified_lines[i-1].get('classification', 'content') if i > 0 else 'content'
            
            # If classified as content but looks like heading, reconsider
            # BUT: Be very strict - don't convert names to headings
            if current_class == 'content' and text:
                # Skip if it has OCR errors or starts with lowercase
                if re.search(r'[a-zA-Z]+\d+[a-zA-Z]*|\d+[a-zA-Z]+\d*', text) or (text and text[0].islower()):
                    continue
                
                word_count = len(text.split())
                is_title = text.istitle() or text.isupper()
                no_ending_punct = not text.endswith(('.', '!', '?', ',', ';', ':'))
                
                # CATCH single-word headings like "Objectives", "Motivations"
                if word_count == 1 and is_title and no_ending_punct and len(text) >= 5 and len(text) <= 20:
                    # Single word, title case, no punctuation - likely a heading
                    if i + 1 < len(classified_lines):
                        next_class = classified_lines[i+1].get('classification', 'content')
                        next_text = classified_lines[i+1].get('text', '').strip()
                        # If next line is content, this is likely a heading
                        if next_class == 'content' and (len(next_text.split()) >= 5 or next_text.startswith(('•', 'The', 'A', 'An', 'This', 'These', 'That'))):
                            line['classification'] = 'heading'
                            continue
                
                # STRICT CHECK: Exclude names (2-4 words, title case, no punctuation, short)
                if 2 <= word_count <= 4 and is_title and no_ending_punct and len(text) < 25:
                    # If previous was content, this is likely a name (author, teacher, etc.)
                    if prev_class == 'content':
                        continue  # Keep as content, don't convert to heading
                
                # Main section headings (like "Objectives of Research", "Motivation for Research")
                # These are title case, 2-4 words, no punctuation, BUT longer than typical names
                if is_title and 2 <= word_count <= 6 and no_ending_punct and len(text) >= 15:
                    # Check next line - if it's content, this is likely a heading
                    if i + 1 < len(classified_lines):
                        next_class = classified_lines[i+1].get('classification', 'content')
                        next_text = classified_lines[i+1].get('text', '').strip()
                        # Only if next line is clearly content (not another heading)
                        if next_class == 'content' and (len(next_text.split()) >= 5 or next_text.startswith(('•', 'The', 'A', 'An', 'This', 'These', 'That'))):
                            line['classification'] = 'heading'
                # Question headings (but must be substantial, not just a name)
                elif text.endswith('?') and word_count <= 8 and len(text) >= 10:
                    if i + 1 < len(classified_lines):
                        next_class = classified_lines[i+1].get('classification', 'content')
                        if next_class == 'content':
                            line['classification'] = 'heading'
            
            # Fix: If it's a comparison-style line (ends with ":") and previous was heading, make it subheading
            elif current_class == 'heading' and (' vs. ' in text or ' vs ' in text) and text.endswith(':'):
                if prev_class == 'heading':
                    line['classification'] = 'subheading'
            # Fix: If it ends with ":" and previous was heading, likely subheading
            elif current_class == 'heading' and text.endswith(':'):
                word_count = len(text.split())
                if prev_class == 'heading' and word_count <= 6:
                    line['classification'] = 'subheading'
        
        return classified_lines


# Advanced version using actual Hugging Face model
class AdvancedLineClassifier(LineClassifier):
    """Advanced classifier using fine-tuned Hugging Face models"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize advanced classifier
        
        Args:
            model_path: Path to fine-tuned model (optional)
        """
        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        else:
            # Use a pre-trained model for sequence classification
            # You would fine-tune this on your heading/subheading/content dataset
            super().__init__()
    
    def classify_line(self, text: str, context: Optional[Dict] = None, line_data: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Classify using the fine-tuned model
        
        Args:
            text: The line of text to classify
            context: Optional context information
            line_data: Optional line data with font features from PDF
            
        Returns:
            Tuple of (classification, confidence_score)
        """
        if self.model is None:
            # Fallback to heuristic method
            return super().classify_line(text, context, line_data)
        
        # Tokenize and classify
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Map class index to label
        class_labels = ['content', 'subheading', 'heading']
        classification = class_labels[predicted_class] if predicted_class < len(class_labels) else 'content'
        
        return (classification, confidence)

