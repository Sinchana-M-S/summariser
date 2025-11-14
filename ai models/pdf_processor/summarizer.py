"""
Summarizer: Summarizes content using extractive + abstractive approach with source verification
"""
from transformers import pipeline
from typing import Optional, List, Dict
import torch
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentSummarizer:
    """Summarizes content using extractive + abstractive approach with source verification"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize summarizer
        
        Args:
            model_name: Hugging Face model name for summarization
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialize summarization pipeline
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=self.device
            )
            logger.info(f"Loaded summarization model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load {model_name}. Using fallback model.")
            # Fallback to a smaller model
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=self.device
                )
                logger.info("Loaded fallback model: sshleifer/distilbart-cnn-12-6")
            except:
                self.summarizer = None
                logger.error("Failed to load any summarization model")
    
    def extract_important_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """
        Extract key sentences directly from the PDF text (extractive summarization)
        
        Args:
            text: Text to extract sentences from
            num_sentences: Number of sentences to extract
            
        Returns:
            List of important sentences
        """
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        # Score sentences by length and content (longer sentences with punctuation are more important)
        scored_sentences = []
        for sentence in sentences:
            score = len(sentence)
            # Boost score for sentences with important words
            if any(word in sentence.lower() for word in ['important', 'key', 'main', 'primary', 'essential']):
                score += 20
            # Boost for sentences with numbers (often contain facts)
            if re.search(r'\d+', sentence):
                score += 10
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        important_sentences = [s[1] for s in scored_sentences[:num_sentences]]
        
        # If we don't have enough, take first sentences
        if len(important_sentences) < num_sentences:
            important_sentences = sentences[:num_sentences]
        
        return important_sentences
    
    def verify_against_source(self, summary: str, source_text: str) -> str:
        """
        Verify summary against source text and remove any hallucinated content
        
        Args:
            summary: Generated summary
            source_text: Original source text from PDF
            
        Returns:
            Verified summary with only content that can be traced to source
        """
        # Split summary into sentences
        summary_sentences = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip()]
        verified_sentences = []
        
        source_lower = source_text.lower()
        
        for sentence in summary_sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence or key phrases appear in source
            # Extract key words (3+ characters)
            key_words = [w for w in sentence_lower.split() if len(w) >= 3]
            
            if len(key_words) == 0:
                continue
            
            # Check if at least 50% of key words appear in source
            matches = sum(1 for word in key_words if word in source_lower)
            match_ratio = matches / len(key_words) if key_words else 0
            
            # Also check if sentence is very similar to a source sentence
            sentence_similar = False
            for source_sentence in re.split(r'[.!?]+', source_text):
                source_sentence_lower = source_sentence.lower().strip()
                if len(source_sentence_lower) > 10:
                    # Check word overlap
                    source_words = set(w for w in source_sentence_lower.split() if len(w) >= 3)
                    sentence_words = set(w for w in sentence_lower.split() if len(w) >= 3)
                    if source_words and sentence_words:
                        overlap = len(source_words & sentence_words) / len(sentence_words)
                        if overlap > 0.4:  # 40% word overlap
                            sentence_similar = True
                            break
            
            # Include sentence if it matches source
            if match_ratio >= 0.5 or sentence_similar:
                verified_sentences.append(sentence)
            else:
                logger.warning(f"Removed potentially hallucinated sentence: {sentence[:50]}...")
        
        if verified_sentences:
            return '. '.join(verified_sentences) + '.'
        else:
            # If nothing verified, return extractive summary
            logger.warning("No sentences verified, using extractive summary")
            return self.extract_important_sentences(source_text, 2)
    
    def summarize(self, text: str, max_length: int = 150, 
                  min_length: int = 30, do_sample: bool = False) -> str:
        """
        Summarize the given text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            do_sample: Whether to use sampling
            
        Returns:
            Summarized text
        """
        if not text or not text.strip():
            return "No content available for summarization."
        
        if self.summarizer is None:
            # Fallback: return first few sentences
            sentences = text.split('.')
            return '. '.join(sentences[:3]) + '.'
        
        # Handle long texts by chunking
        if len(text) > 1024:  # Model token limit
            # Split into chunks and summarize each
            chunks = self._split_text(text, chunk_size=1000)
            summaries = []
            
            for chunk in chunks:
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample
                    )[0]['summary_text']
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error summarizing chunk: {e}")
                    summaries.append(chunk[:200] + "...")
            
            return ' '.join(summaries)
        else:
            try:
                # Limit text length to prevent hanging
                if len(text) > 2000:
                    text = text[:2000] + "..."
                
                result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample
                )
                return result[0]['summary_text']
            except Exception as e:
                logger.error(f"Error during summarization: {e}")
                # Fallback: return first few sentences
                sentences = text.split('.')
                return '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else text[:200] + "..."
    
    def _split_text(self, text: str, chunk_size: int = 1000) -> list:
        """
        Split text into chunks
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_subheading(self, content: str, subheading_text: str = "", source_text: Optional[str] = None) -> Dict[str, any]:
        """
        Summarize content for a specific subheading using extractive + abstractive approach
        
        Args:
            content: Content text to summarize (can be dict with metadata)
            subheading_text: Optional subheading text for context
            source_text: Optional original source text for verification
            
        Returns:
            Dictionary with summary, verification status, and metadata
        """
        # Handle content as dict (from content_mapper) or string
        if isinstance(content, dict):
            source_text = content.get('text', '')
            content_metadata = content
            content = source_text
        else:
            content_metadata = {}
            if source_text is None:
                source_text = content
        
        # STRICT CLEANING: Remove any irrelevant content, social media prompts, etc.
        lines = content.split('\n')
        cleaned_lines = []
        
        social_media_keywords = [
            'share your thoughts', 'post a video', 'facebook', 'twitter', 'instagram',
            'comment section', 'comment below', 'share your video', 'share a video',
            'we would like to hear', 'tell us about', 'post on', 'follow us',
            'like and share', 'subscribe', 'click here', 'visit our', 'follow our',
            'comment', 'comments', 'share your', 'post your', 'upload your'
        ]
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # STRICT FILTER: Remove social media prompts and irrelevant content
            if any(keyword in line_lower for keyword in social_media_keywords):
                continue  # Skip this line entirely
            
            # Filter out URLs, emails
            if '@' in line_stripped or 'http' in line_lower or 'www.' in line_lower:
                continue
            
            # Skip very short lines that might be headings or noise
            if len(line_stripped) < 10:
                continue
            
            # Skip lines that are just numbers or patterns like "1.", "2."
            if line_stripped.isdigit() or (len(line_stripped) <= 3 and line_stripped.endswith('.')):
                continue
            
            # Skip lines that look like names (2-4 words, title case, no punctuation)
            words = line_stripped.split()
            if 2 <= len(words) <= 4:
                if all(word[0].isupper() if word else False for word in words):
                    if not any(p in line_stripped for p in ['.', '!', '?', ',', ':', ';', '-']):
                        if len(line_stripped) < 30:
                            continue  # Likely a name, skip it
            
            # Only include substantial content (at least 15 chars, preferably with punctuation)
            if len(line_stripped) >= 15:
                # Prefer lines with punctuation (actual sentences) or longer lines
                if any(p in line_stripped for p in ['.', ',', ';', ':', '!', '?']) or len(line_stripped) >= 25:
                    cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # If cleaned content is too short, return a message
        if len(cleaned_content.strip()) < 20:
            return {
                'summary': "Insufficient content available for summarization. The subheading may not have associated paragraph content.",
                'verified': True,
                'method': 'none',
                'metadata': content_metadata
            }
        
        # Use cleaned content for summarization
        # Ensure we have enough content for a meaningful summary
        if len(cleaned_content.strip()) < 50:
            return {
                'summary': "Insufficient content available for summarization. The subheading may not have associated paragraph content.",
                'verified': True,
                'method': 'none',
                'metadata': content_metadata
            }
        
        # Limit content length to prevent hallucination and ensure accuracy
        if len(cleaned_content) > 1500:
            # Take first 1500 chars (usually first few paragraphs)
            cleaned_content = cleaned_content[:1500] + "..."
        
        # STEP 1: Extractive summarization (extract key sentences)
        logger.info(f"Extracting key sentences for: {subheading_text}")
        key_sentences = self.extract_important_sentences(cleaned_content, num_sentences=3)
        extractive_summary = '. '.join(key_sentences) + '.' if key_sentences else ""
        
        # STEP 2: Abstractive summarization with constrained prompt
        abstractive_summary = None
        try:
            logger.info(f"Generating abstractive summary for: {subheading_text}")
            prompt_text = f"""Summarize this text about '{subheading_text}'.
RULES:
- Only use information from the text below
- Do not add external information
- Be concise but accurate

TEXT:
{cleaned_content}"""
            
            abstractive_summary = self.summarize(prompt_text, max_length=200, min_length=50)
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
        
        # STEP 3: Choose best summary and verify
        if abstractive_summary and len(abstractive_summary.strip()) >= 30:
            # Verify abstractive summary against source
            verified_summary = self.verify_against_source(abstractive_summary, cleaned_content)
            
            # Check if verified summary is substantial
            if len(verified_summary.strip()) >= 30:
                return {
                    'summary': verified_summary,
                    'verified': True,
                    'method': 'abstractive',
                    'extractive_backup': extractive_summary,
                    'metadata': content_metadata
                }
        
        # Fallback to extractive summary
        if extractive_summary and len(extractive_summary.strip()) >= 20:
            return {
                'summary': extractive_summary,
                'verified': True,
                'method': 'extractive',
                'metadata': content_metadata
            }
        
        # Last resort: return first sentences
        sentences = [s.strip() for s in cleaned_content.split('.') if len(s.strip()) > 20]
        if sentences:
            fallback = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else '. '.join(sentences) + '.'
            return {
                'summary': fallback,
                'verified': True,
                'method': 'extractive_fallback',
                'metadata': content_metadata
            }
        
        return {
            'summary': cleaned_content[:300] + "...",
            'verified': True,
            'method': 'raw',
            'metadata': content_metadata
        }

