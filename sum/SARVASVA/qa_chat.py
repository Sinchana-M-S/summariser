"""
Question Answering Chat: Interactive Q&A based on document content
"""
from transformers import pipeline
from typing import Dict, List, Optional
import torch
import re


class DocumentQAChat:
    """Interactive question-answering chat for document content"""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad", processor=None):
        """
        Initialize QA chat
        
        Args:
            model_name: Hugging Face QA model name
            processor: Optional PDFDocumentProcessor instance for navigation
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.qa_pipeline = None
        self.document_context: Dict[str, str] = {}  # Store content by heading/subheading
        self.processor = processor  # Reference to processor for navigation
        
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                device=self.device
            )
        except Exception as e:
            print(f"Warning: Could not load {model_name}. Using fallback model.")
            try:
                # Fallback to a smaller model
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-uncased-distilled-squad",
                    device=self.device
                )
            except:
                self.qa_pipeline = None
    
    def add_content(self, section_name: str, content: str):
        """
        Add content to the document context
        
        Args:
            section_name: Name of the section (heading/subheading)
            content: Content text for that section
        """
        if content and len(content.strip()) > 20:
            # If section already exists, append new content
            if section_name in self.document_context:
                existing = self.document_context[section_name]
                # Avoid duplicates
                if content not in existing:
                    self.document_context[section_name] = existing + "\n\n" + content
            else:
                self.document_context[section_name] = content
    
    def add_summary(self, section_name: str, summary: str):
        """
        Add a summary to the document context (in addition to content)
        
        Args:
            section_name: Name of the section (heading/subheading)
            summary: Summary text for that section
        """
        if summary and len(summary.strip()) > 20:
            summary_key = f"{section_name} (Summary)"
            self.document_context[summary_key] = summary
    
    def add_multiple_contents(self, contents: Dict[str, str]):
        """
        Add multiple content sections at once
        
        Args:
            contents: Dictionary mapping section names to content
        """
        for section_name, content in contents.items():
            self.add_content(section_name, content)
    
    def answer_question(self, question: str, context_limit: int = 4000) -> Dict[str, any]:
        """
        Answer a question based on the document content
        
        Args:
            question: The question to answer
            context_limit: Maximum context length to use
            
        Returns:
            Dictionary with answer, confidence, and source section
        """
        if not self.qa_pipeline:
            return {
                'answer': "QA model not available. Please check your internet connection and try again.",
                'confidence': 0.0,
                'source': None
            }
        
        if not question or not question.strip():
            return {
                'answer': "Please ask a question.",
                'confidence': 0.0,
                'source': None
            }
        
        # Combine all document contexts with better formatting
        context_parts = []
        for name, content in self.document_context.items():
            # Clean and format content
            cleaned_content = content.strip()
            if len(cleaned_content) > 20:  # Only include substantial content
                context_parts.append(f"Section: {name}\n{cleaned_content}")
        
        all_context = "\n\n---\n\n".join(context_parts)
        
        if not all_context or len(all_context.strip()) < 50:
            return {
                'answer': "No document content available. Please process a document and view some summaries first.",
                'confidence': 0.0,
                'source': None
            }
        
        # Try multiple strategies for better answers
        best_answer = None
        best_confidence = 0.0
        best_source = None
        
        # Strategy 1: Use full context (if not too long)
        if len(all_context) <= context_limit:
            try:
                result = self.qa_pipeline(question=question, context=all_context)
                answer = result.get('answer', '').strip()
                confidence = result.get('score', 0.0)
                
                if answer and confidence > best_confidence:
                    best_answer = answer
                    best_confidence = confidence
                    # Find source
                    for section_name, content in self.document_context.items():
                        if answer.lower() in content.lower() or any(word in content.lower() for word in answer.lower().split()[:3]):
                            best_source = section_name
                            break
            except Exception as e:
                print(f"Error with full context: {e}")
        
        # Strategy 2: If full context failed or answer is poor, try each section individually
        if not best_answer or best_confidence < 0.3:
            for section_name, content in self.document_context.items():
                if len(content.strip()) < 50:
                    continue
                
                try:
                    # Limit individual section context
                    section_context = content[:2000] if len(content) > 2000 else content
                    result = self.qa_pipeline(question=question, context=section_context)
                    answer = result.get('answer', '').strip()
                    confidence = result.get('score', 0.0)
                    
                    if answer and confidence > best_confidence:
                        best_answer = answer
                        best_confidence = confidence
                        best_source = section_name
                        
                        # If we get a good answer, break early
                        if confidence > 0.5:
                            break
                except Exception as e:
                    continue
        
        # Strategy 3: If still no good answer, try with truncated full context
        if (not best_answer or best_confidence < 0.2) and len(all_context) > context_limit:
            try:
                # Split context into chunks and try each
                chunks = []
                current_chunk = ""
                for part in context_parts:
                    if len(current_chunk) + len(part) < context_limit:
                        current_chunk += "\n\n---\n\n" + part if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = part
                if current_chunk:
                    chunks.append(current_chunk)
                
                for chunk in chunks:
                    result = self.qa_pipeline(question=question, context=chunk)
                    answer = result.get('answer', '').strip()
                    confidence = result.get('score', 0.0)
                    
                    if answer and confidence > best_confidence:
                        best_answer = answer
                        best_confidence = confidence
                        # Find source from chunk
                        for section_name in self.document_context.keys():
                            if section_name in chunk:
                                best_source = section_name
                                break
                        if confidence > 0.4:
                            break
            except Exception as e:
                pass
        
        if best_answer and best_confidence > 0.1:
            return {
                'answer': best_answer,
                'confidence': best_confidence,
                'source': best_source
            }
        else:
            return {
                'answer': "I couldn't find a clear answer to that question in the document. Try asking about specific sections or rephrasing your question.",
                'confidence': 0.0,
                'source': None
            }
    
    def get_suggested_questions(self) -> List[str]:
        """
        Get suggested questions based on available content
        
        Returns:
            List of suggested questions
        """
        suggestions = []
        
        if self.document_context:
            # Generate questions based on section names
            for section_name in list(self.document_context.keys())[:5]:
                if "research" in section_name.lower():
                    suggestions.append(f"What is {section_name.lower()}?")
                    suggestions.append(f"Explain {section_name.lower()}")
                elif "objective" in section_name.lower():
                    suggestions.append("What are the objectives of research?")
                elif "motivation" in section_name.lower():
                    suggestions.append("What motivates people to conduct research?")
                elif "type" in section_name.lower():
                    suggestions.append("What are the different types of research?")
        
        # Add general questions
        general_questions = [
            "What is research?",
            "What are the key steps in research?",
            "What are the objectives of research?",
            "What motivates people to conduct research?",
            "What are the different types of research?",
        ]
        
        suggestions.extend(general_questions[:3])
        return suggestions[:8]  # Return top 8 suggestions
    
    def clear_context(self):
        """Clear all stored document context"""
        self.document_context = {}
    
    def process_command(self, user_input: str) -> Optional[Dict[str, any]]:
        """
        Process special commands for navigation and summaries
        
        Args:
            user_input: User's input text
            
        Returns:
            Dictionary with command result or None if not a command
        """
        if not self.processor:
            return None
        
        user_input_lower = user_input.lower().strip()
        
        # Command: List headings
        if any(phrase in user_input_lower for phrase in ['list headings', 'show headings', 'what headings', 'available headings', 'headings available']):
            headings = []
            for heading in self.processor.hierarchy:
                headings.append(heading['text'])
            
            if headings:
                heading_list = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headings)])
                return {
                    'answer': f"**Available Headings:**\n\n{heading_list}\n\nYou can ask me to summarize any of these headings!",
                    'confidence': 1.0,
                    'source': None,
                    'type': 'command',
                    'command': 'list_headings',
                    'headings': headings
                }
            else:
                return {
                    'answer': "No headings found in the document.",
                    'confidence': 1.0,
                    'source': None,
                    'type': 'command'
                }
        
        # Command: Summarize heading or subheading
        # Only match explicit summary commands, not general questions
        summarize_patterns = [
            r'summarize\s+(.+)',
            r'summary\s+of\s+(.+)',
            r'summarise\s+(.+)',
            r'give\s+me\s+summary\s+of\s+(.+)',
            r'show\s+summary\s+of\s+(.+)',
        ]
        
        # Check if it's a clear summary command (not a general question)
        is_summary_command = any(re.search(pattern, user_input_lower, re.IGNORECASE) for pattern in summarize_patterns)
        
        # Also check for "tell me about" or "explain" but only if it matches a section name
        tell_me_pattern = r'tell\s+me\s+about\s+(.+)'
        explain_pattern = r'explain\s+(.+)'
        
        tell_me_match = re.search(tell_me_pattern, user_input_lower, re.IGNORECASE)
        explain_match = re.search(explain_pattern, user_input_lower, re.IGNORECASE)
        
        # Check if "tell me about" or "explain" matches a known section
        section_text = None
        if tell_me_match:
            section_text = tell_me_match.group(1).strip().rstrip('?').strip()
        elif explain_match:
            section_text = explain_match.group(1).strip().rstrip('?').strip()
        
        # If it's a summary command or matches a known section, try to summarize
        if is_summary_command or (section_text and self._matches_section_name(section_text)):
            if not section_text and is_summary_command:
                # Extract section text from summary command
                for pattern in summarize_patterns:
                    match = re.search(pattern, user_input_lower, re.IGNORECASE)
                    if match:
                        section_text = match.group(1).strip().rstrip('?').strip()
                        break
            
            if section_text:
                # Try to find as heading first
                summary = None
                found_section = None
                section_type = None
                
                # Try exact match for heading
                summary = self.processor.summarize_heading(section_text)
                if summary:
                    found_section = section_text
                    section_type = 'heading'
                else:
                    # Try partial match for heading
                    for heading in self.processor.hierarchy:
                        if section_text.lower() in heading['text'].lower() or heading['text'].lower() in section_text.lower():
                            summary = self.processor.summarize_heading(heading['text'])
                            if summary:
                                found_section = heading['text']
                                section_type = 'heading'
                                break
                    
                    # If not found as heading, try subheading
                    if not summary:
                        summary = self.processor.summarize_subheading(section_text)
                        if summary:
                            found_section = section_text
                            section_type = 'subheading'
                        else:
                            # Try partial match for subheading
                            for heading in self.processor.hierarchy:
                                for subheading in heading.get('subheadings', []):
                                    if section_text.lower() in subheading['text'].lower() or subheading['text'].lower() in section_text.lower():
                                        summary = self.processor.summarize_subheading(subheading['text'])
                                        if summary:
                                            found_section = subheading['text']
                                            section_type = 'subheading'
                                            break
                                if summary:
                                    break
                
                if summary and found_section:
                    # Add summary to context for future QA
                    self.add_summary(found_section, summary)
                    
                    section_label = 'Heading' if section_type == 'heading' else 'Subheading'
                    return {
                        'answer': f"**Summary of {section_label}: '{found_section}'**\n\n{summary}",
                        'confidence': 1.0,
                        'source': found_section,
                        'type': 'command',
                        'command': 'summarize',
                        'section': found_section,
                        'section_type': section_type
                    }
                # If no section found but it was a clear summary command, return None to let QA handle it
                elif is_summary_command:
                    # For explicit summary commands, return None so QA can try to answer
                    return None
        
        # Command: Go to / Navigate to heading
        navigate_patterns = [
            r'go\s+to\s+(.+)',
            r'navigate\s+to\s+(.+)',
            r'show\s+me\s+(.+)',
            r'open\s+(.+)',
        ]
        
        for pattern in navigate_patterns:
            match = re.search(pattern, user_input_lower, re.IGNORECASE)
            if match:
                heading_text = match.group(1).strip()
                
                # Find the heading
                for heading in self.processor.hierarchy:
                    if heading_text.lower() in heading['text'].lower() or heading['text'].lower() in heading_text.lower():
                        # Get summary
                        summary = self.processor.summarize_heading(heading['text'])
                        if summary:
                            # Add summary to context for future QA
                            self.add_summary(heading['text'], summary)
                            
                            # Also add content if available
                            try:
                                content = self.processor.content_mapper.get_content_for_heading(heading)
                                if content:
                                    self.add_content(heading['text'], content)
                            except:
                                pass
                            
                            return {
                                'answer': f"**Navigated to: '{heading['text']}'**\n\n**Summary:**\n\n{summary}",
                                'confidence': 1.0,
                                'source': heading['text'],
                                'type': 'command',
                                'command': 'navigate',
                                'heading': heading['text']
                            }
                
                return {
                    'answer': f"I couldn't find a heading matching '{heading_text}'. Try asking 'List headings' to see available headings.",
                    'confidence': 0.0,
                    'source': None,
                    'type': 'command'
                }
        
        return None  # Not a command, treat as regular question
    
    def _matches_section_name(self, text: str) -> bool:
        """
        Check if text matches any section name (heading or subheading)
        
        Args:
            text: Text to check
            
        Returns:
            True if text matches a section name
        """
        if not self.processor:
            return False
        
        text_lower = text.lower().strip()
        
        # Check headings
        for heading in self.processor.hierarchy:
            heading_text = heading['text'].lower()
            if text_lower in heading_text or heading_text in text_lower:
                return True
            # Check subheadings
            for subheading in heading.get('subheadings', []):
                subheading_text = subheading['text'].lower()
                if text_lower in subheading_text or subheading_text in text_lower:
                    return True
        
        return False

