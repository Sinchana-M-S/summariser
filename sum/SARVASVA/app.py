"""
Streamlit Frontend for PDF Document Processor
"""
import streamlit as st
import os
import tempfile
from main import PDFDocumentProcessor
from qa_chat import DocumentQAChat
import json


# Page configuration
st.set_page_config(
    page_title="PDF Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .heading-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .subheading-item {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.25rem 0 0.25rem 2rem;
        border-left: 3px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'content_overview' not in st.session_state:
        st.session_state.content_overview = []
    if 'selected_subheading' not in st.session_state:
        st.session_state.selected_subheading = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'qa_chat' not in st.session_state:
        st.session_state.qa_chat = DocumentQAChat()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []


def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📄 PDF Document Processor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to process"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.session_state.pdf_path = tmp_path
            
            # Process button
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                process_document(tmp_path)
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("""
        - ✅ **Indexing**: Page and line number tracking
        - ✅ **Classification**: ML/DL heading detection
        - ✅ **Hierarchy**: Automatic structure building
        - ✅ **Summarization**: AI-powered content summaries
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This tool uses Hugging Face transformers to:
        1. Index PDF documents
        2. Classify lines as headings/subheadings/content
        3. Build hierarchical structure
        4. Summarize content on demand
        """)
    
    # Main content area
    if st.session_state.processed and st.session_state.processor:
        display_results()
    else:
        display_welcome()


def process_document(pdf_path: str):
    """Process the uploaded PDF document"""
    try:
        with st.spinner("🔄 Processing document... This may take a moment."):
            # Initialize processor
            processor = PDFDocumentProcessor(pdf_path)
            
        # Process document
        results = processor.process_document()
        
        # Store in session state
        st.session_state.processor = processor
        st.session_state.processed = True
        st.session_state.content_overview = processor.get_content_overview()
        
        # Update QA chat with processor reference
        if 'qa_chat' not in st.session_state:
            st.session_state.qa_chat = DocumentQAChat(processor=processor)
        else:
            st.session_state.qa_chat.processor = processor
        
        # Add all content to QA chat automatically
        if st.session_state.qa_chat:
            try:
                for heading in processor.hierarchy:
                    # Add heading content
                    heading_content = processor.content_mapper.get_content_for_heading(heading)
                    if heading_content and len(heading_content.strip()) > 50:
                        st.session_state.qa_chat.add_content(heading['text'], heading_content)
                    
                    # Add subheading content
                    subheadings = heading.get('subheadings', [])
                    for i, subheading in enumerate(subheadings):
                        # Get next subheading for boundary
                        next_subheading = subheadings[i + 1] if i + 1 < len(subheadings) else None
                        subheading_content = processor.content_mapper.get_content_for_subheading(subheading, next_subheading)
                        if subheading_content and len(subheading_content.strip()) > 50:
                            st.session_state.qa_chat.add_content(subheading['text'], subheading_content)
            except Exception as e:
                print(f"Warning: Could not add all content to QA chat: {e}")
        
        st.success(f"✅ Document processed successfully! Found {results['headings']} headings.")
            
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.exception(e)


def display_welcome():
    """Display welcome message"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>Welcome to PDF Document Processor</h2>
            <p style="font-size: 1.1rem; color: #666;">
                Upload a PDF document using the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 How it works:")
        st.markdown("""
        1. **Upload PDF**: Use the sidebar to upload your PDF document
        2. **Process**: Click the "Process Document" button
        3. **Explore**: View the content overview and click on subheadings
        4. **Summarize**: Get AI-powered summaries for any section
        """)


def display_results():
    """Display processing results"""
    processor = st.session_state.processor
    content_overview = st.session_state.content_overview
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Headings", len(content_overview))
    
    total_subheadings = sum(len(entry.get('subheadings', [])) for entry in content_overview)
    with col2:
        st.metric("📑 Total Subheadings", total_subheadings)
    
    with col3:
        st.metric("📊 Total Sections", len(content_overview) + total_subheadings)
    
    with col4:
        if st.button("🔄 Process New Document"):
            reset_session()
            st.rerun()
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Content Overview", "🔍 Explore Sections", "💬 Ask Questions", "📊 Statistics"])
    
    with tab1:
        display_content_overview(content_overview)
    
    with tab2:
        display_explore_sections(content_overview, processor)
    
    with tab3:
        display_qa_chat(processor)
    
    with tab4:
        display_statistics(processor)


def display_content_overview(content_overview):
    """Display content overview (table of contents)"""
    st.markdown('<div class="subheader">📋 Content Overview</div>', unsafe_allow_html=True)
    
    if not content_overview:
        st.info("No headings found in the document.")
        return
    
    for i, entry in enumerate(content_overview, 1):
        # Heading
        with st.expander(f"📌 {i}. {entry['heading']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Page:** {entry['page']} | **Line:** {entry['line']}")
            with col2:
                st.write(f"**Range:** Page {entry['start_page']}-{entry['end_page']}, Lines {entry['start_line']}-{entry['end_line']}")
            
            # Subheadings
            if entry.get('subheadings'):
                st.markdown("**Subheadings:**")
                for j, subheading in enumerate(entry['subheadings'], 1):
                    st.markdown(f"""
                    <div class="subheading-item">
                        <strong>{i}.{j} {subheading['subheading']}</strong><br>
                        <small>Page {subheading['page']}, Line {subheading['line']} | 
                        Range: Page {subheading['start_page']}-{subheading['end_page']}, 
                        Lines {subheading['start_line']}-{subheading['end_line']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No subheadings found under this heading.")


def display_explore_sections(content_overview, processor):
    """Display interactive section explorer"""
    st.markdown('<div class="subheader">🔍 Explore Sections</div>', unsafe_allow_html=True)
    
    if not content_overview:
        st.info("No headings found in the document.")
        return
    
    # Select heading
    heading_options = [f"{i+1}. {entry['heading']}" for i, entry in enumerate(content_overview)]
    selected_heading_idx = st.selectbox(
        "Select a Heading:",
        options=range(len(heading_options)),
        format_func=lambda x: heading_options[x]
    )
    
    selected_entry = content_overview[selected_heading_idx]
    
    # Display heading info
    st.markdown(f"### 📌 {selected_entry['heading']}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Page:** {selected_entry['page']} | **Line:** {selected_entry['line']}")
    with col2:
        st.write(f"**Range:** Page {selected_entry['start_page']}-{selected_entry['end_page']}")
    
    # Subheadings
    if selected_entry.get('subheadings'):
        st.markdown("---")
        st.markdown("#### 📑 Subheadings")
        
        subheading_options = [sub['subheading'] for sub in selected_entry['subheadings']]
        selected_subheading_text = st.selectbox(
            "Select a Subheading to Summarize:",
            options=subheading_options
        )
        
        # Summarize button
        if st.button("📝 Generate Summary", type="primary"):
            with st.spinner("🔄 Generating summary... This may take a moment."):
                try:
                    import time
                    start_time = time.time()
                    summary = processor.summarize_subheading(selected_subheading_text)
                    elapsed = time.time() - start_time
                    
                    if summary:
                        st.session_state.summary = summary
                        st.session_state.selected_subheading = selected_subheading_text
                        # Add content and summary to QA chat
                        if st.session_state.qa_chat:
                            try:
                                subheading = processor.hierarchy_builder.find_subheading(selected_subheading_text)
                                if subheading:
                                    # Find next subheading for boundary
                                    next_subheading = None
                                    for h in processor.hierarchy:
                                        for i, sub in enumerate(h.get('subheadings', [])):
                                            if sub['text'] == selected_subheading_text:
                                                if i + 1 < len(h.get('subheadings', [])):
                                                    next_subheading = h['subheadings'][i + 1]
                                                break
                                    content = processor.content_mapper.get_content_for_subheading(subheading, next_subheading)
                                    if content:
                                        st.session_state.qa_chat.add_content(selected_subheading_text, content)
                                # Also add the summary itself to context
                                st.session_state.qa_chat.add_summary(selected_subheading_text, summary)
                            except Exception as e:
                                print(f"Warning: Could not add content to QA chat: {e}")
                    else:
                        st.warning("Could not generate summary. The subheading may not have sufficient content.")
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.info("💡 Tip: The summarization model may be loading. Try again in a moment.")
        
        # Display summary if available
        if st.session_state.summary and st.session_state.selected_subheading == selected_subheading_text:
            st.markdown("---")
            st.markdown("#### 📄 Summary")
            st.markdown(f"""
            <div class="summary-box">
                <strong>Subheading:</strong> {selected_subheading_text}<br><br>
                <strong>Summary:</strong><br>
                {st.session_state.summary}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No subheadings found under this heading.")
        
        # Option to summarize the heading itself
        if st.button("📝 Summarize This Heading", type="primary"):
            with st.spinner("🔄 Generating summary... This may take a moment."):
                try:
                    import time
                    start_time = time.time()
                    summary = processor.summarize_heading(selected_entry['heading'])
                    elapsed = time.time() - start_time
                    
                    if summary:
                        st.markdown("---")
                        st.markdown("#### 📄 Summary")
                        st.markdown(f"""
                        <div class="summary-box">
                            <strong>Heading:</strong> {selected_entry['heading']}<br><br>
                            <strong>Summary:</strong><br>
                            {summary}
                        </div>
                        """, unsafe_allow_html=True)
                        # Add content and summary to QA chat
                        if st.session_state.qa_chat:
                            try:
                                heading = processor.hierarchy_builder.find_heading(selected_entry['heading'])
                                if heading:
                                    content = processor.content_mapper.get_content_for_heading(heading)
                                    if content:
                                        st.session_state.qa_chat.add_content(selected_entry['heading'], content)
                                # Also add the summary itself to context
                                st.session_state.qa_chat.add_summary(selected_entry['heading'], summary)
                            except Exception as e:
                                print(f"Warning: Could not add content to QA chat: {e}")
                    else:
                        st.warning("Could not generate summary. The heading may not have sufficient content.")
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.info("💡 Tip: Try processing the document again or check if the PDF has readable content.")


def display_qa_chat(processor):
    """Display interactive Q&A chat interface"""
    st.markdown('<div class="subheader">💬 Ask Questions About the Document</div>', unsafe_allow_html=True)
    
    if not processor or not st.session_state.processed:
        st.info("Please process a document first to enable question answering.")
        st.markdown("""
        **How to use:**
        1. Process a PDF document
        2. View summaries of headings/subheadings (this adds content to the chat context)
        3. Ask questions about the content
        """)
        return
    
    # Initialize QA chat if not already done
    if 'qa_chat' not in st.session_state:
        st.session_state.qa_chat = DocumentQAChat(processor=processor)
    else:
        # Update processor reference
        st.session_state.qa_chat.processor = processor
    
    qa_chat = st.session_state.qa_chat
    
    # Info box
    context_count = len(qa_chat.document_context) if hasattr(qa_chat, 'document_context') else 0
    if context_count > 0:
        st.success(f"✅ {context_count} section(s) loaded. You can ask questions about the document!")
    else:
        st.info("💡 **Tip:** Content is automatically added when you process the document. You can also view summaries to add more context.")
    
    # Display chat messages
    st.markdown("### 💬 Chat")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        if st.session_state.chat_messages:
            for msg in st.session_state.chat_messages:
                if msg['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(msg['content'])
                else:
                    with st.chat_message("assistant"):
                        # Check if it's markdown content (from commands)
                        if msg.get('type') == 'command' and '**' in msg['content']:
                            st.markdown(msg['content'])
                        else:
                            st.write(msg['content'])
                        if 'source' in msg and msg['source']:
                            st.caption(f"📄 Source: {msg['source']}")
                        if 'confidence' in msg and msg.get('type') != 'command':
                            confidence = msg['confidence']
                            if confidence > 0.7:
                                st.success(f"Confidence: {confidence:.0%}")
                            elif confidence > 0.4:
                                st.warning(f"Confidence: {confidence:.0%}")
                            else:
                                st.info(f"Confidence: {confidence:.0%}")
        else:
            st.info("👋 Start a conversation! Ask questions about the document content.")
    
    # Quick actions
    st.markdown("#### ⚡ Quick Actions")
    quick_cols = st.columns(3)
    
    with quick_cols[0]:
        if st.button("📋 List Headings", use_container_width=True, type="secondary"):
            command = "List headings"
            st.session_state.chat_messages.append({'role': 'user', 'content': command})
            result = qa_chat.process_command(command) or qa_chat.answer_question(command)
            st.session_state.chat_messages.append({
                'role': 'assistant',
                'content': result['answer'],
                'source': result.get('source'),
                'confidence': result.get('confidence', 0.0),
                'type': result.get('type', 'qa')
            })
            st.rerun()
    
    # Suggested questions
    suggested_questions = qa_chat.get_suggested_questions()
    if suggested_questions:
        st.markdown("#### 💡 Suggested Questions")
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions[:6]):
            with cols[i % 2]:
                if st.button(f"❓ {question}", key=f"suggest_{i}", use_container_width=True):
                    # Add to chat
                    st.session_state.chat_messages.append({
                        'role': 'user',
                        'content': question
                    })
                    # Check if it's a command first
                    command_result = qa_chat.process_command(question)
                    if command_result:
                        result = command_result
                    else:
                        result = qa_chat.answer_question(question)
                    st.session_state.chat_messages.append({
                        'role': 'assistant',
                        'content': result['answer'],
                        'source': result.get('source'),
                        'confidence': result.get('confidence', 0.0),
                        'type': result.get('type', 'qa')
                    })
                    st.rerun()
    
    # Help section
    with st.expander("💡 Chat Commands & Tips", expanded=False):
        st.markdown("""
        **You can use these commands in the chat:**
        
        📋 **Navigation Commands:**
        - "List headings" or "Show headings" - See all available headings
        - "Go to [heading name]" - Navigate to a specific heading and get its summary
        - "Show me [heading name]" - Same as go to
        
        📝 **Summary Commands:**
        - "Summarize [heading/subheading name]" - Get summary of any heading or subheading
        - "Give me summary of [heading name]" - Alternative way to request summary
        - "Tell me about [heading name]" - Get information about a section
        - "Explain [heading name]" - Explain a section
        
        ❓ **Question Commands:**
        - Just ask any question about the document content!
        - The AI will search through all loaded content to answer
        
        **Examples:**
        - "List headings"
        - "Summarize Objectives of Research"
        - "Go to Motivation for Research"
        - "Summarize Descriptive vs. Analytical:" (for subheadings)
        - "What is research?"
        - "What are the objectives of research?"
        """)
    
    # Question input
    st.markdown("---")
    question = st.chat_input("Ask a question or use commands (e.g., 'List headings', 'Summarize [heading]')...")
    
    if question:
        # Add user message
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': question
        })
        
        # Check if it's a command first
        with st.spinner("🤔 Processing..."):
            command_result = qa_chat.process_command(question)
            
            if command_result:
                # It's a command, use the command result
                result = command_result
            else:
                # Regular question, use QA
                result = qa_chat.answer_question(question)
            
            # Add assistant response
            st.session_state.chat_messages.append({
                'role': 'assistant',
                'content': result['answer'],
                'source': result.get('source'),
                'confidence': result.get('confidence', 0.0),
                'type': result.get('type', 'qa')
            })
        
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()


def display_statistics(processor):
    """Display processing statistics"""
    st.markdown('<div class="subheader">📊 Processing Statistics</div>', unsafe_allow_html=True)
    
    if not processor or not st.session_state.processed:
        st.info("No statistics available. Please process a document first.")
        return
    
    # Get statistics from processor
    content_overview = st.session_state.content_overview
    
    # Calculate statistics
    total_headings = len(content_overview)
    total_subheadings = sum(len(entry.get('subheadings', [])) for entry in content_overview)
    total_lines = len(processor.indexed_lines) if hasattr(processor, 'indexed_lines') else 0
    
    # Classification counts
    if hasattr(processor, 'classified_lines'):
        heading_count = sum(1 for line in processor.classified_lines if line.get('classification') == 'heading')
        subheading_count = sum(1 for line in processor.classified_lines if line.get('classification') == 'subheading')
        content_count = sum(1 for line in processor.classified_lines if line.get('classification') == 'content')
    else:
        heading_count = subheading_count = content_count = 0
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📄 Total Lines Indexed", total_lines)
        st.metric("📌 Total Headings", total_headings)
        st.metric("📑 Total Subheadings", total_subheadings)
    
    with col2:
        st.metric("🏷️ Heading Lines", heading_count)
        st.metric("🏷️ Subheading Lines", subheading_count)
        st.metric("📝 Content Lines", content_count)
    
    # Classification distribution
    st.markdown("---")
    st.markdown("#### Classification Distribution")
    
    if heading_count + subheading_count + content_count > 0:
        import pandas as pd
        
        data = {
            'Type': ['Headings', 'Subheadings', 'Content'],
            'Count': [heading_count, subheading_count, content_count]
        }
        df = pd.DataFrame(data)
        
        st.bar_chart(df.set_index('Type'))
    
    # Download results
    st.markdown("---")
    st.markdown("#### 💾 Download Results")
    
    if st.button("📥 Download Processing Results (JSON)"):
        results = {
            'statistics': {
                'total_lines': total_lines,
                'headings': total_headings,
                'total_subheadings': total_subheadings,
                'heading_lines': heading_count,
                'subheading_lines': subheading_count,
                'content_lines': content_count
            },
            'content_overview': content_overview
        }
        
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name="processing_results.json",
            mime="application/json"
        )


def reset_session():
    """Reset session state"""
    if st.session_state.processor:
        try:
            st.session_state.processor.close()
        except:
            pass
    
    # Clean up temp file
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        try:
            os.unlink(st.session_state.pdf_path)
        except:
            pass
    
    # Reset all session state
    st.session_state.processor = None
    st.session_state.processed = False
    st.session_state.pdf_path = None
    st.session_state.content_overview = []
    st.session_state.selected_subheading = None
    st.session_state.summary = None
    if 'qa_chat' in st.session_state:
        st.session_state.qa_chat.clear_context()
    st.session_state.chat_messages = []


if __name__ == "__main__":
    main()

