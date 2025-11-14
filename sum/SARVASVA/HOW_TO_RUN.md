# How to Run the PDF Document Processor

## Quick Start (Recommended)

### Step 1: Install Dependencies

Open your terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install all required packages:
- PyMuPDF (for PDF processing)
- PyTorch (for ML models)
- Transformers (Hugging Face)
- Streamlit (for web interface)
- Pandas (for data visualization)
- NumPy

### Step 2: Launch the Web Interface

Run this command:

```bash
streamlit run app.py
```

This will:
1. Start the Streamlit server
2. Automatically open your web browser
3. Display the PDF Document Processor interface

**Note:** The first time you run it, Hugging Face models will be downloaded automatically (this may take a few minutes).

### Step 3: Use the Application

1. **Upload a PDF**: Use the sidebar to upload your PDF file
2. **Process Document**: Click the "🚀 Process Document" button
3. **Explore**: 
   - View the **Content Overview** tab to see all headings
   - Use the **Explore Sections** tab to browse and generate summaries
   - Use the **Ask Questions** tab to chat with the document
   - Check **Statistics** for processing details

## Alternative: Command Line Usage

If you prefer command line:

```bash
python main.py your_document.pdf
```

This will:
- Process the PDF
- Display results in the terminal
- Save results to `processing_results.json`

## Troubleshooting

### Issue: "Module not found" error
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Models not downloading
**Solution:** 
- Check your internet connection
- Models download automatically on first use
- Wait a few minutes for the download to complete

### Issue: Port already in use
**Solution:** Streamlit will try to use port 8501. If it's busy:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Out of memory
**Solution:** 
- Close other applications
- Use smaller PDFs for testing
- The system will automatically use CPU if GPU is not available

## Features Available

✅ **Indexing**: Every line indexed by page and line number  
✅ **Classification**: Automatic heading/subheading/content detection  
✅ **Hierarchy**: Structured table of contents  
✅ **Summarization**: AI-powered summaries for any section  
✅ **Interactive Chat**: Ask questions about the document content  
✅ **Navigation**: Jump to any heading and get its summary  

## Example Workflow

1. Run `streamlit run app.py`
2. Upload a PDF (e.g., research paper, book chapter)
3. Click "Process Document"
4. Wait for processing to complete
5. Go to "Ask Questions" tab
6. Try asking:
   - "List headings"
   - "What is research?"
   - "Summarize [heading name]"
   - "What are the goals of research?"

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: At least 4GB (8GB recommended)
- **Storage**: ~2GB for models (downloaded automatically)
- **Internet**: Required for first-time model download

## Need Help?

If you encounter any issues:
1. Check that all dependencies are installed
2. Ensure Python 3.8+ is being used
3. Check your internet connection (for model downloads)
4. Review the error messages for specific guidance

