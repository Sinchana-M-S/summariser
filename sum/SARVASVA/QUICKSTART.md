# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Web Frontend (Easiest Way) ⭐

**Launch the interactive web interface:**

```bash
streamlit run app.py
```

This opens a browser with a user-friendly interface where you can:
- Upload PDF files
- Process documents with one click
- View content overview interactively
- Generate summaries by clicking subheadings
- See statistics and download results

**Recommended for first-time users!**

### 2. Command Line

```bash
python main.py your_document.pdf
```

This will:
- Index the document by page and line number
- Classify each line as heading/subheading/content
- Build a hierarchical structure
- Create a content overview
- Save results to `processing_results.json`

### 3. Use in Python Code

```python
from main import PDFDocumentProcessor

# Initialize
processor = PDFDocumentProcessor("document.pdf")

# Process
processor.process_document()

# View overview
processor.print_content_overview()

# Summarize a subheading
summary = processor.summarize_subheading("Introduction")
print(summary)

# Clean up
processor.close()
```

## Features Overview

### ✅ Indexing
Every line is indexed with:
- Page number
- Line number
- Text content

### ✅ Classification
Lines are automatically classified as:
- **Heading**: Main section titles
- **Subheading**: Subsection titles  
- **Content**: Regular text

### ✅ Hierarchy Building
Automatically creates:
- Nested structure (headings → subheadings)
- Start/end positions for each section
- Content mapping

### ✅ Summarization
Click on any subheading to get:
- Content extraction by page/line numbers
- AI-powered summary using Hugging Face models

## Example Output

```
📄 Step 1: Indexing document by page and line number...
   ✓ Indexed 1250 lines

🤖 Step 2: Classifying lines (heading/subheading/content)...
   ✓ Classified 1250 lines
   - Headings: 12, Subheadings: 45, Content: 1193

🏗️  Step 3: Building hierarchical structure...
   ✓ Built hierarchy with 12 headings

📋 Step 4: Creating content overview...
   ✓ Created content overview with 12 entries
```

## Customization

### Use Different Models

```python
processor = PDFDocumentProcessor(
    pdf_path="document.pdf",
    summarizer_model="facebook/bart-large-cnn"  # or "t5-base", etc.
)
```

### Fine-tune Classifier

1. Prepare labeled data (heading/subheading/content)
2. Fine-tune a Hugging Face model
3. Use `AdvancedLineClassifier` with your model path

## Troubleshooting

### Model Download Issues
Models download automatically on first use. If you have internet issues:
- Download models manually from Hugging Face
- Use offline mode with pre-downloaded models

### Memory Issues
For large PDFs:
- Use smaller summarization models (e.g., `distilbart-cnn-12-6`)
- Process in chunks
- Use CPU instead of GPU if needed

### Classification Accuracy
To improve classification:
- Fine-tune the classifier on your document type
- Adjust heuristics in `LineClassifier`
- Use domain-specific models

## Next Steps

1. **Try the web frontend**: `streamlit run app.py` (easiest way to test)
2. Process your first PDF: `python main.py document.pdf`
3. Explore the results in `processing_results.json`
4. Try summarizing different sections
5. Customize models for your use case

