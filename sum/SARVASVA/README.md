# PDF Document Processor with ML/DL Classification

A comprehensive PDF document processing system that uses Machine Learning and Deep Learning (Hugging Face) to automatically index, classify, and summarize PDF documents.

## Features

### 1️⃣ **Indexing**
Breaks down each line of the PDF by page and line number, creating a complete index of the document structure.

### 2️⃣ **Categorization**
Uses a trained classifier (Hugging Face transformers) to automatically determine if a line is:
- A **heading** (main section title)
- A **subheading** (subsection title)
- **Content** (regular text)

### 3️⃣ **Content Overview**
Automatically builds a structured hierarchy of headings and subheadings, mapping each to its start and end positions in the document. This creates a detailed table of contents.

### 4️⃣ **Summarization**
When a user clicks on a subheading, the system:
- Pulls the content from the PDF using page and line numbers
- Summarizes it using Hugging Face summarization models (BART, T5, etc.)

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download models (automatic on first use):**
   - Classification models will be downloaded automatically
   - Summarization models will be downloaded automatically

## Usage

### Basic Usage

```python
from main import PDFDocumentProcessor

# Initialize processor
processor = PDFDocumentProcessor("document.pdf")

# Process the entire document
results = processor.process_document()

# Get content overview
overview = processor.get_content_overview()

# Summarize a specific subheading
summary = processor.summarize_subheading("Introduction to Machine Learning")
print(summary)

# Close the processor
processor.close()
```

### Command Line Usage

```bash
python main.py document.pdf
```

This will:
1. Index the document
2. Classify all lines
3. Build the hierarchy
4. Create content overview
5. Save results to `processing_results.json`

### Advanced Usage

```python
# Use custom models
processor = PDFDocumentProcessor(
    pdf_path="document.pdf",
    classifier_model="distilbert-base-uncased",  # Custom classifier
    summarizer_model="facebook/bart-large-cnn"   # Custom summarizer
)

# Process document
processor.process_document()

# Print formatted overview
processor.print_content_overview()

# Save results
processor.save_results("my_results.json")
```

### Web Frontend (Streamlit)

**Launch the interactive web interface:**

```bash
streamlit run app.py
```

This will open a web browser with an interactive interface where you can:

1. **Upload PDF**: Drag and drop or select a PDF file
2. **Process Document**: Click the process button to analyze the PDF
3. **View Content Overview**: See all headings and subheadings in an organized table
4. **Explore Sections**: Select headings and subheadings to explore
5. **Generate Summaries**: Click on any subheading to get an AI-powered summary
6. **View Statistics**: See processing statistics and classification distribution
7. **Download Results**: Export processing results as JSON

**Features of the Frontend:**
- 📤 Easy PDF upload
- 📋 Interactive content overview
- 🔍 Section explorer with dropdowns
- 📝 One-click summarization
- 📊 Visual statistics and charts
- 💾 Download results

## Architecture

### Components

1. **`pdf_indexer.py`**: Indexes PDF documents by page and line number
2. **`line_classifier.py`**: Classifies lines using Hugging Face transformers
3. **`hierarchy_builder.py`**: Builds hierarchical structure from classifications
4. **`content_mapper.py`**: Maps headings/subheadings to content positions
5. **`summarizer.py`**: Summarizes content using Hugging Face models
6. **`main.py`**: Main application that ties everything together
7. **`app.py`**: Streamlit web frontend for interactive use

## Model Details

### Classification
- **Default**: Heuristic-based classification with ML features
- **Advanced**: Can use fine-tuned Hugging Face models (e.g., DistilBERT, BERT)
- **Classes**: `heading`, `subheading`, `content`

### Summarization
- **Default Model**: `facebook/bart-large-cnn`
- **Fallback**: `sshleifer/distilbart-cnn-12-6`
- **Features**: Automatic text chunking for long documents

## Output Format

The system generates a JSON file with:
- Document statistics
- Complete content overview (table of contents)
- Hierarchical structure with positions
- All classified lines

## Customization

### Fine-tuning the Classifier

To improve classification accuracy, you can fine-tune a model:

1. Prepare labeled data (heading/subheading/content)
2. Fine-tune a Hugging Face model
3. Save the model
4. Use it in `LineClassifier` or `AdvancedLineClassifier`

### Using Different Summarization Models

You can use any Hugging Face summarization model:

```python
summarizer = ContentSummarizer("t5-base")
# or
summarizer = ContentSummarizer("google/pegasus-xsum")
```

## Requirements

- Python 3.8+
- PyMuPDF (for PDF processing)
- PyTorch (for ML models)
- Transformers (Hugging Face)
- NumPy
- Streamlit (for web frontend)
- Pandas (for data visualization)

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

================================================================================
CONTENT OVERVIEW (Table of Contents)
================================================================================

1. Introduction
   Page 1, Line 5
   Range: Page 1-3, Lines 5-120
   Subheadings (3):
      1.1 Background
      1.2 Objectives
      1.3 Scope
...
```

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

