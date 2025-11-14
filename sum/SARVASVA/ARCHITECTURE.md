# System Architecture

## Overview

This PDF Document Processor uses Machine Learning and Deep Learning (Hugging Face) to automatically process PDF documents through four main stages:

1. **Indexing** → 2. **Classification** → 3. **Hierarchy Building** → 4. **Summarization**

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PDFDocumentProcessor                       │
│                      (main.py)                               │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ PDFIndexer   │   │LineClassifier│   │Hierarchy     │
│              │   │              │   │Builder       │
│ - Index by   │   │ - Classify   │   │ - Build      │
│   page/line  │   │   lines      │   │   structure  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ContentMapper │
                   │              │
                   │ - Map content│
                   │   positions  │
                   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │Summarizer    │
                   │              │
                   │ - Summarize  │
                   │   content    │
                   └──────────────┘
```

## Data Flow

### Stage 1: Indexing
```
PDF File → PDFIndexer → List[Dict]
                        {
                          'page': int,
                          'line_number': int,
                          'text': str
                        }
```

### Stage 2: Classification
```
Indexed Lines → LineClassifier → Classified Lines
                (Hugging Face)    {
                                    'page': int,
                                    'line_number': int,
                                    'text': str,
                                    'classification': 'heading'|'subheading'|'content'
                                  }
```

### Stage 3: Hierarchy Building
```
Classified Lines → HierarchyBuilder → Hierarchy Structure
                                      {
                                        'type': 'heading',
                                        'text': str,
                                        'page': int,
                                        'start_page': int,
                                        'end_page': int,
                                        'subheadings': [...]
                                      }
```

### Stage 4: Content Mapping & Summarization
```
Subheading Click → ContentMapper → Content Text → Summarizer → Summary
                  (Get by page/line)              (Hugging Face)
```

## Key Components

### 1. PDFIndexer (`pdf_indexer.py`)
- **Purpose**: Extract and index PDF content by page and line
- **Technology**: PyMuPDF (fitz)
- **Output**: List of indexed lines with page/line numbers

### 2. LineClassifier (`line_classifier.py`)
- **Purpose**: Classify lines as heading/subheading/content
- **Technology**: 
  - Hugging Face Transformers (DistilBERT, BERT, etc.)
  - Heuristic-based classification with ML features
- **Features Used**:
  - Text length, word count
  - Case patterns (uppercase, title case)
  - Special characters, formatting
  - Context (previous line type)

### 3. HierarchyBuilder (`hierarchy_builder.py`)
- **Purpose**: Build nested structure from classifications
- **Logic**:
  - Headings start new top-level sections
  - Subheadings nest under headings
  - Content lines update end positions
- **Output**: Hierarchical tree structure

### 4. ContentMapper (`content_mapper.py`)
- **Purpose**: Map headings/subheadings to their content
- **Functionality**: Extract content using page/line ranges

### 5. ContentSummarizer (`summarizer.py`)
- **Purpose**: Summarize extracted content
- **Technology**: Hugging Face summarization models
  - Default: `facebook/bart-large-cnn`
  - Fallback: `sshleifer/distilbart-cnn-12-6`
- **Features**:
  - Automatic text chunking for long documents
  - Configurable summary length

## ML/DL Models

### Classification Models
- **Base Model**: `distilbert-base-uncased`
- **Customization**: Can fine-tune on labeled data
- **Classes**: 3 classes (heading, subheading, content)

### Summarization Models
- **Primary**: BART (Bidirectional and Auto-Regressive Transformers)
- **Alternatives**: T5, Pegasus, etc.
- **Task**: Abstractive summarization

## Extensibility

### Adding Custom Classifiers
1. Fine-tune a model on your data
2. Save the model
3. Use `AdvancedLineClassifier` with model path

### Adding Custom Summarizers
```python
summarizer = ContentSummarizer("your-model-name")
```

### Custom Features
- Add domain-specific features in `LineClassifier.extract_features()`
- Modify heuristics in `LineClassifier.classify_line()`
- Adjust hierarchy logic in `HierarchyBuilder`

## Performance Considerations

### Memory
- Models load on initialization
- Large PDFs processed incrementally
- Summarization chunks long texts

### Speed
- GPU acceleration (if available)
- Batch processing possible
- Caching of classifications

### Accuracy
- Classification improves with fine-tuning
- Heuristics provide baseline
- Context-aware classification

## Future Enhancements

1. **Fine-tuned Models**: Pre-trained models on document datasets
2. **Multi-language Support**: Language detection and models
3. **Formatting Analysis**: Font size, style, position-based classification
4. **Interactive UI**: Web interface for document exploration
5. **Batch Processing**: Process multiple PDFs
6. **Export Formats**: Export to Markdown, HTML, etc.


