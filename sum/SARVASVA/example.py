"""
Example script demonstrating how to use the PDF Document Processor
"""
from main import PDFDocumentProcessor
import json


def example_basic_usage():
    """Basic usage example"""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Replace with your PDF path
    pdf_path = "sample_document.pdf"
    
    try:
        # Initialize processor
        processor = PDFDocumentProcessor(pdf_path)
        
        # Process document
        results = processor.process_document()
        
        # Print overview
        processor.print_content_overview()
        
        # Save results
        processor.save_results("example_results.json")
        
        processor.close()
        
    except FileNotFoundError:
        print(f"PDF file '{pdf_path}' not found. Please provide a valid PDF path.")
    except Exception as e:
        print(f"Error: {e}")


def example_summarization():
    """Example of summarizing specific subheadings"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Summarization")
    print("="*80)
    
    pdf_path = "sample_document.pdf"
    
    try:
        processor = PDFDocumentProcessor(pdf_path)
        processor.process_document()
        
        # Get content overview
        overview = processor.get_content_overview()
        
        # Summarize first subheading if available
        if overview and overview[0].get('subheadings'):
            first_subheading = overview[0]['subheadings'][0]
            subheading_text = first_subheading['subheading']
            
            print(f"\nSummarizing: '{subheading_text}'")
            summary = processor.summarize_subheading(subheading_text)
            
            if summary:
                print(f"\nSummary:\n{summary}")
        
        processor.close()
        
    except FileNotFoundError:
        print(f"PDF file '{pdf_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


def example_custom_models():
    """Example using custom Hugging Face models"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Models")
    print("="*80)
    
    pdf_path = "sample_document.pdf"
    
    try:
        # Use custom models
        processor = PDFDocumentProcessor(
            pdf_path=pdf_path,
            # classifier_model="your-fine-tuned-classifier",  # Optional
            summarizer_model="facebook/bart-large-cnn"  # Custom summarizer
        )
        
        processor.process_document()
        processor.print_content_overview()
        processor.close()
        
    except FileNotFoundError:
        print(f"PDF file '{pdf_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


def example_interactive():
    """Interactive example for exploring document"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Interactive Exploration")
    print("="*80)
    
    pdf_path = "sample_document.pdf"
    
    try:
        processor = PDFDocumentProcessor(pdf_path)
        processor.process_document()
        
        overview = processor.get_content_overview()
        
        print("\nAvailable Headings:")
        for i, entry in enumerate(overview, 1):
            print(f"{i}. {entry['heading']}")
            if entry['subheadings']:
                for j, sub in enumerate(entry['subheadings'], 1):
                    print(f"   {i}.{j} {sub['subheading']}")
        
        # Example: Summarize a specific subheading
        if overview and overview[0].get('subheadings'):
            subheading_to_summarize = overview[0]['subheadings'][0]['subheading']
            print(f"\n\nSummarizing: '{subheading_to_summarize}'")
            summary = processor.summarize_subheading(subheading_to_summarize)
            if summary:
                print(f"\n{summary}")
        
        processor.close()
        
    except FileNotFoundError:
        print(f"PDF file '{pdf_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    # example_summarization()
    # example_custom_models()
    # example_interactive()


