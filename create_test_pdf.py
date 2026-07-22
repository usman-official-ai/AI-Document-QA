"""
Create a test PDF for the AI Document QA System
"""
import os
import sys

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit
except ImportError:
    print("Installing reportlab...")
    os.system("pip install reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit

def create_test_pdf():
    """Create a test PDF for demonstration"""
    print("📄 Creating test PDF...")
    
    c = canvas.Canvas("test_document.pdf", pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height - 50, "Sample Document for QA Testing")
    
    # Content
    c.setFont("Helvetica", 12)
    y = height - 100
    text = """This is a sample document created for testing the AI Document QA System.

Section 1: Introduction
This document demonstrates the capabilities of the RAG (Retrieval-Augmented Generation) system.
The system can answer questions based on the content of uploaded PDFs.

Section 2: Features
- PDF Upload and Processing
- Text Extraction and Splitting
- Vector Embeddings using Sentence Transformers
- ChromaDB for Vector Storage
- Groq LLM for Answer Generation

Section 3: How It Works
The system processes PDFs, creates embeddings, stores them in ChromaDB,
and uses Groq's Llama 3 model to generate answers based on retrieved context.

Section 4: Benefits
1. Accurate answers based on document content
2. Source attribution for transparency
3. Support for multiple PDFs
4. User-friendly interface

Section 5: Conclusion
This system provides a powerful way to query information from PDF documents
using state-of-the-art AI technology."""
    
    # Split text into lines
    lines = simpleSplit(text, "Helvetica", 12, width - 100)
    for line in lines:
        c.drawString(100, y, line)
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)
    
    c.save()
    print("✅ Test PDF created: test_document.pdf")
    print("📁 File location:", os.path.abspath("test_document.pdf"))

if __name__ == "__main__":
    create_test_pdf()