"""
PDF loading and text extraction module - Windows Compatible
"""
import os
import logging
from typing import List
from PyPDF2 import PdfReader
from langchain_core.documents import Document  # Changed this import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """Handles PDF loading and text extraction"""
    
    def __init__(self):
        self.documents = []
    
    def load_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """Load multiple PDFs and extract text"""
        all_documents = []
        
        if not pdf_paths:
            logger.warning("No PDF paths provided")
            return all_documents
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.error(f"File not found: {pdf_path}")
                continue
                
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    text = ""
                    page_count = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "pages": page_count,
                                "loader": "PyPDF2"
                            }
                        )
                        all_documents.append(doc)
                        logger.info(f"✅ Loaded {page_count} pages from {os.path.basename(pdf_path)}")
                    else:
                        logger.error(f"❌ No text extracted from {pdf_path}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to load {pdf_path}: {e}")
        
        return all_documents
    
    def save_uploaded_files(self, uploaded_files, save_dir: str = "data") -> List[str]:
        """Save uploaded PDF files to disk"""
        if not uploaded_files:
            return []
        
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = []
        for uploaded_file in uploaded_files:
            try:
                filename = self._sanitize_filename(uploaded_file.name)
                file_path = os.path.join(save_dir, filename)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                saved_paths.append(file_path)
                logger.info(f"✅ Saved: {filename}")
                
            except Exception as e:
                logger.error(f"❌ Error saving {uploaded_file.name}: {e}")
        
        return saved_paths
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        filename = os.path.basename(filename)
        filename = filename.replace(' ', '_')
        filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
        return filename
    
    def get_file_info(self, file_path: str) -> dict:
        """Get information about a PDF file"""
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
            
            return {
                "name": file_name,
                "size": file_size,
                "pages": page_count,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {
                "name": os.path.basename(file_path),
                "error": str(e)
            }