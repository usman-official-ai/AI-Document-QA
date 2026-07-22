"""
Text splitting module for document chunking
"""
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TextSplitter:
    """Handles text chunking with configurable parameters"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"TextSplitter initialized: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with metadata"""
        if not documents:
            logger.warning("No documents to split")
            return []
        
        try:
            chunks = self.splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                if not hasattr(chunk, 'metadata'):
                    chunk.metadata = {}
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_total'] = len(chunks)
                chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            logger.info(f"✂️ Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        try:
            chunks = self.splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get splitter configuration"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.splitter.separators
        }