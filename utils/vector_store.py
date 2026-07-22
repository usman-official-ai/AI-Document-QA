"""
Vector store management using ChromaDB
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        self.collection_name = "pdf_documents"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = None
        logger.info(f"VectorStore initialized: {persist_directory}")
    
    def create_collection(self, embedding_dim: int = 384) -> bool:
        """
        Create or reset collection
        
        Args:
            embedding_dim: Dimension of embeddings
            
        Returns:
            True if successful
        """
        try:
            # Delete existing collection if it exists
            existing_collections = self.client.list_collections()
            if self.collection_name in [col.name for col in existing_collections]:
                self.client.delete_collection(self.collection_name)
                logger.info(f"🗑️ Deleted existing collection: {self.collection_name}")
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_dim": embedding_dim
                }
            )
            
            logger.info(f"📚 Created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Add documents with embeddings to collection
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            Number of documents added
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
        
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs and documents must match")
        
        try:
            # Add documents
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"📥 Added {len(documents)} documents to vector store")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, query_embedding: List[float], n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.collection:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except Exception as e:
                logger.error(f"Error getting collection: {e}")
                return []
        
        if not query_embedding:
            logger.warning("Empty query embedding")
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"🔍 Found {len(formatted_results)} relevant chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching collection: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.collection:
                self.collection = self.client.get_collection(self.collection_name)
            
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "exists": True
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "document_count": 0,
                "error": str(e),
                "exists": False
            }
    
    def reset(self) -> bool:
        """Reset the vector store"""
        try:
            if self.collection:
                self.client.delete_collection(self.collection_name)
                self.collection = None
                logger.info(f"🔄 Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False