"""
Embedding generation module using Sentence Transformers
"""
import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles embedding generation with caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.batch_size = 32
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"🧠 Loaded embedding model: {model_name}")
            logger.info(f"📊 Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("No texts to embed")
            return []
        
        try:
            # Process in batches
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
            
            # Convert to list of lists
            result = [embedding.tolist() for embedding in embeddings]
            logger.info(f"Generated {len(result)} embeddings")
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding"""
        return self.embedding_dim
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(max(0, min(1, similarity)))  # Clamp between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0