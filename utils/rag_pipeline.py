"""
RAG Pipeline with Groq API Integration (Direct SDK)
"""
import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline with Groq LLM"""
    
    def __init__(self, pdf_loader, text_splitter, embedding_generator, vector_store):
        self.pdf_loader = pdf_loader
        self.text_splitter = text_splitter
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        
        # Initialize Groq client directly
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", "gemma2-9b-it"
        
        logger.info(f"🤖 Groq initialized with model: {self.model_name}")
    
    def process_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Process PDFs and store in vector store"""
        try:
            documents = self.pdf_loader.load_pdfs(pdf_paths)
            if not documents:
                return {
                    "success": False,
                    "error": "No documents could be loaded",
                    "chunks": 0
                }
            
            chunks = self.text_splitter.split_documents(documents)
            if not chunks:
                return {
                    "success": False,
                    "error": "No text chunks could be created",
                    "chunks": 0
                }
            
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            metadatas = []
            for chunk in chunks:
                metadata = chunk.metadata.copy()
                metadata['chunk_size'] = len(chunk.page_content)
                metadatas.append(metadata)
            
            self.vector_store.create_collection(
                self.embedding_generator.get_embedding_dimension()
            )
            
            added_count = self.vector_store.add_documents(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return {
                "success": True,
                "chunks": added_count,
                "pages": len(documents),
                "files_processed": len(pdf_paths)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks": 0
            }
    
    def ask_question(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Ask a question using RAG pipeline"""
        try:
            # Generate embedding for question
            question_embedding = self.embedding_generator.generate_embedding(question)
            
            # Retrieve relevant chunks
            results = self.vector_store.search(question_embedding, n_results=n_results)
            
            if not results:
                return {
                    "answer": "❌ I don't have enough information in the uploaded documents to answer this question. Please upload relevant PDFs first.",
                    "sources": [],
                    "error": True
                }
            
            # Prepare context
            context = "\n\n---\n\n".join([result['document'] for result in results])
            
            # Create prompt
            prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "I don't have enough information to answer this question"
3. Be concise, accurate, and specific
4. Use complete sentences

Answer:"""
            
            # Use Groq client directly
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            answer = completion.choices[0].message.content
            
            # Prepare sources
            sources = []
            for i, result in enumerate(results, 1):
                relevance_score = 1 - result['distance'] if result['distance'] else 1.0
                sources.append({
                    "content": result['document'],
                    "metadata": result['metadata'],
                    "relevance_score": round(relevance_score, 3),
                    "rank": i
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "error": False
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"❌ Error generating answer: {str(e)}",
                "sources": [],
                "error": True
            }
    
    def reset(self) -> bool:
        """Reset the pipeline"""
        return self.vector_store.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.vector_store.get_collection_stats()
        return {
            "model": self.model_name,
            "embedding_model": self.embedding_generator.model_name,
            "vector_store": stats,
            "chunk_size": self.text_splitter.chunk_size,
            "chunk_overlap": self.text_splitter.chunk_overlap
        }