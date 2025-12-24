"""
Service Layer - Business Logic
"""
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from loguru import logger
import time
import numpy as np

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from app.schemas.query import RAGResponse
from app.core.config import settings, config_manager

from app.vector_store.faiss_store_manager import FAISSStoreManager

class RAGService:
    """Service for RAG-based question answering"""
    
    def __init__(self):

        """Create LLM"""
        config = config_manager.get_llm_config()
        
        self.llm_model = ChatOllama(
            base_url=config['base_url'],
            model=config['model'],
        )
    
    def query(
        self,
        db: Session,
        query: str,
        collection_id: int,
        top_k: int = 5,
        use_reranking: bool = True
    ) -> RAGResponse:
        """
        Answer a question using RAG
        
        Args:
            db: Database session
            query: User question
            collection_id: Collection to search
            top_k: Number of context chunks
            use_reranking: Whether to use reranking
        
        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()

        embedding = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL_NAME
        )

        # 1. Retrieve relevant chunks
        vector_store = FAISSStoreManager.get_or_create_store(
            collection_id=collection_id,
            embedding=embedding
        )

        # 2. Create retriver
        retriever = vector_store.as_retriever(search_type="similarity",
                                              search_kwargs={"k": top_k * 2})
        # 3. Retrive relevant documents and Re-rank if needed
        search_results = retriever.invoke(query)

        if use_reranking:
            pass  # Reranking logic can be implemented here

        # 4. Build context
        context = self._build_context(search_results)
        
        # 4. Generate answer
        answer = self._generate_answer(self.llm_model, query, context)
        
        execution_time = time.time() - start_time
        
        return RAGResponse(
            query=query,
            answer=answer,
            execution_time=execution_time,
        )
    
    def _build_context(self, results: List[Document]) -> str:
        """Build context from search results"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            
            context_part = f"""
[Source {i}]
Paper: {result.metadata['title']}
Authors: {result.metadata.get('authors', 'N/A')}
Section: {result.metadata.get('section', 'N/A')}

Content:
{result.page_content}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, llm: ChatOllama, query: str, context: str) -> str:
        """Generate answer using LLM"""
        
        prompt = f"""You are a helpful research assistant. Answer the question based on the provided research paper excerpts.

Context from research papers:
{context}

Question: {query}

Instructions:
1. Provide a clear, accurate answer based ONLY on the context provided
2. If the context doesn't contain enough information, say so
3. Cite sources by referring to [Source N] numbers
4. Be concise but comprehensive
5. Use technical terms appropriately

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content
    
    async def stream_query(
        self,
        db: Session,
        query: str,
        collection_id: int,
        top_k: int = 5
    ):
        pass


# Create service instances
rag_service = RAGService()