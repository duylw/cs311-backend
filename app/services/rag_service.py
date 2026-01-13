"""
Service Layer - Business Logic
"""
from typing import List, Dict
from sqlalchemy.orm import Session
from loguru import logger
import time
import json

from app.schemas.query import RAGResponse
from app.core.config import settings

from app.vector_store.pinecone_store_manager import pinecone_manager
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service
from pydantic import BaseModel, Field

from app.repositories.chat_log import chat_log_repository

from app.services.retrieval_graph import chatbot

class ListSubQueries(BaseModel):
    sub_queries: List[str] = Field(
        description="List of sub-queries derived from the main query"
    )

class RAGService:
    """Service for RAG-based question answering"""
    
    def __init__(self):
        pass
    
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
        # Log Chat (For User)
        chat_log_repository.create(db, {
            'collection_id': collection_id,
            'text': query,
            'role': 'user',
        })

        start_time = time.time()

        """TODO:
        - Query Decomposition
        - Rerank
        """

        config = {
            "configurable": {"thread_id": collection_id}
        }
        
        respond = chatbot.invoke({
            "messages": [("user", query)],
            "collection_id": collection_id,
            "top_k": top_k,
            "use_reranking": use_reranking,
        }, config)

        answer = respond["messages"][-1].content    
        
        execution_time = time.time() - start_time
        
     

        # Log Chat (For Assistant)
        chat_log_repository.create(db, {
            'collection_id': collection_id,
            'text': answer,
            'role': 'assistant',
        })
            
        return RAGResponse(
            query=query,
            answer=answer,
            execution_time=execution_time,
        )
    
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