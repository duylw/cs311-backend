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
        start_time = time.time()

        """TODO:
        - Query Decomposition
        - Rerank
        """

        # 1. Query Decomposition
        decomposed_queries = self._query_decompose(query)

        # 2. Get or create vector store coresponding to collection_id
        vector_store = pinecone_manager.get_store(
            index_name=settings.PINECONE_INDEX_NAME,
        )

        # If no store loaded in memory, return a safe response
        if vector_store is None:
            logger.warning("No Pinecone store loaded for index %s", settings.PINECONE_INDEX_NAME)
            execution_time = time.time() - start_time
            return RAGResponse(
                query=query,
                answer="No vector store available for retrieval. Please create or load the index first.",
                execution_time=execution_time,
            )

        elif vector_store:
            # 3. Create retriver for that collection_id
            logger.info(f"Creating retriever for collection_id: {collection_id}")
            logger.info(f"Embedding model used: {vector_store.embeddings.model}")
            

            retriever = vector_store.as_retriever(search_type="similarity",
                                                search_kwargs={"k": top_k,
                                                                "filter": {"collection_id": collection_id}
                                                                })
            
            # 4. Retrieve relevant documents and Re-rank if needed
            res = {}
            for sub_query in decomposed_queries:
                res[sub_query] = retriever.invoke(sub_query)

            if use_reranking:
                pass  # Reranking logic can be implemented here

            # 5. Build context
            context = self._build_context(res)
            
            # 6. Generate answer
            answer = self._generate_answer(query,
                                           context)
            
            execution_time = time.time() - start_time
            
            # Log Chat (For User)
            chat_log_repository.create(db, {
                'collection_id': collection_id,
                'text': query,
                'role': 'user',
            })

            # Log Chat (For BOT)
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
    
    def _query_decompose(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        
        prompt = prompt_service.get_prompt("retrieval_query_decompose").format(query=query)
        
        llm_structured = llm_service.get_llm(settings.GOOGLE_LLM_MODEL).with_structured_output(ListSubQueries)

        response = llm_structured.invoke(prompt)

        sub_queries = response.sub_queries

        return sub_queries

    def _build_context(self, results: List[Dict]) -> str:
        keys = results.keys()
        norm = ""

        for key in keys:
            norm += f"Sub Query: {key}\n"
            docs = results[key]
            for i, doc in enumerate(docs):
                norm += f"From Paper: {doc.metadata.get('title', 'N/A').strip()}\n"
                norm += f"From Section: {doc.metadata.get('section', 'N/A')}:\n"
                norm += f"Content: {doc.page_content.strip()}\n"
            norm += "\n"
        return norm   
        
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        
        prompt = prompt_service.get_prompt("retrieval_generate_answer").format(
            query=query,
            context=context
        )
        
        response = llm_service.call_llm(settings.GOOGLE_LLM_MODEL, prompt)
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