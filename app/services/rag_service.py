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

from dotenv import load_dotenv
load_dotenv()

from app.vector_store.pinecone_store_manager import pinecone_manager
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service

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
        decomposed_queries = self._query_translation(query)

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
            
                
            return RAGResponse(
                query=query,
                answer=answer,
                execution_time=execution_time,
            )
    
    def _query_translation(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        
        prompt = prompt_service.get_prompt("retrieval_query_translation").format(query=query)
        
        response = llm_service.call_llm(settings.GOOGLE_LLM_MODEL, prompt).content

        sub_queries = json.loads(response).get("sub_queries", [query])

        return sub_queries

    def _build_context(self, results: List[Dict]) -> str:
        keys = results.keys()
        norm = ""

        for key in keys:
            norm += f"Sub Query: {key}\n"
            docs = results[key]
            for i, doc in enumerate(docs):
                norm += f"  Result {i+1}:\n"
                norm += f"    Content: {doc.page_content}\n"
                norm += f"    Title: {doc.metadata.get('title', 'N/A')}\n"
                norm += f"    Arxiv ID: {doc.metadata.get('arxiv_id', 'N/A')}\n"
                norm += f"    Authors: {doc.metadata.get('authors', 'N/A')}\n"
            norm += "\n"
        return norm   
        
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        
        prompt = prompt_service.get_prompt("generate_answer").format(
            context=context,
            query=query
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