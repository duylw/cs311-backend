"""
Service Layer - Business Logic
"""
from typing import List, Dict
from sqlalchemy.orm import Session
from loguru import logger
import time

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.schemas.query import RAGResponse
from app.core.config import settings, config_manager

from app.vector_store.pinecone_store_manager import pinecone_manager

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
            # base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL_NAME
        )
        """TODO:
        - Query Decomposition
        - Rerank
        """

        # 1. Query Decomposition
        decomposed_queries = self._query_decomposition(query)

        # 2. Get or create vector store coresponding to collection_id
        vector_store = pinecone_manager.get_store(
            index_name=settings.PINECONE_INDEX_NAME,
        )

        # If no store loaded in memory, return a safe response (don't create)
        if vector_store is None:
            logger.warning("No Pinecone store loaded for index %s", settings.PINECONE_INDEX_NAME)
            execution_time = time.time() - start_time
            return RAGResponse(
                query=query,
                answer="No vector store available for retrieval. Please create or load the index first.",
                execution_time=execution_time,
            )

        # Ensure embedding model compatibility
        existing_embedding = getattr(vector_store, "embeddings", None)
        existing_model = getattr(existing_embedding, "model", None)

        # If models mismatch, return a safe response (do not recreate index)
        if existing_model != embedding.model:
            logger.error(
                "Embedding model mismatch for index '%s': existing=%s requested=%s",
                settings.PINECONE_INDEX_NAME,
                existing_model,
                embedding.model,
            )
            execution_time = time.time() - start_time
            return RAGResponse(
                query=query,
                answer=(
                    f"Embedding model mismatch: index uses '{existing_model}' "
                    f"but the request uses '{embedding.model}'. Re-index with the same model or use the matching model."
                ),
                execution_time=execution_time,
            )


        if vector_store:
            # 3. Create retriver for that collection_id
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
            answer = self._generate_answer(self.llm_model, query, context)
            
            execution_time = time.time() - start_time
            
                
            return RAGResponse(
                query=query,
                answer=answer,
                execution_time=execution_time,
            )
    
    def _query_decomposition(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        # Placeholder for actual decomposition logic
        return [query]

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
                norm += f"    Score: {doc.metadata.get('score', 'N/A')}\n"
            norm += "\n"
        return norm   
        
    
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