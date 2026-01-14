from typing import Dict, Optional
from loguru import logger
from sqlalchemy import String

from app.core.config import settings

# from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from app.services.embedding_service import embedding_service

class PineconeStoreManager:
    """Manage multiple Pinecone indexes"""
    
    def __init__(self):
        self.vector_stores: Dict[str, PineconeVectorStore] = {}


    def _load_vector_store(self, index_name: str) -> PineconeVectorStore:
        """Load an existing Pinecone store
        If not exists, create a new one"""
        pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())
        
        # Default to Google embedding
        embedding = embedding_service.get_embedding(settings.GOOGLE_EMBEDDING_MODEL)

        # If has index
        if pinecone_client.has_index(index_name):
            logger.info(f"Loading existing Pinecone index: {index_name}")
            index = pinecone_client.Index(index_name)

            # Note: Embedding model is not stored in Pinecone, so we cannot retrieve it here
            vector_store = PineconeVectorStore(index=index, embedding=embedding)
            logger.info(f"Pinecone index {index_name} loaded successfully")
        else:
            logger.info(f"Creating new Pinecone index: {index_name}")
            
            # Create new index (Use default embedding model from settings)
            logger.info(f"Creating Pinecone index {index_name} with embedding model {embedding.model}")
            dimension = len(embedding.embed_query("hello world"))
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            index = pinecone_client.Index(index_name)
            vector_store = PineconeVectorStore(index=index, embedding=embedding)
            logger.info(f"Pinecone index {index_name} created successfully")
            
        self.vector_stores[index_name] = vector_store        

    def get_store(
        self,
        index_name: str,
    ) -> Optional[PineconeVectorStore]:
        """Return existing Pinecone store or None"""

        logger.info(f"Getting Pinecone index: {index_name}")
        store = self.vector_stores.get(index_name)
        if store is None:
            logger.debug(f"No Pinecone store loaded for index: {index_name}")
            store = self._load_vector_store(index_name)

        return store

    def delete_documents_of_collection(
        self,
        index_name: str,
        collection_id: int,
        arxiv_id: str,
        namespace: str = "__default__",
    ):
        vector_store = self.vector_stores.get(index_name)
        if not vector_store:
            logger.warning("Pinecone index %s not found (delete documents)", index_name)
            return
        filter = {"collection_id": collection_id, "arxiv_id": arxiv_id}
        vector_store.delete(filter=filter, namespace=namespace)

    def delete_index_collection(self, index_name: str, collection_id: int):
        logger.info(
            "Deleting all documents from Pinecone index %s for collection %s",
            index_name,
            collection_id,
        )
        vector_store = self.vector_stores.get(index_name)
        if not vector_store:
            logger.warning("Pinecone index %s not found (delete collection)", index_name)
            return
        filter = {"collection_id": collection_id}
        vector_store.delete(filter=filter, namespace="__default__")

# Global instance
pinecone_manager = PineconeStoreManager()