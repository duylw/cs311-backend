import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from app.core.config import settings

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class FAISSStoreManager:
    """Manage multiple FAISS stores for different collections"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or settings.FAISS_INDEX_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.stores: Dict[int, FAISS] = {}
    
    def get_or_create_store(
        self,
        collection_id: int,
        embedding: OllamaEmbeddings,
    ) -> FAISS:
        """Get or create a FAISS store for a collection"""
        
        if collection_id in self.stores:
            return self.stores[collection_id]
        
        # Try to load existing index
        index_path = self.base_dir / f"collection_{collection_id}"
        
        if index_path.with_suffix('.faiss').exists():
            logger.info(f"Loading existing index for collection {collection_id}")
            store = FAISS.load_local(index_path)
        else:
            logger.info(f"Creating new index for collection {collection_id}")
            
            index = faiss.IndexFlatL2(len(embedding.embed_query("hello world")))

            store = FAISS(
                embedding=embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        
        self.stores[collection_id] = store
        return store
    
    def save_store(self, collection_id: int):
        """Save a specific collection's store"""
        if collection_id in self.stores:
            index_path = self.base_dir / f"collection_{collection_id}"
            self.stores[collection_id].save_local(index_path)
    
    def save_all(self):
        """Save all stores"""
        for collection_id in self.stores:
            self.save_store(collection_id)
    
    def delete_store(self, collection_id: int):
        """Delete a collection's store"""
        if collection_id in self.stores:
            del self.stores[collection_id]
        
        # Delete files
        index_path = self.base_dir / f"collection_{collection_id}"
        for suffix in ['.faiss', '.pkl']:
            file_path = index_path.with_suffix(suffix)
            if file_path.exists():
                file_path.unlink()
        
        logger.info(f"Deleted index for collection {collection_id}")

# Global instance
faiss_manager = FAISSStoreManager()