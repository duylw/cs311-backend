from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from fastapi import HTTPException, status


from app.models.collection import Collection
from app.models.paper import Paper
from app.repositories.base import BaseRepository

class CollectionRepository(BaseRepository[Collection]):
    """Repository for Collection operations"""
    
    def __init__(self):
        super().__init__(Collection)
    
    def get_by_name(self, db: Session, name: str) -> Optional[Collection]:
        """Get collection by name"""
        return db.query(Collection).filter(Collection.name == name).first()
    
    def get_with_stats(self, db: Session, collection_id: int) -> Dict[str, Any]:
        """Get collection with statistics"""
        collection = self.get_or_404(db, collection_id)
        
        # Get statistics
        total_papers = db.query(func.count(Paper.id))\
            .filter(Paper.collection_id == collection_id)\
            .scalar()

        return {
            "collection": collection,
            "total_papers": total_papers,
        }

collection_repository = CollectionRepository()