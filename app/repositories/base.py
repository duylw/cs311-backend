"""
Repository Pattern - Data Access Layer
"""
from typing import Generic, TypeVar, Type, Optional, List, Dict, Any, TYPE_CHECKING
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from fastapi import HTTPException, status

from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


# ============================================================================
# Base Repository (app/repositories/base.py)
# ============================================================================

class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations"""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    def get(self, db: Session, id: int) -> Optional[ModelType]:
        """Get single record by ID"""
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_or_404(self, db: Session, id: int) -> ModelType:
        """Get single record or raise 404"""
        obj = self.get(db, id)
        if not obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{self.model.__name__} not found"
            )
        return obj
    
    def get_multi(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """Get multiple records with pagination"""
        query = db.query(self.model)
        
        if filters:
            query = self._apply_filters(query, filters)
        
        return query.offset(skip).limit(limit).all()
    
    def get_count(self, db: Session, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get total count"""
        query = db.query(func.count(self.model.id))
        
        if filters:
            query = self._apply_filters(query, filters)
        
        return query.scalar()
    
    def create(self, db: Session, obj_in: Dict[str, Any]) -> ModelType:
        """Create new record"""
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update(
        self,
        db: Session,
        db_obj: ModelType,
        obj_in: Dict[str, Any]
    ) -> ModelType:
        """Update existing record"""
        for field, value in obj_in.items():
            if value is not None and hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def delete(self, db: Session, id: int) -> bool:
        """Delete record"""
        obj = self.get(db, id)
        if obj:
            db.delete(obj)
            db.commit()
            return True
        return False
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to query"""
        for key, value in filters.items():
            if hasattr(self.model, key):
                if isinstance(value, list):
                    query = query.filter(getattr(self.model, key).in_(value))
                else:
                    query = query.filter(getattr(self.model, key) == value)
        return query