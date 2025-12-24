from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from fastapi import HTTPException, status


from app.models.collection import Collection
from app.models.query_log import QueryLog
from app.repositories.base import BaseRepository

class QueryLogRepository(BaseRepository[QueryLog]):
    """Repository for QueryLog operations"""
    
    def __init__(self):
        super().__init__(QueryLog)
    
    def create(self, db: Session, obj_in: Dict[str, Any]) -> QueryLog:
        """Create a new query log entry"""
        db_obj = QueryLog(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj