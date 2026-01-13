from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from fastapi import HTTPException, status


from app.models.collection import Collection
from app.models.chat_log import ChatLog
from app.repositories.base import BaseRepository

class ChatLogRepository(BaseRepository[ChatLog]):
    """Repository for ChatLog operations"""
    
    def __init__(self):
        super().__init__(ChatLog)
    
    def create(self, db: Session, obj_in: Dict[str, Any]) -> ChatLog:
        """Create a new query log entry"""
        db_obj = ChatLog(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get_with_collection(self, db, collection_id: int, limit: int) -> List[ChatLog]:
        """Retrieve chat logs for a specific collection"""
        return db.query(ChatLog).filter(ChatLog.collection_id == collection_id).order_by(ChatLog.created_at.desc()).limit(limit).all()

chat_log_repository = ChatLogRepository()