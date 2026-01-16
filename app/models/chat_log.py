from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, 
    ForeignKey, JSON, Float, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base

class ChatLog(Base):
    """Log user queries and chatbot results of retrieval phase"""
    __tablename__ = "chat_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("collections.id"), nullable=True)
    
    # Chat Info
    text = Column(Text, nullable=False)
    role = Column(String(50), nullable=True)  # user, system, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    collection = relationship("Collection", back_populates="chat_logs")

    __table_args__ = (
        Index('idx_query_collection', 'collection_id'),
        Index('idx_query_created', 'created_at'),
    )