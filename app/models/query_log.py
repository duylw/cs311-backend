from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, 
    ForeignKey, JSON, Float, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base

class QueryLog(Base):
    """Log user queries and results"""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("collections.id"), nullable=True)
    
    # Query Info
    query = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)  # search, rag, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_query_collection', 'collection_id'),
        Index('idx_query_created', 'created_at'),
    )