"""
SQLAlchemy Models for Research Papers
"""
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, 
    ForeignKey, JSON, Float, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from app.db.base import Base

class Paper(Base):
    """Research paper metadata"""
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    
    # Arxiv Info
    arxiv_id = Column(String(50), nullable=False, index=True, unique=True)
    title = Column(Text, nullable=False)
    authors = Column(String, nullable=False)  # List of author names
    abstract = Column(Text, nullable=True)
  
    # External References
    pdf_url = Column(String(512), nullable=False)
    is_ingested = Column(Boolean, default=False)
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    collection = relationship("Collection", back_populates="papers")
    
    __table_args__ = (
        Index('idx_paper_arxiv_id', 'arxiv_id'),
        Index('idx_paper_collection', 'collection_id'),
    )