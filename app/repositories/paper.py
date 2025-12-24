# ============================================================================
# Paper Repository (app/repositories/paper.py)
# ============================================================================
from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, String
from fastapi import HTTPException, status

from app.repositories.base import BaseRepository
from app.models.paper import Paper
from datetime import datetime, timedelta


class PaperRepository(BaseRepository[Paper]):
    """Repository for Paper operations"""
    
    def __init__(self):
        super().__init__(Paper)
    
    def get_by_arxiv_id(self, db: Session, arxiv_id: str) -> Optional[Paper]:
        """Get paper by Arxiv ID"""
        return db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    
    def get_by_collection(
        self,
        db: Session,
        collection_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Paper]:
        """Get papers in a collection"""
        return db.query(Paper)\
            .filter(Paper.collection_id == collection_id)\
            .offset(skip)\
            .limit(limit)\
            .all()
    
    def search_papers(
        self,
        db: Session,
        query: str,
        collection_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Paper]:
        """Full-text search in papers"""
        search_query = db.query(Paper)
        
        if collection_id:
            search_query = search_query.filter(Paper.collection_id == collection_id)
        
        # Search in title, abstract, and authors
        search_filter = or_(
            Paper.title.ilike(f"%{query}%"),
            Paper.abstract.ilike(f"%{query}%"),
            Paper.authors.cast(String).ilike(f"%{query}%")
        )
        
        return search_query.filter(search_filter)\
            .offset(skip)\
            .limit(limit)\
            .all()
    
    def get_by_category(
        self,
        db: Session,
        category: str,
        collection_id: Optional[int] = None
    ) -> List[Paper]:
        """Get papers by category"""
        query = db.query(Paper).filter(
            Paper.primary_category == category
        )
        
        if collection_id:
            query = query.filter(Paper.collection_id == collection_id)
        
        return query.all()
    
    def update_indexing_status(
        self,
        db: Session,
        paper_id: int,
        is_downloaded: bool = None,
        is_parsed: bool = None,
        is_indexed: bool = None
    ) -> Paper:
        """Update paper processing status"""
        paper = self.get_or_404(db, paper_id)
        
        if is_downloaded is not None:
            paper.is_downloaded = is_downloaded
        if is_parsed is not None:
            paper.is_parsed = is_parsed
        if is_indexed is not None:
            paper.is_indexed = is_indexed
        
        db.commit()
        db.refresh(paper)
        return paper
    
    def bulk_create(self, db: Session, papers: List[Dict[str, Any]]) -> List[Paper]:
        """Bulk create papers"""
        db_papers = [Paper(**paper) for paper in papers]
        db.bulk_save_objects(db_papers)
        db.commit()
        return db_papers
    
paper_repository = PaperRepository()
