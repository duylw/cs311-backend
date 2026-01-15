from sqlalchemy.orm import Session
from typing import Optional, List

from app.repositories.paper import paper_repository
from app.models.paper import Paper
from fastapi import HTTPException, status
from app.vector_store.pinecone_store_manager import pinecone_manager
from app.core.config import settings

class PaperService:

    @staticmethod
    def list_by_collection(
        db: Session,
        collection_id: int,
        skip: int,
        limit: int
    ) -> List[Paper]:
        return paper_repository.get_by_collection(
            db, collection_id, skip, limit
        )

    @staticmethod
    def delete_paper_from_collection(db: Session, paper_id: int, collection_id: int) -> bool:
        """Delete a paper from a collection"""
        # Get paper or raise 404
        paper = paper_repository.get_or_404(db, paper_id)
        
        # Verify the paper belongs to the collection
        if paper.collection_id != collection_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Paper not found in this collection"
            )
        
        pinecone_manager.delete_document_of_collection(settings.PINECONE_INDEX_NAME, collection_id, paper.arxiv_id)

        return paper_repository.delete(db, paper_id)
    

paper_service = PaperService()
