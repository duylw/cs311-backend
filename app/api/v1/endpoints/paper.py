"""
API Endpoints for Papers
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List

from app.api.deps import get_db
from app.schemas.paper import PaperResponse
from app.services.paper_service import paper_service
from app.repositories.collection import collection_repository

router_papers = APIRouter()
@router_papers.get(
    "/collections/{collection_id}/papers",
    response_model=List[PaperResponse]
)
def list_papers(
    collection_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    collection_repository.get_or_404(db, collection_id)

    papers = paper_service.list_by_collection(
        db, collection_id, skip, limit
    )

    return [
        PaperResponse(
            id=p.id,
            arxiv_id=p.arxiv_id,
            title=p.title,
            authors=[a.strip() for a in p.authors.split(",")],
            abstract=p.abstract,
            collection_id=p.collection_id,
            pdf_url=p.pdf_url,
            updated_at=p.updated_at,
            created_at=p.created_at,
        )
        for p in papers
]


