from sqlalchemy.orm import Session
from typing import Optional, List

from app.repositories.paper import paper_repository
from app.models.paper import Paper


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

paper_service = PaperService()
