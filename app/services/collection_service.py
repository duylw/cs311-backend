from loguru import logger
from app.services.decompose_service import QueryGenerationService
from app.services.search_service import search_papers
from app.services.ingest_service import IngestService
from app.core.config import settings
from fastapi import HTTPException

class CollectionService:

    @staticmethod
    def ingest_topic(
        collection_id: int,
        topic: str,
        index_name: str,
    ):
        logger.info(f"Start ingest topic: {topic}")

        # 1️⃣ Generate queries
        queries = QueryGenerationService().generate_queries(
            topic=topic,
            min_score=settings.MIN_RERANK_SCORE
        )

        if not queries:
            raise HTTPException(
                status_code=400,
                detail="No valid queries generated from topic"
            )

        # 2️⃣ Search + rerank (abstract-level)
        docs = search_papers(queries)

        # 3️⃣ Deduplicate papers
        unique_arxiv_ids = {
            d.metadata["arxiv_id"] for d in docs
        }

        # 4️⃣ Ingest full PDFs
        IngestService.ingest_papers(
            index_name=index_name,
            papers=docs,
            collection_id=collection_id,
        )

        logger.success("Ingest completed")

        return {
            "queries": len(queries),
            "abstract_hits": len(docs),
            "unique_papers": len(unique_arxiv_ids),
        }
