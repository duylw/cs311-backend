from loguru import logger
from app.services.decompose_service import query_generation_service
from app.services.search_service import search_service
from app.services.ingest_service import IngestService
from app.core.config import settings
from app.repositories.collection import collection_repository
from sqlalchemy.orm import Session
class CollectionService:

    @staticmethod
    def ingest_topic(
        collection_id: int,
        topic: str,
        db: Session,
        index_name: str,
    ):
        logger.info(f"Start ingest topic: {topic}")
        # 1. Generate + evaluate queries
        queries = query_generation_service.generate_queries(topic=topic)
       
        # 2. Search + rerank arxiv (abstract level)
        assert isinstance(queries, list)

        for q in queries:
            assert isinstance(q["query"], str)
            assert not q["query"].startswith("[")
        docs = search_service(queries)

        # 3. Fetch PDF + chunk + ingest Pinecone
        IngestService.ingest_search_results(
            index_name=index_name,
            docs=docs,
            collection_id=collection_id,
        )

         # 4. Update DB collection stats
        collection = collection_repository.get_or_404(db, collection_id)
        new_total = (collection.total_papers or 0) + len(docs)
        collection_repository.update(
            db,
            collection,
            {"total_papers": new_total}
        )

        logger.success(
            f"Ingest completed | papers={len(docs)} | collection_id={collection_id}"
        )

        return {
            "queries": queries,
            "abstract_hits": docs,
            "unique_papers": len(docs),
            "total_papers": new_total,
        }
