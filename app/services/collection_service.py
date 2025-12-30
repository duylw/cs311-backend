from loguru import logger
from app.services.decompose_service import QueryGenerationService
from app.services.search_service import search_papers
from app.services.ingest_service import IngestService
from app.core.config import settings
class CollectionService:

    @staticmethod
    def ingest_topic(
        collection_id: int,
        topic: str,
        index_name: str,
    ):
        logger.info(f"Start ingest topic: {topic}")
        min_score = settings.MIN_RERANK_SCORE
        # 1. Generate + evaluate queries
        queries = QueryGenerationService().generate_queries(topic=topic, 
                                                            min_score=min_score)

        # 2. Search + rerank arxiv (abstract level)
        papers = search_papers(queries)

        # 3. Fetch PDF + chunk + ingest Pinecone
        IngestService.ingest_papers(
            index_name=index_name,
            papers=papers,
            collection_id=collection_id,
        )

        logger.success("Ingest completed")

        return {
            "total_queries": len(queries),
            "total_papers": len(papers),
        }
