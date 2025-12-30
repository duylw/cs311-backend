from pydantic import BaseModel, Field
from typing import List


class IngestTopicRequest(BaseModel):
    topic: str = Field(..., min_length=5, description="Research topic to ingest")


class IngestTopicResponse(BaseModel):
    collection_id: int
    topic: str
    total_queries: int
    total_papers: int
    status: str
