from pydantic import BaseModel, Field
from typing import List, Optional


class IngestTopicRequest(BaseModel):
    topic: str = Field(..., min_length=5, description="Research topic to ingest")


class IngestTopicResponse(BaseModel):
    collection_id: int
    topic: str
    queries: Optional[int]
    abstract_hits: Optional[int]
    unique_papers: Optional[int]
    status: str
