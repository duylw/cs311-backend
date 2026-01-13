from pydantic import BaseModel, Field
from typing import List, Optional


class IngestTopicRequest(BaseModel):
    topic: str = Field(..., min_length=5, description="Research topic to ingest")

class AbstractHit(BaseModel):
    content: str
    metadata: dict

class IngestTopicResponse(BaseModel):
    collection_id: int
    topic: str
    queries: Optional[list[dict]]
    abstract_hits: Optional[list[AbstractHit]]
    unique_papers: Optional[int]
    status: str
