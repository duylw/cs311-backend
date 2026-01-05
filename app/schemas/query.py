
from pydantic import BaseModel, Field, field_validator 
from typing import Optional, List, Dict, Any
from datetime import datetime


class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    collection_id: int = Field(..., description="Collection to query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks")
    use_reranking: bool = Field(default=True, description="Use reranking")
    stream: bool = Field(default=False, description="Stream response")

class RAGResponse(BaseModel):
    query: str
    answer: str
    execution_time: float


class QueryLogResponse(BaseModel):
    id: int
    query: str
    query_type: str
    execution_time: float
    created_at: datetime