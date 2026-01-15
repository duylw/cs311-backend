from typing import List, Dict, Optional
from pathlib import Path
import time
from pydantic import BaseModel, Field

class AxisModel(BaseModel):
    axis: str = Field(..., description="Name of the research axis")
    queries: List[str] = Field(
        default_factory=list,
        description="Simple, searchable sub-queries"
    )


class AxisResponse(BaseModel):
    axes: List[AxisModel]


class EvaluatedQuery(BaseModel):
    axis: str
    query: str
    score: int = Field(..., ge=1, le=5)
    keep: bool


class EvaluationResponse(BaseModel):
    results: List[EvaluatedQuery]