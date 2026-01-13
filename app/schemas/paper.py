from pydantic import BaseModel, Field, field_validator 
from typing import Optional, List, Dict, Any
from datetime import datetime

class PaperBase(BaseModel):
    arxiv_id: str = Field(..., description="Paper identifier (arXiv / URL)")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(..., description="Paper abstract")

    class Config:
        from_attributes = True


class PaperCreate(PaperBase):
    collection_id: int
    authors: Optional[List[str]] = Field(
        default=None,
        description="List of authors"
    )
    pdf_url: Optional[str] = None

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("authors cannot be an empty list")
        return v


class PaperResponse(PaperBase):
    id: int
    collection_id: int
    authors: List[str] = Field(default_factory=list)
    pdf_url: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
