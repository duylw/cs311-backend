from pydantic import BaseModel, Field, field_validator 
from typing import Optional, List, Dict, Any
from datetime import datetime

class PaperBase(BaseModel):
    arxiv_id: str = Field(..., description="Arxiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    
    @field_validator('authors')
    def validate_authors(cls, v):
        if not v:
            raise ValueError("Authors list cannot be empty")
        return v

class PaperCreate(PaperBase):
    collection_id: int
    updated_date: Optional[datetime] = None
    pdf_url: str

class PaperResponse(PaperBase):
    collection_id: int
    pdf_url: str
    pass