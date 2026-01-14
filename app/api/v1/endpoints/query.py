"""
API Endpoints for retrieval phase in RAG
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.query import RAGQuery, RAGResponse
from app.services.rag_service import rag_service
from app.repositories.collection import collection_repository


router_query = APIRouter()

@router_query.post("/ask", response_model=RAGResponse)
async def ask_question(
    rag_query: RAGQuery,
    db: Session = Depends(get_db)
):
    """
    Ask a question and get an AI-generated answer based on research papers
    
    - **query**: Your question
    - **collection_id**: Collection to search in
    - **top_k**: Number of context chunks to use
    - **use_reranking**: Whether to rerank results for better quality
    """
    # Verify collection exists
    collection = collection_repository.get_or_404(db, rag_query.collection_id)
    
    if not collection.total_papers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Collection has no indexed papers. Please index some papers first."
        )
    
    # Generate answer
    response = rag_service.query(
        db=db,
        query=rag_query.query,
        collection_id=rag_query.collection_id,
        top_k=rag_query.top_k,
        use_reranking=rag_query.use_reranking
    )
    
    return response


@router_query.post("/ask/stream")
async def ask_question_stream(
    rag_query: RAGQuery,
    db: Session = Depends(get_db)
):
    """
    Ask a question with streaming response
    
    Returns a stream of text chunks as they are generated.
    """
    pass