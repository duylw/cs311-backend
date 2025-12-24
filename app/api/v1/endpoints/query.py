"""
API Endpoints for Search and RAG
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import asyncio

from app.api.deps import get_db
from app.schemas.query import RAGQuery, RAGResponse, RAGStreamChunk
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
    
    # Log query
    from app.repositories.paper import query_log_repository
    query_log_repository.create(db, {
        'collection_id': rag_query.collection_id,
        'query': rag_query.query,
        'query_type': 'rag',
        'execution_time': response.execution_time,
    })
    
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

# ============================================================================
# Query History Endpoints
# ============================================================================

@router_query.get("/history")
async def get_query_history(
    collection_id: Optional[int] = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get query history
    
    - **collection_id**: Filter by collection (optional)
    - **limit**: Number of recent queries to return
    """
    from app.models.query_log import QueryLog
    
    query = db.query(QueryLog)
    
    if collection_id:
        query = query.filter(QueryLog.collection_id == collection_id)
    
    logs = query.order_by(QueryLog.created_at.desc()).limit(limit).all()
    
    return {
        'total': len(logs),
        'queries': [
            {
                'id': log.id,
                'query': log.query,
                'query_type': log.query_type,
                'execution_time': log.execution_time,
                'created_at': log.created_at
            }
            for log in logs
        ]
    }