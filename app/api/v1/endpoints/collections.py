from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy.orm import Session
from typing import List, Optional

from app.vector_store.pinecone_store_manager import pinecone_manager
from app.api.deps import get_db
from app.schemas.collection import (
    CollectionCreate,
    CollectionUpdate,
    CollectionResponse,
    CollectionStats,
)
from app.schemas.ingest import (
    IngestTopicRequest,
    IngestTopicResponse,
)

from app.services.collection_service import CollectionService
from app.repositories.collection import collection_repository
from app.repositories.chat_log import chat_log_repository
from app.vector_store import pinecone_store_manager
from app.core.config import settings

router = APIRouter()


@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection: CollectionCreate,
    db: Session = Depends(get_db)
):
    '''Create a new collection'''
    # Create collection
    db_collection = collection_repository.create(db, collection.model_dump())
    
    return db_collection


@router.get("/", response_model=List[CollectionResponse])
async def list_collections(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    '''List all collections'''
    collections = collection_repository.get_multi(db, skip=skip, limit=limit)
    return collections


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: int,
    db: Session = Depends(get_db)
):
    '''Get a specific collection'''
    collection = collection_repository.get_or_404(db, collection_id)
    return collection


@router.get("/{collection_id}/stats", response_model=CollectionStats)
async def get_collection_stats(
    collection_id: int,
    db: Session = Depends(get_db)
):
    '''Get collection statistics'''
    stats = collection_repository.get_with_stats(db, collection_id)
    
    return CollectionStats(
        collection_id=collection_id,
        total_papers=stats['total_papers'],
    )


@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: int,
    collection_update: CollectionUpdate,
    db: Session = Depends(get_db)
):
    '''Update a collection'''
    db_collection = collection_repository.get_or_404(db, collection_id)
    
    updated = collection_repository.update(
        db,
        db_collection,
        collection_update.model_dump(exclude_unset=True)
    )
    
    return updated

@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: int,
    db: Session = Depends(get_db)
):
    '''Delete a collection and all its papers'''
    success = collection_repository.delete(db, collection_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    try:
        pinecone_manager.delete_index_collection(
            index_name=settings.PINECONE_INDEX_NAME,
            collection_id=collection_id,
        )
    except Exception:
        logger.exception("Failed to delete collection from Pinecone for collection_id=%s", collection_id)

    return None

@router.post(
    "/{collection_id}/ingest-topic",
    response_model=IngestTopicResponse,
    status_code=status.HTTP_200_OK
)
async def ingest_topic(
    collection_id: int,
    payload: IngestTopicRequest,
    db: Session = Depends(get_db),
):
    """
    Ingest a research topic:
    - Generate queries
    - Search + rerank papers
    - Fetch PDFs
    - Chunk + store into Pinecone
    """

    collection_repository.get_or_404(db, collection_id)

    try:
        result = CollectionService.ingest_topic(
            db=db,
            collection_id=collection_id,
            topic=payload.topic,
            index_name=settings.PINECONE_INDEX_NAME,
        )

    except Exception as e:
        logger.exception("Ingest topic failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


    raw_hits = result.get("abstract_hits", [])

    abstract_hits = [
        {
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in raw_hits
    ]

    return IngestTopicResponse(
        collection_id=collection_id,
        topic=payload.topic,
        queries=result.get("queries"),
        abstract_hits=abstract_hits,
        unique_papers=result.get("unique_papers"),
        status="success",
    )

@router.get("/{collection_id}/chat-history")
async def get_chat_history(
    collection_id: int,
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get chat history
    
    - **collection_id**: Filter by collection (optional)
    - **limit**: Number of recent chat messages to return
    """
    logs = chat_log_repository.get_with_collection(collection_id=collection_id, limit=limit, db=db)
    
    return {
        'total': len(logs),
        'queries': [
            {
                'id': log.id,
                'text': log.text,
                'role': log.role,
                'collection_id': log.collection_id,
                'created_at': log.created_at
            }
            for log in logs
        ]
    }