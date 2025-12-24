from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.deps import get_db
from app.schemas.collection import (
    CollectionCreate,
    CollectionUpdate,
    CollectionResponse,
    CollectionStats
)
from app.repositories.collection import collection_repository

router = APIRouter()


@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection: CollectionCreate,
    db: Session = Depends(get_db)
):
    '''Create a new collection'''
    # Check if collection already exists
    existing = collection_repository.get_by_name(db, collection.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Collection with this name already exists"
        )
    
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
        # Delete FAISS index
        from app.vector_store.faiss_store_manager import faiss_manager
        faiss_manager.delete_store(collection_id)
    except Exception as e:
        pass  # Log error if needed
    
    return None