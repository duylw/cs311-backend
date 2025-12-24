from fastapi import APIRouter
from app.api.v1.endpoints import query, collections

api_router = APIRouter()

api_router.include_router(
    collections.router,
    prefix="/collections",
    tags=["collections"]
)

# api_router.include_router(
#     papers.router,
#     prefix="/papers",
#     tags=["papers"]
# )

# api_router.include_router(
#     search.router_search,
#     prefix="/search",
#     tags=["search"]
# )

api_router.include_router(
    query.router_query,
    prefix="/query",
    tags=["query"]
)