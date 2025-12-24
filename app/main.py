from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import time

from contextlib import asynccontextmanager
from app.core.config import settings
from app.api.v1.router import api_router
from app.db.session import engine
from app.db.base import Base

from dotenv import load_dotenv
load_dotenv()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run on startup"""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"API documentation: {settings.API_V1_STR}/docs")
    
    # Initialize FAISS stores
    from app.vector_store.faiss_store_manager import faiss_manager
    logger.info("FAISS vector store initialized")
    
    # Log configuration
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL_NAME}")
    logger.info(f"LLM: {settings.OLLAMA_MODEL}")

    # Create tables
    try:
        import app.models.collection
        import app.models.paper
        import app.models.query_log
        Base.metadata.create_all(bind=engine)
    except Exception:
        logger.exception("Failed to create tables")

    yield
    
    """Run on shutdown"""
    logger.info("Shutting down application")
    
    # Save FAISS indices
    from app.vector_store.faiss_store_manager import faiss_manager
    logger.info("Saving FAISS indices...")
    faiss_manager.save_all()
    logger.info("Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"{request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} "
            f"completed in {process_time:.2f}s with status {response.status_code}"
        )
        
        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.LOG_LEVEL == "DEBUG" else "An error occurred"
        }
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": "production" if settings.LOG_LEVEL == "INFO" else "development"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "docs": f"{settings.API_V1_STR}/docs",
        "openapi": f"{settings.API_V1_STR}/openapi.json"
    }


# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# CLI runner
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
