"""
Core Configuration using Pydantic Settings
"""
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path
import secrets


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Research Paper RAG API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Production RAG system for research papers"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    # Allow any database URL (Postgres, SQLite, etc.) — set via env `DATABASE_URL`.
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")

    # File Storage
    UPLOAD_DIR: Path = Field(default=Path("./data/uploads"))
    PAPERS_DIR: Path = Field(default=Path("./data/papers"))
    PARSED_DIR: Path = Field(default=Path("./data/parsed"))
    IMAGES_DIR: Path = Field(default=Path("./data/images"))
    FAISS_INDEX_DIR: Path = Field(default=Path("./data/faiss_indices"))
    
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = Field(
        default="nomic-embed-text:latest",
        env="EMBEDDING_MODEL_NAME"
    )
    
    # LLM Configuration
    LLM_PROVIDER: str = Field(default="ollama", env="LLM_PROVIDER")  # ollama, openai, etc.
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="llama3.2", env="OLLAMA_MODEL")
    
    # Chunking Configuration
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Search Configuration
    ARXIV_MAX_RESULTS: int = Field(default=50, env="ARXIV_MAX_RESULTS")
    ARXIV_DELAY: float = Field(default=1.0, env="ARXIV_DELAY")
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = Field(default=5, env="RETRIEVAL_TOP_K")
    RERANK_TOP_K: int = Field(default=10, env="RERANK_TOP_K")
    USE_RERANKING: bool = Field(default=True, env="USE_RERANKING")

    
    # PDF Processing
    PDF_EXTRACT_IMAGES: bool = Field(default=True, env="PDF_EXTRACT_IMAGES")
    PDF_EXTRACT_TABLES: bool = Field(default=True, env="PDF_EXTRACT_TABLES")
    PDF_USE_OCR: bool = Field(default=False, env="PDF_USE_OCR")
    
    # Filtering
    MIN_PUBLICATION_YEAR: int = Field(default=2015, env="MIN_PUBLICATION_YEAR")
    MIN_ABSTRACT_LENGTH: int = Field(default=100, env="MIN_ABSTRACT_LENGTH")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = Field(default=20, env="DEFAULT_PAGE_SIZE")
    MAX_PAGE_SIZE: int = Field(default=100, env="MAX_PAGE_SIZE")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default="logs/app.log", env="LOG_FILE")
    
    # Background Tasks
    INDEXING_QUEUE_NAME: str = "indexing_queue"
    SEARCH_QUEUE_NAME: str = "search_queue"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()

# Create necessary directories
for directory in [
    settings.UPLOAD_DIR,
    settings.PAPERS_DIR,
    settings.PARSED_DIR,
    settings.IMAGES_DIR,
    settings.FAISS_INDEX_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Dynamic configuration loader
class ConfigManager:
    """Manage dynamic configuration changes"""
    
    def __init__(self):
        self._settings = settings
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return getattr(self._settings, key, default)
    
    def update(self, **kwargs):
        """Update configuration (runtime)"""
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
    
    def get_embedding_config(self):
        """Get embedding configuration"""
        return {
            "model_name": self._settings.EMBEDDING_MODEL_NAME,
            "device": self._settings.EMBEDDING_MODEL_DEVICE,
            "batch_size": self._settings.EMBEDDING_BATCH_SIZE,
            "dimension": self._settings.VECTOR_DIMENSION,
        }
    
    def get_llm_config(self):
        """Get LLM configuration"""
        return {
            "provider": self._settings.LLM_PROVIDER,
            "base_url": self._settings.OLLAMA_BASE_URL,
            "model": self._settings.OLLAMA_MODEL,
        }
    
    def get_retrieval_config(self):
        """Get retrieval configuration"""
        return {
            "top_k": self._settings.RETRIEVAL_TOP_K,
            "rerank_top_k": self._settings.RERANK_TOP_K,
            "use_reranking": self._settings.USE_RERANKING,
        }


config_manager = ConfigManager()