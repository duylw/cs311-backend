from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
from loguru import logger

class EmbeddingService:
    def __init__(self):
        self.embedding_models = {
            # settings.OLLAMA_EMBEDDING_MODEL_NAME: OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL_NAME),
            settings.GOOGLE_EMBEDDING_MODEL: GoogleGenerativeAIEmbeddings(model=settings.GOOGLE_EMBEDDING_MODEL,
                                                                          api_key=settings.GOOGLE_GCP_API_KEY.get_secret_value(),
                                                                          output_dimensionality=768)
        }
    def get_embedding(self, model_name: str):
        return self.embedding_models.get(model_name)
    
    def embed_query(self, query: str, model_name: str):
        logger.info(f"Embedding query using model: {model_name}")
        return self.embedding_models.get(model_name).embed_query(query)

    def embed_documents(self, documents: list[str], model_name: str):
        logger.info(f"Embedding documents using model: {model_name}")
        return self.embedding_models.get(model_name).embed_documents(documents)

embedding_service = EmbeddingService()
