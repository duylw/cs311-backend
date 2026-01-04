from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from app.core.config import settings

class LLMService:
    def __init__(self):
        self.llm_models = {
            settings.OLLAMA_LLM_MODEL: ChatOllama(model=settings.OLLAMA_LLM_MODEL,
                                 base_url=settings.OLLAMA_BASE_URL),
            settings.GOOGLE_LLM_MODEL: ChatGoogleGenerativeAI(model=settings.GOOGLE_LLM_MODEL),
        }

    def get_llm(self, model_name: str):
        return self.llm_models.get(model_name)

    def call_llm(self, model_name: str, prompt: str):

        logger.info(f"Calling LLM model: {model_name}")
        
        llm = self.get_llm(model_name)
        if llm:
            return llm.invoke(prompt)
        else:
            raise ValueError(f"LLM model '{model_name}' not found.")
    
    def call_llm_streaming(self, model_name: str, prompt: str):
        pass

llm_service = LLMService()