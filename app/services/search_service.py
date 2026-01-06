import os
from typing import List, Dict
from googleapiclient.discovery import build
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.core.config import settings
from loguru import logger
from dotenv import load_dotenv
load_dotenv()
class SearchService:
    def __init__(self):
        self.reranker = (
            CrossEncoder(settings.RERANKING_MODEL)
            if settings.USE_RERANKING
            else None
        )

        self.api_key = os.getenv("GOOGLE_GCP_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        if not self.api_key or not self.cse_id:
            raise ValueError("GOOGLE_GCP_API_KEY or GOOGLE_CSE_ID not set in environment")

        self.service = build("customsearch", "v1", developerKey=self.api_key)

    def search_google(self, query: str, max_results: int) -> List[Dict]:
        try:
            response = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=max_results
            ).execute()

            results = []
            for item in response.get("items", []):
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                })
            return results

        except Exception as e:
            logger.error(f"Google search failed for query '{query}': {e}")
            return []

    def build_documents(self, axis: str, query: str, results: List[Dict]) -> List[Document]:
        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r["snippet"],
                    metadata={
                        "title": r["title"],
                        "link": r["link"],
                        "axis": axis,
                        "query": query,
                        "source": "google",
                    }
                )
            )
        return docs

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:

        if not self.reranker:
            return docs[:top_k]

        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)

        for d, s in zip(docs, scores):
            d.metadata["rerank_score"] = float(s)

        docs.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        return docs[:top_k]


def search_service(queries: List[Dict], top_k_per_query: int = 5) -> List[Document]:

    service = SearchService()
    all_docs = []

    for q in queries:
        axis = q["axis"]
        query = q["query"]

        raw_results = service.search_google(query=query, max_results=top_k_per_query)
        docs = service.build_documents(axis=axis, query=query, results=raw_results)
        top_docs = service.rerank(query=query, docs=docs, top_k=top_k_per_query)
        all_docs.extend(top_docs)

    return all_docs
