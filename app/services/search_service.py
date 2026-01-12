import os
import re
from typing import List, Dict, Optional, Set

import arxiv
from googleapiclient.discovery import build
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from loguru import logger
from dotenv import load_dotenv
from app.core.config import settings

load_dotenv()

ARXIV_ID_REGEX = re.compile(
    r"arxiv\.org/(abs|pdf)/(?P<id>\d{4}\.\d{4,5})(v\d+)?"
)

def extract_arxiv_id(url: str) -> Optional[str]:
    match = ARXIV_ID_REGEX.search(url)
    if match:
        return match.group("id")
    return None


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
            raise ValueError("GOOGLE_GCP_API_KEY or GOOGLE_CSE_ID not set")

        self.google_service = build(
            "customsearch",
            "v1",
            developerKey=self.api_key,
        )

    def search_google_arxiv_ids(
        self,
        query: str,
        max_results: int,
    ) -> List[str]:
        try:
            response = self.google_service.cse().list(
                q=f"site:arxiv.org {query}",
                cx=self.cse_id,
                num=max_results,
            ).execute()

            arxiv_ids: Set[str] = set()

            for item in response.get("items", []):
                link = item.get("link", "")
                arxiv_id = extract_arxiv_id(link)
                if arxiv_id:
                    arxiv_ids.add(arxiv_id)

            return list(arxiv_ids)

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []

    def fetch_arxiv_papers(
        self,
        arxiv_ids: List[str],
    ) -> List[Document]:
        if not arxiv_ids:
            return []

        docs: List[Document] = []

        try:
            search = arxiv.Search(
                id_list=arxiv_ids,
                max_results=len(arxiv_ids),
            )

            for r in search.results():
                docs.append(
                    Document(
                        page_content=r.summary,
                        metadata={
                            "arxiv_id": r.entry_id.split("/")[-1],
                            "title": r.title,
                            "authors": [a.name for a in r.authors],
                            "categories": r.categories,
                            "published": r.published.isoformat(),
                            "updated": r.updated.isoformat(),
                            "source": "arxiv",
                        },
                    )
                )

            return docs

        except Exception as e:
            logger.error(f"arXiv fetch failed: {e}")
            return []

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not docs:
            return []

        if not self.reranker:
            return docs[:top_k]

        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)

        for d, s in zip(docs, scores):
            d.metadata["rerank_score"] = float(s)

        docs.sort(
            key=lambda x: x.metadata.get("rerank_score", 0.0),
            reverse=True,
        )

        return docs[:top_k]


def search_service(
    queries: List[Dict],
    top_k_per_query: int = 5,
) -> List[Document]:
    service = SearchService()
    all_docs: List[Document] = []

    for q in queries:
        axis = q["axis"]
        query = q["query"]

        arxiv_ids = service.search_google_arxiv_ids(
            query=query,
            max_results=top_k_per_query * 2,
        )

        docs = service.fetch_arxiv_papers(arxiv_ids)

        for d in docs:
            d.metadata["axis"] = axis
            d.metadata["query"] = query

        top_docs = service.rerank(
            query=query,
            docs=docs,
            top_k=top_k_per_query,
        )

        all_docs.extend(top_docs)

    return all_docs
