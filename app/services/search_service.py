import os
import re
from typing import List, Dict, Optional, Set

import arxiv
from googleapiclient.discovery import build
from langchain_core.documents import Document
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

            logger.info(f"[Google] Query='{query}' → {len(arxiv_ids)} unique arXiv IDs")
            return list(arxiv_ids)

        except Exception as e:
            logger.error(f"[Google] Search failed: {e}")
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

            logger.info(f"[arXiv] Fetched {len(docs)} papers")
            return docs

        except Exception as e:
            logger.error(f"[arXiv] Fetch failed: {e}")
            return []


def search_service(
    queries: List[Dict],
    sort_by_date: bool = True,
) -> List[Document]:
    service = SearchService()
    all_docs: List[Document] = []
    seen_arxiv_ids: Set[str] = set()
    general_k = settings.TOP_K_PAPERS_PER_QUERY
    axis_k = settings.TOP_K_PAPERS_PER_AXIS
    
    for q in queries:
        axis = q["axis"]
        query = q["query"]

        if axis == "General":
            top_k = general_k
        else:
            top_k = axis_k

        logger.info(f"=== Searching axis='{axis}' | query='{query}' | k={top_k} ===")

        arxiv_ids = service.search_google_arxiv_ids(
            query=query,
            max_results=top_k * 2,  
        )

        docs = service.fetch_arxiv_papers(arxiv_ids)

        for d in docs:
            d.metadata["axis"] = axis
            d.metadata["query"] = query

        if sort_by_date:
            docs.sort(
                key=lambda x: x.metadata.get("published", ""),
                reverse=True,
            )
        selected = []
        for d in docs:
            aid = d.metadata.get("arxiv_id")
            if aid not in seen_arxiv_ids:
                seen_arxiv_ids.add(aid)
                selected.append(d)
            if len(selected) >= top_k:
                break

        all_docs.extend(selected)

        logger.info(f"[Result] axis='{axis}' → selected {len(selected)} papers")

    logger.info(f"TOTAL collected papers: {len(all_docs)}")
    return all_docs
