import arxiv
from typing import List, Dict
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from app.core.config import settings


class SearchService:
    def __init__(self):
        self.reranker = (
            CrossEncoder(settings.RERANKING_MODEL)
            if settings.USE_RERANKING
            else None
        )

    @staticmethod
    def search_arxiv(query: str, max_results: int):
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return list(search.results())

    def build_documents(
        self,
        axis: str,
        query: str,
        results
    ) -> List[Document]:

        docs = []

        for r in results:
            arxiv_id = r.entry_id.split("/")[-1].split("v")[0]

            docs.append(
                Document(
                    page_content=r.summary,
                    metadata={
                        "arxiv_id": arxiv_id,
                        "title": r.title,
                        "authors": [a.name for a in r.authors],
                        "axis": axis,
                        "query": query,
                        "source": "arxiv",
                        "level": "abstract",
                        "pdf_url": r.pdf_url
                    }
                )
            )

        return docs

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int
    ) -> List[Document]:

        if not self.reranker:
            return docs[:top_k]

        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)

        for d, s in zip(docs, scores):
            d.metadata["rerank_score"] = float(s)

        docs.sort(
            key=lambda x: x.metadata["rerank_score"],
            reverse=True
        )

        return docs[:top_k]


def search_papers(queries: List[Dict]) -> List[Document]:
    service = SearchService()
    all_docs = []

    max_results_per_query = max(
        1, int(settings.ARXIV_MAX_RESULTS / len(queries))
    )

    for q in queries:
        axis = q["axis"]
        query = q["query"]

        raw_results = service.search_arxiv(
            query=query,
            max_results=max_results_per_query
        )

        docs = service.build_documents(
            axis=axis,
            query=query,
            results=raw_results
        )

        top_docs = service.rerank(
            query=query,
            docs=docs,
            top_k=settings.RETRIEVAL_TOP_K
        )

        all_docs.extend(top_docs)

    return all_docs
