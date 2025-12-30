from loguru import logger
from app.vector_store.pinecone_store_manager import pinecone_manager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings

import requests
from tempfile import NamedTemporaryFile
import os
from langchain_community.document_loaders import PyPDFLoader


class IngestService:
    @staticmethod
    @staticmethod
    def load_pdf_from_url(pdf_url: str) -> list[Document]:
        r = requests.get(pdf_url, timeout=30)
        r.raise_for_status()

        tmp = NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(r.content)
            tmp.flush()
            tmp.close()  
            loader = PyPDFLoader(tmp.name)
            return loader.load()

        finally:
            os.unlink(tmp.name)  

    @staticmethod
    def chunk_pdf_docs(
        docs: list[Document],
        base_metadata: dict
    ) -> list[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        chunks = splitter.split_documents(docs)

        for c in chunks:
            c.metadata.update(base_metadata)
            c.metadata["section"] = f"page_{c.metadata.get('page', 'na')}"

        return chunks

    @staticmethod
    def ingest_papers(
        index_name: str,
        papers: list[Document],   
        collection_id: int,
    ):
        store = pinecone_manager.get_store(index_name)

        all_chunks: list[Document] = []

        for doc in papers:
            meta = doc.metadata

            arxiv_id = meta["arxiv_id"]
            pdf_url = meta["pdf_url"]

            logger.info(f"Fetching PDF {arxiv_id}")

            pdf_docs = IngestService.load_pdf_from_url(pdf_url)

            base_metadata = {
                "collection_id": collection_id,
                "arxiv_id": arxiv_id,
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "source": meta.get("source", "arxiv"),
            }

            chunks = IngestService.chunk_pdf_docs(
                pdf_docs,
                base_metadata
            )

            all_chunks.extend(chunks)


        if not all_chunks:
            logger.warning("No chunks to ingest")
            return

        logger.info(f"Total chunks to ingest: {len(all_chunks)}")

        store.add_documents(all_chunks)

        logger.success(
            f"Ingested {len(all_chunks)} chunks into Pinecone index '{index_name}'"
        )
