from loguru import logger
from sqlalchemy.orm import Session
from uuid import uuid4

from app.vector_store.pinecone_store_manager import pinecone_manager
from app.models.paper import Paper
from app.models.chunk import PaperChunk
from app.db.session import SessionLocal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings

import requests
from tempfile import NamedTemporaryFile
import os
from langchain_community.document_loaders import PyPDFLoader


class IngestService:

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

        for i, c in enumerate(chunks):
            c.metadata.update(base_metadata)
            c.metadata["section"] = f"page_{c.metadata.get('page', 'na')}"
            c.metadata["chunk_index"] = i

        return chunks

    @staticmethod
    def ingest_papers(
        index_name: str,
        papers: list[Document],
        collection_id: int,
    ):
        store = pinecone_manager.get_store(index_name)
        db: Session = SessionLocal()

        try:
            for doc in papers:
                meta = doc.metadata
                arxiv_id = meta["arxiv_id"]

                paper = (
                    db.query(Paper)
                    .filter(
                        Paper.arxiv_id == arxiv_id,
                        Paper.collection_id == collection_id
                    )
                    .first()
                )

                if not paper:
                    paper = Paper(
                        collection_id=collection_id,
                        arxiv_id=arxiv_id,
                        title=meta.get("title"),
                        authors=", ".join(meta.get("authors", [])),
                        abstract=doc.page_content,
                        pdf_url=meta.get("pdf_url"),
                    )
                    db.add(paper)
                    db.flush()  

                logger.info(f"Fetching PDF {arxiv_id}")
                pdf_docs = IngestService.load_pdf_from_url(meta["pdf_url"])

                chunks = IngestService.chunk_pdf_docs(
                    pdf_docs,
                    base_metadata={
                        "collection_id": collection_id,
                        "arxiv_id": arxiv_id,
                        "paper_id": paper.id,
                    },
                )

                for c in chunks:
                    vector_id = str(uuid4())
                    c.metadata["vector_id"] = vector_id

                    store.add_documents(
                        [c],
                        ids=[vector_id]
                    )

                    chunk = PaperChunk(
                        paper_id=paper.id,
                        chunk_index=c.metadata["chunk_index"],
                        section=c.metadata["section"],
                        content=c.page_content,
                        vector_id=vector_id,
                        metadata=c.metadata,
                    )
                    db.add(chunk)

            db.commit()
            logger.success("Ingest completed: DB + Pinecone")

        except Exception as e:
            db.rollback()
            logger.exception("Ingest failed")
            raise e

        finally:
            db.close()
