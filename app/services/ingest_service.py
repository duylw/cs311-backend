import requests
from pathlib import Path
from uuid import uuid4
from loguru import logger
from sqlalchemy.orm import Session
import re


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


from app.vector_store.pinecone_store_manager import pinecone_manager
from app.models.paper import Paper
from app.db.session import SessionLocal
from app.core.config import settings



class PDFService:
    DATA_DIR = Path("data/papers")

    @staticmethod
    def resolve_pdf_url(link: str) -> str | None:
        if "arxiv.org" not in link:
            return None

        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", link)
        if not m:
            return None

        arxiv_id = m.group(1)
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


    @staticmethod
    def download_pdf(pdf_url: str) -> Path:
        PDFService.DATA_DIR.mkdir(parents=True, exist_ok=True)

        pdf_name = pdf_url.split("/")[-1]
        pdf_path = PDFService.DATA_DIR / pdf_name

        if pdf_path.exists():
            return pdf_path

        logger.info(f"Downloading PDF: {pdf_url}")
        r = requests.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30
        )
        r.raise_for_status()
        pdf_path.write_bytes(r.content)

        return pdf_path

    @staticmethod
    def load_pdf_docs(pdf_path: Path) -> list[Document]:
        loader = PyPDFLoader(str(pdf_path))
        return loader.load()
  
    @staticmethod
    def extract_arxiv_id(link: str) -> str | None:
        m = re.search(r"(\d{4}\.\d{4,5})", link)
        return m.group(1) if m else None


class IngestService:

    @staticmethod
    def chunk_text_docs(
        docs: list[Document],
        base_metadata: dict
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        chunks = splitter.split_documents(docs)

        for i, c in enumerate(chunks):
            c.metadata = {
                **c.metadata,
                **base_metadata,
                "chunk_index": i,
            }

        return chunks

    @staticmethod
    def ingest_search_results(
        index_name: str,
        docs: list[Document],
        collection_id: int,
    ):
        """
        docs (from Google search):
        - metadata: title, link, source, axis, query, rerank_score
        """

        store = pinecone_manager.get_store(index_name)
        db: Session = SessionLocal()
        try:
            for doc in docs:
                meta = doc.metadata
                abs_link = meta.get("link")
                axriv_id = PDFService.extract_arxiv_id(abs_link)
                if not abs_link:
                    continue

                # 1️⃣ Resolve PDF
                pdf_url = PDFService.resolve_pdf_url(abs_link)
                if not pdf_url:
                    logger.warning(f"Skip non-arXiv link: {abs_link}")
                    continue

                # 2️⃣ DB: upsert paper
                paper = (
                    db.query(Paper)
                    .filter(
                        Paper.arxiv_id == axriv_id,
                        Paper.collection_id == collection_id
                    )
                    .first()
                )

                if not paper:
                    paper = Paper(
                        collection_id=collection_id,
                        arxiv_id=axriv_id,
                        title=meta.get("title"),
                        authors=meta.get("authors", "Unknown"),
                        abstract=meta.get("abstract"),
                        pdf_url=pdf_url,
                    )
                    db.add(paper)
                    db.flush()

                logger.info(f"Ingesting paper: {paper.title}")

                # 3️⃣ Download + parse PDF
                pdf_path = PDFService.download_pdf(pdf_url)
                pdf_docs = PDFService.load_pdf_docs(pdf_path)

                # 4️⃣ Chunk full paper
                chunks = IngestService.chunk_text_docs(
                    pdf_docs,
                    base_metadata={
                        "collection_id": collection_id,
                        "paper_id": paper.id,
                        "source": meta.get("source"),
                        "axis": meta.get("axis"),
                        "query": meta.get("query"),
                        "rerank_score": meta.get("rerank_score"),
                        "pdf_url": pdf_url,
                        "title": paper.title,
                    }
                )

                # 5️⃣ Ingest Pinecone (batch)
                ids = []
                for c in chunks:
                    vector_id = str(uuid4())
                    c.metadata["vector_id"] = vector_id
                    ids.append(vector_id)

                store.add_documents(chunks, ids=ids)

            db.commit()
            logger.success("Ingest completed successfully")

        except Exception as e:
            db.rollback()
            logger.exception("Ingest failed")
            raise e

        finally:
            db.close()
