import requests
import re
from pathlib import Path
from uuid import uuid4
from loguru import logger
from sqlalchemy.orm import Session

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
    def build_pdf_url(arxiv_id: str) -> str:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    @staticmethod
    def download_pdf(pdf_url: str) -> Path:
        PDFService.DATA_DIR.mkdir(parents=True, exist_ok=True)

        pdf_name = pdf_url.split("/")[-1]
        pdf_path = PDFService.DATA_DIR / pdf_name

        if pdf_path.exists():
            return pdf_path

        r = requests.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        r.raise_for_status()
        pdf_path.write_bytes(r.content)

        return pdf_path

    @staticmethod
    def load_pdf_docs(pdf_path: Path) -> list[Document]:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        current_section = "unknown"

        section_pattern = re.compile(
            r"^(\d+(\.\d+)*\s+)?(Abstract|Introduction|Related Work|Method|Conclusion|References)",
            re.IGNORECASE
        )

        for doc in docs:
            lines = doc.page_content.split("\n")
            for line in lines:
                if section_pattern.match(line.strip()):
                    current_section = line.strip()
                    break

            doc.metadata["section"] = current_section

        return docs

    @staticmethod
    def extract_figures_from_docs(docs: list[Document]) -> list[dict]:
        figures = []
        pattern = re.compile(r'^(Figure|Fig\.?)\s*\d+[:.]?\s*(.+)', re.IGNORECASE)

        for doc in docs:
            page = doc.metadata.get("page")
            section = doc.metadata.get("section", "unknown")

            for line in doc.page_content.split("\n"):
                line = line.strip()
                match = pattern.match(line)

                if not match or len(line) < 20:
                    continue

                figures.append({
                    "figure_id": match.group(0).split(":")[0],
                    "caption": line,
                    "section": section,
                    "page": page
                })

        return figures
    
    @staticmethod
    def extract_tables_from_docs(docs: list[Document]) -> list[dict]:
        tables = []
        pattern = re.compile(
            r'^(Table)\s*\d+[:.]?\s*(.+)',
            re.IGNORECASE
        )

        for doc in docs:
            page = doc.metadata.get("page")
            section = doc.metadata.get("section", "unknown")

            for line in doc.page_content.split("\n"):
                line = line.strip()
                match = pattern.match(line)

                if not match or len(line) < 15:
                    continue

                tables.append({
                    "table_id": match.group(0).split(":")[0],
                    "caption": line,
                    "section": section,
                    "page": page
                })

        return tables

    @staticmethod
    def _infer_figure_type(caption: str) -> str:
        caption = caption.lower()

        if any(k in caption for k in ["architecture", "framework", "pipeline", "overview"]):
            return "architecture"
        if any(k in caption for k in ["results", "accuracy", "performance", "comparison"]):
            return "results"
        if any(k in caption for k in ["ablation"]):
            return "ablation"
        if any(k in caption for k in ["example", "visualization"]):
            return "qualitative"

        return "other"

    @staticmethod
    def _heuristic_figure_description(caption: str) -> str:
        caption = caption.lower()

        if "architecture" in caption:
            return "This figure illustrates the overall model architecture."
        if "pipeline" in caption:
            return "This figure shows the processing pipeline."
        if "results" in caption or "comparison" in caption:
            return "This figure compares experimental results."
        if "ablation" in caption:
            return "This figure presents ablation study results."

        return "This figure provides visual support for the paper."

    @staticmethod
    def _infer_table_type(caption: str) -> str:
        caption = caption.lower()

        if any(k in caption for k in ["results", "performance", "accuracy"]):
            return "results"
        if any(k in caption for k in ["ablation"]):
            return "ablation"
        if any(k in caption for k in ["dataset", "statistics"]):
            return "dataset"
        if any(k in caption for k in ["hyperparameter", "settings"]):
            return "hyperparameter"

        return "other"

    @staticmethod
    def _heuristic_table_description(caption: str) -> str:
        caption = caption.lower()

        if "results" in caption or "performance" in caption:
            return "This table reports quantitative experimental results."
        if "ablation" in caption:
            return "This table presents ablation study results."
        if "dataset" in caption:
            return "This table summarizes dataset statistics."
        if "hyperparameter" in caption:
            return "This table lists hyperparameter settings."

        return "This table provides structured numerical information."

    @staticmethod
    def extract_equations_from_docs(
        docs: list[Document],
        context_window: int = 2,
    ) -> list[dict]:

        equations = []

        equation_pattern = re.compile(
            r"""
            (^Eq\.?\s*\(?\d+\)?[:.]?) |
            (^Equation\s*\(?\d+\)?[:.]?) |
            (^\(?\d+\)?\s*=\s*.+) |
            (\\begin\{equation\}) |
            (\$\$)
            """,
            re.VERBOSE | re.IGNORECASE
        )

        for doc in docs:
            page = doc.metadata.get("page")
            section = doc.metadata.get("section", "unknown")
            lines = doc.page_content.split("\n")

            for i, line in enumerate(lines):
                line_strip = line.strip()

                if not equation_pattern.search(line_strip):
                    continue

                prev_ctx = lines[max(0, i - context_window):i]
                next_ctx = lines[i + 1:i + 1 + context_window]

                equations.append({
                    "equation_raw": line_strip,
                    "prev_text": " ".join(prev_ctx).strip(),
                    "next_text": " ".join(next_ctx).strip(),
                    "section": section,
                    "page": page,
                })

        return equations

class IngestService:

    @staticmethod
    def chunk_text_docs(
        docs: list[Document],
        base_metadata: dict,
    ) -> list[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        chunks = splitter.split_documents(docs)

        for i, c in enumerate(chunks):
            c.metadata = {
                **base_metadata,
                **c.metadata,
                "type": "text",
                "chunk_index": i,
            }

        return chunks

    @staticmethod
    def chunk_figure(
        figures: list,
        paper_id: int,
        collection_id: int,
        pdf_url: str = "",
        generate_description: bool = True
    ) -> list[Document]:

        docs = []

        for fig in figures:
            caption = fig["caption"]
            description = ""
            if generate_description:
                description = PDFService._heuristic_figure_description(caption)

            content = "\n".join(
                p for p in [
                    fig["figure_id"],
                    caption,
                    f"Description: {description}" if description else None
                ]
                if p
            )

            metadata = {
                "type": "figure",
                "paper_id": paper_id,
                "collection_id": collection_id,
                "figure_id": fig["figure_id"],
                "section": fig.get("section") or "unknown",
                "page": fig.get("page"),
                "pdf_url": pdf_url,
                "figure_kind": PDFService._infer_figure_type(caption)
            }

            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    @staticmethod
    def chunk_table(
        tables: list,
        paper_id: int,
        collection_id: int,
        pdf_url: str = "",
        generate_description: bool = True
    ) -> list[Document]:

        docs = []

        for tbl in tables:
            caption = tbl["caption"]

            description = ""
            if generate_description:
                description = PDFService._heuristic_table_description(caption)

            content = "\n".join(
                p for p in [
                    tbl["table_id"],
                    caption,
                    f"Description: {description}" if description else None
                ]
                if p
            )

            metadata = {
                "type": "table",
                "paper_id": paper_id,
                "collection_id": collection_id,
                "table_id": tbl["table_id"],
                "section": tbl.get("section") or "unknown",
                "page": tbl.get("page"),
                "pdf_url": pdf_url,
                "table_kind": PDFService._infer_table_type(caption)
            }

            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    @staticmethod
    def chunk_equation(
        equations: list,
        paper_id: int,
        collection_id: int,
        pdf_url: str = "",
    ) -> list[Document]:

        docs = []

        for i, eq in enumerate(equations):

            explanation_parts = []
            if eq["prev_text"]:
                explanation_parts.append(eq["prev_text"])
            if eq["next_text"]:
                explanation_parts.append(eq["next_text"])

            explanation = (
                " ".join(explanation_parts)
                if explanation_parts
                else "This equation is defined in the paper."
            )

            content = "\n".join([
                f"Section: {eq['section']}",
                "",
                "Context:",
                explanation,
                "",
                "Equation:",
                eq["equation_raw"],
            ])

            metadata = {
                "type": "equation",
                "level": 2,
                "paper_id": paper_id,
                "collection_id": collection_id,
                "section": eq.get("section") or "unknown",
                "page": eq.get("page"),
                "equation_index": i,
                "pdf_url": pdf_url,
            }

            docs.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return docs
    @staticmethod
    def ingest_search_results(
        index_name: str,
        docs: list[Document],
        collection_id: int,
    ):
        store = pinecone_manager.get_store(index_name)
        db: Session = SessionLocal()

        try:
            for doc in docs:
                meta = doc.metadata
                arxiv_id = meta.get("arxiv_id")

                if not arxiv_id:
                    continue

                pdf_url = PDFService.build_pdf_url(arxiv_id)

                paper = (
                    db.query(Paper)
                    .filter(
                        Paper.arxiv_id == arxiv_id,
                        Paper.collection_id == collection_id,
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
                        pdf_url=pdf_url,
                    )
                    db.add(paper)
                    db.flush()

                pdf_path = PDFService.download_pdf(pdf_url)
                pdf_docs = PDFService.load_pdf_docs(pdf_path)

                text_chunks = IngestService.chunk_text_docs(
                    pdf_docs,
                    base_metadata={
                        "collection_id": collection_id,
                        "paper_id": paper.id,
                        "arxiv_id": arxiv_id,
                        "title": paper.title,
                        "source": meta.get("source"),
                        "query": meta.get("query"),
                        "pdf_url": pdf_url,
                    },
                )

                figures = PDFService.extract_figures_from_docs(pdf_docs)
                figure_chunks = IngestService.chunk_figure(
                    figures,
                    paper_id=paper.id,
                    collection_id=collection_id,
                    pdf_url=pdf_url,
                )

                tables = PDFService.extract_tables_from_docs(pdf_docs)
                table_chunks = IngestService.chunk_table(
                    tables,
                    paper_id=paper.id,
                    collection_id=collection_id,
                    pdf_url=pdf_url,
                )

                equations = PDFService.extract_equations_from_docs(pdf_docs)
                equation_chunks = IngestService.chunk_equation(
                    equations,
                    paper_id=paper.id,
                    collection_id=collection_id,
                    pdf_url=pdf_url,
                )


                all_docs = text_chunks + figure_chunks + table_chunks + equation_chunks


                ids = []
                for d in all_docs:
                    vector_id = str(uuid4())
                    d.metadata["vector_id"] = vector_id
                    ids.append(vector_id)   

                store.add_documents(all_docs, ids=ids)
                logger.info(f"text chunks: {len(text_chunks)}")
                logger.info(f"figure chunks: {len(figure_chunks)}")
                logger.info(f"table chunks: {len(table_chunks)}")


            db.commit()

        except Exception as e:
            db.rollback()
            logger.error(f"Ingest failed: {e}")
            raise e

        finally:
            db.close()
