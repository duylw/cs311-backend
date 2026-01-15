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
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.heuristic_config import FIGURE_CONFIG, TABLE_CONFIG   
SECTION_PATTERN = re.compile(
    r"^(\d+(\.\d+)*\s+)?(Abstract|Introduction|Related Work|Method|Methods|Approach|Experiments|Results|Discussion|Conclusion|References)",
    re.IGNORECASE
)

CAPTION_PATTERN = re.compile(
    r"^(Figure|Fig\.?|Table)\s*\d+",
    re.IGNORECASE
)

class CaptionHeuristicEngine:
    def __init__(self, config: dict):
        self.config = config

    def infer_type(self, caption: str) -> str:
        caption = caption.lower()

        for item_type, cfg in self.config.items():
            if any(k in caption for k in cfg["keywords"]):
                return item_type

        return "other"

    def describe(self, caption: str) -> str:
        item_type = self.infer_type(caption)
        return self.config[item_type]["description"]

FIGURE_ENGINE = CaptionHeuristicEngine(FIGURE_CONFIG)
TABLE_ENGINE = CaptionHeuristicEngine(TABLE_CONFIG)


def is_caption(paragraph: str) -> bool:
    first_line = paragraph.strip().split("\n")[0]
    return bool(CAPTION_PATTERN.match(first_line))


def split_into_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text)

    paragraphs = []
    for p in raw:
        p = p.strip()
        if len(p) < 50:
            continue
        if is_caption(p):
            continue
        paragraphs.append(p)

    return paragraphs

def chunk_paragraphs(
    paragraphs: list[str],
    max_chars: int,
) -> list[str]:

    chunks = []
    current = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(p)
        current_len += len(p)

    if current:
        chunks.append("\n\n".join(current))

    return chunks   

class PDFService:
    DATA_DIR = Path("data/papers")

    @staticmethod
    def build_pdf_url(arxiv_id: str) -> str:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    @staticmethod
    def download_pdf(pdf_url: str) -> Path:
        PDFService.DATA_DIR.mkdir(parents=True, exist_ok=True)

        pdf_path = PDFService.DATA_DIR / pdf_url.split("/")[-1]
        if pdf_path.exists():
            return pdf_path

        r = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        pdf_path.write_bytes(r.content)
        return pdf_path

    @staticmethod
    def load_pdf_docs(pdf_path: Path) -> list[Document]:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        current_section = "unknown"

        for doc in docs:
            for line in doc.page_content.split("\n"):
                if SECTION_PATTERN.match(line.strip()):
                    current_section = line.strip()
                    break
            doc.metadata["section"] = current_section

        return docs

    @staticmethod
    def group_docs_by_section(docs: list[Document]) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}

        for doc in docs:
            section = doc.metadata.get("section", "unknown")
            text = doc.page_content.strip()
            if not text:
                continue
            sections.setdefault(section, []).append(text)

        return sections


    @staticmethod
    def extract_figures_from_docs(docs: list[Document], context_window: int = 5) -> list[dict]:
        figures = []
        pattern = re.compile(r'^(Figure|Fig\.?)\s*((?:\d+|[IVXLCDM]+))[:.]?\s*(.*)', re.IGNORECASE)

        for doc in docs:
            page = doc.metadata.get("page")
            section = doc.metadata.get("section", "unknown")
            lines = doc.page_content.split("\n")

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                match = pattern.match(line)
                if match:
                    figure_id = f"Figure {match.group(2).upper()}"

                    caption = match.group(3).strip()

                    if not caption and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not SECTION_PATTERN.match(next_line):
                            caption = next_line
                            i += 1

                    caption_lines = [caption] if caption else []

                    i += 1
                    while (
                        i < len(lines)
                        and lines[i].strip()
                        and not pattern.match(lines[i].strip())
                        and not SECTION_PATTERN.match(lines[i].strip())
                    ):
                        caption_lines.append(lines[i].strip())
                        i += 1

                    caption = " ".join(caption_lines)

                    prev_ctx = [lines[j].strip() for j in range(max(0, i - context_window - len(caption_lines)), i - len(caption_lines)) if lines[j].strip()]
                    next_ctx = [lines[j].strip() for j in range(i, min(len(lines), i + context_window)) if lines[j].strip()]

                    figures.append({
                        "figure_id": figure_id,
                        "caption": caption,
                        "prev_text": " ".join(prev_ctx),
                        "next_text": " ".join(next_ctx),
                        "section": section,
                        "page": page
                    })
                else:
                    i += 1

        return figures
    
    @staticmethod
    def extract_tables_from_docs(docs: list[Document], context_window: int = 5) -> list[dict]:
        tables = []
        pattern = re.compile(r'^(Table)\s*((?:\d+|[IVXLCDM]+))[:.]?\s*(.+)', re.IGNORECASE)

        for doc in docs:
            page = doc.metadata.get("page")
            section = doc.metadata.get("section", "unknown")
            lines = doc.page_content.split("\n")

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                match = pattern.match(line)

                if match and len(line) >= 15:
                    raw_id = match.group(2)
                    table_id = f"Table {raw_id.upper()}"
                    caption_lines = [match.group(3).strip()]

                    i += 1
                    while i < len(lines) and not CAPTION_PATTERN.match(lines[i].strip()) and not SECTION_PATTERN.match(lines[i].strip()) and len(lines[i].strip()) > 0 and not re.match(r'^\d+(\.\d+)*\s', lines[i].strip()):
                        caption_lines.append(lines[i].strip())
                        i += 1

                    caption = " ".join(caption_lines)

                    table_lines = []
                    while i < len(lines) and not CAPTION_PATTERN.match(lines[i].strip()) and not SECTION_PATTERN.match(lines[i].strip()) and len(lines[i].strip()) > 0:
                        table_lines.append(lines[i].strip())
                        i += 1

                    table_content = ""
                    if table_lines:
                        def split_row(row): return re.split(r'\s{2,}|\t', row)
                        headers = split_row(table_lines[0])
                        table_content += "| " + " | ".join(headers) + " |\n"
                        table_content += "| " + "--- | " * len(headers) + "\n"
                        for row in table_lines[1:]:
                            cells = split_row(row)
                            cells += [""] * (len(headers) - len(cells))
                            table_content += "| " + " | ".join(cells[:len(headers)]) + " |\n"

                    next_ctx = [lines[j].strip() for j in range(i, min(len(lines), i + context_window)) if lines[j].strip()]
                    prev_ctx = [lines[j].strip() for j in range(max(0, i - context_window - len(caption_lines) - len(table_lines)), i - len(caption_lines) - len(table_lines)) if lines[j].strip()]

                    tables.append({
                        "table_id": table_id,
                        "caption": caption,
                        "content": table_content if table_content else "\n".join(table_lines),
                        "prev_text": " ".join(prev_ctx),
                        "next_text": " ".join(next_ctx),
                        "section": section,
                        "page": page
                    })
                else:
                    i += 1

        return tables

    @staticmethod
    def is_likely_equation_line(line: str) -> bool:
        if len(line) < 10:
            return False
        math_pattern = re.compile(r'[=+\-*/^()[\]{}∫∑∂∞√∏αβγδθλμσφψωΓΔΘΛΠΣΦΨΩ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]')
        math_count = len(math_pattern.findall(line))
        word_count = len(re.findall(r'\b\w+\b', line))
        if math_count > 5 and word_count < 5 and math_count / len(line) > 0.15:
            return True
        return False

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

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if equation_pattern.match(line) or PDFService.is_likely_equation_line(line):
                    equation_lines = [line]
                    orig_i = i
                    i += 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if not next_line:
                            break
                        if SECTION_PATTERN.match(next_line) or CAPTION_PATTERN.match(next_line) or equation_pattern.match(next_line):
                            break
                        if PDFService.is_likely_equation_line(next_line):
                            equation_lines.append(next_line)
                            i += 1
                        else:
                            # For non-highly math lines, stop unless it's part of labeled
                            if equation_pattern.match(line):  # if started with label, collect one more if short
                                if len(next_line) < 50:
                                    equation_lines.append(next_line)
                                    i += 1
                                else:
                                    break
                            else:
                                break
                    equation_raw = "\n".join(equation_lines)

                    prev_ctx = [lines[j].strip() for j in range(max(0, orig_i - context_window), orig_i) if lines[j].strip()]
                    next_ctx = [lines[j].strip() for j in range(i, min(len(lines), i + context_window)) if lines[j].strip()]

                    equations.append({
                        "equation_raw": equation_raw,
                        "prev_text": " ".join(prev_ctx),
                        "next_text": " ".join(next_ctx),
                        "section": section,
                        "page": page,
                    })
                else:
                    i += 1

        return equations

class IngestService:

    @staticmethod
    def chunk_text(
        docs: list[Document],
        base_metadata: dict,
    ) -> list[Document]:

        section_map = PDFService.group_docs_by_section(docs)
        docs = []

        chunk_index = 0

        for section, texts in section_map.items():
            full_text = "\n\n".join(texts)
            paragraphs = split_into_paragraphs(full_text)
            chunks = chunk_paragraphs(
                paragraphs,
                max_chars=settings.CHUNK_SIZE,
            )

            for c in chunks:
                docs.append(
                    Document(
                        page_content=c,
                        metadata={
                            **base_metadata,
                            "type": "text",
                            "section": section,
                        }
                    )
                )
                chunk_index += 1

        return docs

    @staticmethod
    def chunk_figure(
        figures: list,
        arxiv_id: int,
        generate_description: bool = True
    ) -> list[Document]:

        docs = []

        for fig in figures:
            caption = fig["caption"]
            fig_type = FIGURE_ENGINE.infer_type(caption)
            description = FIGURE_ENGINE.describe(caption) if generate_description else ""

            context_parts = []
            if fig.get("prev_text"):
                context_parts.append(f"Previous context: {fig['prev_text']}")
            if fig.get("next_text"):
                context_parts.append(f"Following context: {fig['next_text']}")

            content = "\n\n".join(
                p for p in [
                    f"Figure ID: {fig['figure_id']}",
                    f"Caption: {caption}",
                    f"Description: {description}" if description else None,
                    "\n".join(context_parts) if context_parts else None
                ]
                if p
            )

            metadata = {
                "type": "figure",
                "arxiv_id": arxiv_id,
                "section": fig.get("section") or "unknown",
                "page": fig.get("page"),
                "figure_kind": fig_type,
            }

            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    @staticmethod
    def chunk_table(
        tables: list,
        arxiv_id: int,
        generate_description: bool = True
    ) -> list[Document]:

        docs = []

        for tbl in tables:
            caption = tbl["caption"]
            table_type = TABLE_ENGINE.infer_type(caption)
            description = TABLE_ENGINE.describe(caption) if generate_description else ""

            context_parts = []
            if tbl.get("prev_text"):
                context_parts.append(f"Previous context: {tbl['prev_text']}")
            if tbl.get("next_text"):
                context_parts.append(f"Following context: {tbl['next_text']}")

            content = "\n\n".join(
                p for p in [
                    f"Table ID: {tbl['table_id']}",
                    f"Caption: {caption}",
                    f"Description: {description}" if description else None,
                    f"Table Content:\n{tbl['content']}" if tbl.get("content") else None,
                    "\n".join(context_parts) if context_parts else None
                ]
                if p
            )

            metadata = {
                "type": "table",
                "arxiv_id": arxiv_id,
                "section": tbl.get("section") or "unknown",
                "page": tbl.get("page"),
                "table_kind": table_type
            }

            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    @staticmethod
    def chunk_equation(
        equations: list,
        arxiv_id: int,
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
                "arxiv_id": arxiv_id,
                "section": eq.get("section") or "unknown",
                "page": eq.get("page"),
            }

            docs.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return docs

    @staticmethod
    def process_single_paper(
        doc: Document,
        collection_id: int,
    ):
        meta = doc.metadata
        arxiv_id = meta.get("arxiv_id")
        if not arxiv_id:
            return None

        pdf_url = PDFService.build_pdf_url(arxiv_id)
        pdf_path = PDFService.download_pdf(pdf_url)
        pdf_docs = PDFService.load_pdf_docs(pdf_path)

        base_metadata = {
            "arxiv_id": arxiv_id,
            "title": meta.get("title"),
            "query": meta.get("query", ""),
        }

        text_chunks = IngestService.chunk_text(
            pdf_docs,
            base_metadata,
        )

        figure_chunks = IngestService.chunk_figure(
            PDFService.extract_figures_from_docs(pdf_docs),
            arxiv_id=arxiv_id
        )

        table_chunks = IngestService.chunk_table(
            PDFService.extract_tables_from_docs(pdf_docs),
            arxiv_id=arxiv_id
        )

        equation_chunks = IngestService.chunk_equation(
            PDFService.extract_equations_from_docs(pdf_docs),
            arxiv_id=arxiv_id,
        )

        logger.info({
            "arxiv_id": arxiv_id,
            "text_chunks": len(text_chunks),
            "figure_chunks": len(figure_chunks),
            "table_chunks": len(table_chunks),
            "equation_chunks": len(equation_chunks),
        })

        return {
            "paper": {
                "collection_id": collection_id,
                "arxiv_id": arxiv_id,
                "title": meta.get("title"),
                "authors": ", ".join(meta.get("authors", [])),
                "abstract": doc.page_content,
                "pdf_url": pdf_url,
            },
            "documents": (
                text_chunks
                + figure_chunks
                + table_chunks
                + equation_chunks
            )
        }

    @staticmethod
    def persist_results(index_name: str, results: list[dict]):
        store = pinecone_manager.get_store(index_name)
        db: Session = SessionLocal()

        try:
            for r in results:
                paper_data = r["paper"]
                docs = r["documents"]

                paper = (
                    db.query(Paper)
                    .filter(
                        Paper.arxiv_id == paper_data["arxiv_id"],
                        Paper.collection_id == paper_data["collection_id"],
                    )
                    .first()
                )

                if not paper:
                    paper = Paper(**paper_data)
                    db.add(paper)
                    db.flush()

                ids = []
                for d in docs:
                    d.metadata["paper_id"] = paper.id
                    vid = str(uuid4())
                    d.metadata["vector_id"] = vid
                    ids.append(vid)

                store.add_documents(docs, ids=ids)

            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    @staticmethod
    def ingest_search_results(
        index_name: str,
        docs: list[Document],
        collection_id: int,
        max_workers: int = 4,
    ):
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    IngestService.process_single_paper,
                    doc,
                    collection_id
                )
                for doc in docs
            ]

            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)

        IngestService.persist_results(index_name, results)