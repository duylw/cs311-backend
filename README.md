# Research Paper RAG System

**Course:** CS331 - Kĩ thuật lập trình trí tuệ nhân tạo  
**Project:** Intelligent Discovery and Semantic Q&A ASSISTANCE for Academic Research  
**Institution:** UIT (University of Information Technology)

## 👥 Team Members

| Student ID | Full Name |
|------------|-----------|
| 23520368 | Lương Quang Duy |  
| 23521408 | Hồ Phương Tây |
| 20520888 | Lê Trung Nhân |

## 📋 Project Overview

This project implements a comprehensive **Retrieval-Augmented Generation (RAG)** system designed to help researchers efficiently query and analyze academic papers. The system combines advanced NLP techniques, vector databases, and large language models to provide intelligent answers based on research paper collections.

### Key Features

- 📚 **Document Management**: Upload and manage research paper collections
- 🔍 **Intelligent Search**: Multi-axis semantic search with query decomposition
- 🤖 **AI-Powered Q&A**: Context-aware answer generation using RAG
- 📊 **Evaluation Framework**: Comprehensive retrieval quality assessment using RAGAS metrics
- 🗄️ **Vector Storage**: Pinecone integration for efficient similarity search
- 🔄 **Graph-based Retrieval**: LangGraph implementation for complex query workflows
- 💾 **Persistent Storage**: SQLite database for metadata and chat history

## 🏗️ Architecture

### Technology Stack

**Backend Framework:**
- FastAPI - High-performance REST API
- SQLAlchemy - Database ORM
- Pydantic - Data validation

**AI/ML Components:**
- LangChain - LLM orchestration framework
- LangGraph - Stateful agent workflows
- RAGAS - Retrieval evaluation metrics
- Google Gemini - LLM and embeddings
- Ollama - Local LLM support

**Vector Database:**
- Pinecone - Cloud vector storage

**Document Processing:**
- Docling - PDF parsing
- Unstructured - Multi-format document parsing
- PyMuPDF, PDFPlumber - PDF extraction

## 🚀 Setup & Installation

### Prerequisites

- Python 3.13+
- uv (Python package manager)
- Pinecone account
- Google Cloud API access (for Gemini)

### Installation Steps

1. **Clone the repository and navigate to backend**
```bash
cd backend
```

2. **Install dependencies using uv**
```bash
uv sync
```

3. **Configure environment variables**

Create a `.env` file in the backend directory and setup as `.env.example`

4. **Run the application**
```bash
uv run python -m app.main
```

The API will be available at `http://localhost:8000`  
API documentation: `http://localhost:8000/api/v1/docs`

## 💡 Usage

### 1. Create a Collection
```bash
POST /api/v1/collections
{
  "name": "Machine Learning Papers",
  "description": "Collection of ML research papers"
}
```

### 2. Upload Papers
```bash
POST /api/v1/papers/upload
{
  "collection_id": 1,
  "file": <PDF file>
}
```

### 3. Query the System
```bash
POST /api/v1/query/ask
{
  "query": "What are the latest techniques in transfer learning?",
  "collection_id": 1,
  "top_k": 5,
  "use_reranking": true
}
```

## 📊 Evaluation

The system includes comprehensive evaluation capabilities using RAGAS metrics:

- **Faithfulness**: Measures factual consistency with source documents
- **Answer Relevancy**: Assesses relevance of generated answers
- **Context Relevance**: Evaluates quality of retrieved context
- **Context Utilization**: Measures effective use of provided context

Run evaluation:
```bash
uv run python app/evaluation/retrieval_eval_ragas.py
```

Results are saved in `app/evaluation/results/`

## 🔬 Key Components

### RAG Pipeline
1. **Query Processing**: Decomposition and enhancement using LLM
2. **Retrieval**: Vector similarity search in Pinecone
3. **Reranking**: Optional reranking for better precision
4. **Generation**: Context-aware answer generation
5. **Logging**: Conversation history tracking

### Graph-based Retrieval
Implemented using LangGraph for:
- Multi-step reasoning
- Query decomposition
- Iterative refinement
- State management

## 📝 API Documentation

Full API documentation is available at `/api/v1/docs` when running the server.

### Main Endpoints

- `POST /api/v1/collections` - Create collection
- `GET /api/v1/collections` - List collections
- `POST /api/v1/papers/upload` - Upload paper
- `GET /api/v1/papers` - List papers
- `POST /api/v1/query/ask` - Ask question
- `POST /api/v1/query/ask/stream` - Streaming response

## 🧪 Testing

```bash
uv run pytest tests/
```

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Pinecone Documentation](https://docs.pinecone.io/)

## 📄 License

This project is developed for educational purposes as part of CS331 course.

---

**Last Updated:** January 2026