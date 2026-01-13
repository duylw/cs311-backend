
from dataclasses import dataclass

from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from typing import Annotated, List, Dict, Literal, TypedDict, Union
from operator import add


from app.core.config import settings

from app.services import rag_service
from app.vector_store.pinecone_store_manager import pinecone_manager
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service
from pydantic import BaseModel, Field

class ComplexityEvaluation(BaseModel):
    """Evaluation of query complexity."""
    complexity: Literal["simple", "complex"] = Field(
        description="Whether the query is simple or complex"
    )
    reasoning: str = Field(
        description="Brief explanation of the complexity assessment"
    )

class DecomposedQueries(BaseModel):
    queries: List[str] = Field(
        description="List of sub-queries derived from the main query"
    )

class EnhancedQuery(BaseModel):
    query: str = Field(
        description="Optimized version of the original query"
    )

class ThreadState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    collection_id: int
    complexity_evaluation: ComplexityEvaluation
    sub_queries: DecomposedQueries
    enhanced_query: EnhancedQuery
    retrieved_documents: Union[List[Dict], str]

def evaluate(state: ThreadState) -> ThreadState:
    # Make LLM call to evaluate complexity
    llm_structured = llm_service.get_llm(settings.GOOGLE_LLM_MODEL)\
                                .with_structured_output(ComplexityEvaluation)
    
    prompt = prompt_service.get_prompt("retrieval_query_evaluate")
    prompt = prompt.format(query=state["user_query"])

    response = llm_structured.invoke(prompt)

    return {
        "complexity_evaluation": response
    }

def query_complexity(state: ThreadState) -> str:
    # For Routing
    return state["complexity_evaluation"].complexity

def enhance(state: ThreadState) -> ThreadState:
    # Make LLM call to optimize simple query
    llm_structured = llm_service.get_llm(settings.GOOGLE_LLM_MODEL)\
                                .with_structured_output(EnhancedQuery)
    
    prompt = prompt_service.get_prompt("retrieval_query_enhance")
    prompt = prompt.format(query=state["user_query"])

    response = llm_structured.invoke(prompt)

    return {
        "enhanced_query": response
    }

def decompose(state: ThreadState) -> ThreadState:
    # Make LLM call to decompose complex query
    llm_structured = llm_service.get_llm(settings.LLM_DEFAULT_MODEL)\
                                .with_structured_output(DecomposedQueries)
    
    prompt = prompt_service.get_prompt("retrieval_query_decompose")
    prompt = prompt.format(query=state["user_query"])

    response = llm_structured.invoke(prompt)
    
    return {
        "sub_queries": response
    }

def retrive_documents(state: ThreadState) -> ThreadState:
    # Retrieve documents from vector store
    vector_store = pinecone_manager.get_store(
            index_name=settings.PINECONE_INDEX_NAME,
        )

    retriever = vector_store.as_retriever(
                                        search_type="similarity",
                                        search_kwargs={
                                            "k": 15,
                                            "filter": {"collection_id": state["collection_id"]}
                                            }
                                        )
    res = {}

    query_complexity = state["complexity_evaluation"].complexity

    if query_complexity == "complex":
        for sub_query in state["sub_queries"].queries:
            res[sub_query] = retriever.invoke(sub_query)
    else:
        enhanced = state["enhanced_query"].query
        res[enhanced] = retriever.invoke(enhanced)
    
    return {
        "retrieved_documents": res
    }

def build_context(result:Union[List[Dict]], query_complexity="simple") -> str:
    # Build context string from retrieved documents for answer generation
    norm = ""
    
    keys = result.keys()

    if query_complexity == "simple":
        docs = result[keys[0]]
        for doc in docs:
            norm += f"From Paper: {doc.metadata.get('title', 'N/A').strip()}\n"
            norm += f"From Section: {doc.metadata.get('section', 'N/A')}:\n"
            norm += f"Content: {doc.page_content.strip()}\n"
    else:
        for key in keys:
            norm += f"Sub Query: {key}\n"
            docs = result[key]
            for doc in docs:
                norm += f"From Paper: {doc.metadata.get('title', 'N/A').strip()}\n"
                norm += f"From Section: {doc.metadata.get('section', 'N/A')}:\n"
                norm += f"Content: {doc.page_content.strip()}\n"
            norm += "\n"

    return norm   

def answer(state: ThreadState) -> ThreadState:
    # Generate final answer using RAG

    retrive_documents = rag_service.retrive_documents
    query_complexity = state["complexity_evaluation"].complexity

    context = build_context(result=retrive_documents,
                            query_complexity=query_complexity)

    prompt = prompt_service.get_prompt("retrieval_generate_answer").format(
            query=state["user_query"],
            context=context
        )

    response = llm_service.call_llm(settings.GOOGLE_LLM_MODEL, prompt)

    return {
        "messages": [response]
    }