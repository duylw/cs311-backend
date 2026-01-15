
from dataclasses import dataclass
import os

from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from typing import Annotated, List, Dict, Literal, TypedDict, Union
from pydantic import BaseModel, Field

from app.core.config import settings

from app.vector_store.pinecone_store_manager import pinecone_manager
from app.services.llm_service import llm_service
from app.services.prompt_service import prompt_service


from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

class ComplexityEvaluation(BaseModel):
    """Evaluation of query complexity."""
    complexity: Literal["vague", "simple", "complex"] = Field(
        description="Whether the query vague, simple or complex"
    )
    reasoning: str = Field(
        description="Brief explanation of the query complexity assessment"
    )

class DecomposedQueries(BaseModel):
    queries: List[str] = Field(
        description="List of sub-queries derived from the main complex query"
    )

class EnhancedQuery(BaseModel):
    query: str = Field(
        description="Optimized version of the original simple query"
    )

class ThreadState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    collection_id: int
    complexity_evaluation: ComplexityEvaluation
    sub_queries: DecomposedQueries
    enhanced_query: EnhancedQuery
    retrieved_documents: Union[List[Dict], str]
    top_k: int
    use_reranking: bool

def evaluate(state: ThreadState) -> ThreadState:
    # Make LLM call to evaluate complexity
    llm_structured = llm_service.get_llm(settings.GOOGLE_LLM_MODEL)\
                                .with_structured_output(ComplexityEvaluation)
    
    prompt = prompt_service.get_prompt("retrieval_query_evaluate")
    prompt = prompt.format(query=state["messages"][-1].content)

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
    prompt = prompt.format(query=state["messages"][-1].content, reasoning=state["complexity_evaluation"].reasoning)

    response = llm_structured.invoke(prompt)

    return {
        "enhanced_query": response
    }

def decompose(state: ThreadState) -> ThreadState:
    # Make LLM call to decompose complex query
    llm_structured = llm_service.get_llm(settings.GOOGLE_LLM_MODEL)\
                                .with_structured_output(DecomposedQueries)
    
    prompt = prompt_service.get_prompt("retrieval_query_decompose")
    prompt = prompt.format(query=state["messages"][-1].content, reasoning=state["complexity_evaluation"].reasoning)
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
                                            "k": state.get("top_k", 15),
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
        for key in keys:
            docs = result[key]
            for doc in docs:
                norm += f"[{doc.metadata.get('title', 'N/A').strip()}]\n"
                norm += f"Content: {doc.page_content.strip()}\n"
    else:
        for key in keys:
            norm += f"Sub Query: {key}\n"
            docs = result[key]
            for doc in docs:
                norm += f"[{doc.metadata.get('title', 'N/A').strip()}]\n"
                norm += f"Content: {doc.page_content.strip()}\n"
            norm += "\n"

    return norm   

def concat_trimmed_messages(messages: List[AnyMessage], max_length: int) -> str:
    # Concatenate and trim messages to fit within max_length
    combined = ""
    for msg in messages:
        if isinstance(msg.content, HumanMessage):
            role = "Human"
        elif isinstance(msg.content, AIMessage):
            role = "AI"
        else:
            role = "Unknown"
        combined +=f"Role {role}: {msg.content} \n"
        if len(combined) > max_length:
            break
    return combined[:-max_length]

def answer(state: ThreadState) -> ThreadState:
    # Generate final answer using RAG

    query_complexity = state["complexity_evaluation"].complexity

    trimmed_mess = concat_trimmed_messages(state["messages"][:-3], max_length=1000)

    if query_complexity == "vague":

        prompt = prompt_service.get_prompt("retrieval_generate_vague").format(
                query=state["messages"][-1].content,
                history=trimmed_mess,
                reasoning=state["complexity_evaluation"].reasoning
        )
        
        response = llm_service.call_llm(settings.GOOGLE_LLM_MODEL, prompt)
    else:

        retrive_documents = state["retrieved_documents"]

        context = build_context(result=retrive_documents,
                                query_complexity=query_complexity)

        prompt = prompt_service.get_prompt("retrieval_generate_answer").format(
                query=state["messages"][-1].content,
                context=context,
                history=trimmed_mess
            )

        response = llm_service.call_llm(settings.GOOGLE_LLM_MODEL, prompt)

    return {
        "messages": [response]
    }


workflow = StateGraph(state_schema=ThreadState)

workflow.add_node("evaluate", evaluate)
workflow.add_node("decompose", decompose)
workflow.add_node("enhance", enhance)
workflow.add_node("retrive_documents", retrive_documents)
workflow.add_node("answer", answer)

workflow.add_edge(START, "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    query_complexity,
    {
        "simple": "enhance",
        "complex": "decompose",
        "vague": "answer",
    })

workflow.add_edge("enhance", "retrive_documents")
workflow.add_edge("decompose", "retrive_documents")
workflow.add_edge("retrive_documents", "answer")
workflow.add_edge("answer", END)

# Use SQLite for persistent checkpointing
# Remove the sqlite:/// prefix to get the actual file path
db_path = settings.LANGGRAPH_CHECKPOINT_URL.replace("sqlite:///", "")

# Ensure directory exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)

chatbot = workflow.compile(checkpointer=checkpointer)