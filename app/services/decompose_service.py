from typing import List, Dict, Optional
from pathlib import Path
import time

from loguru import logger
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.schemas.axis import (
    AxisModel,
    AxisResponse,
    EvaluationResponse,
)

from app.core.config import settings

load_dotenv()


class QueryGenerationService:
    MAX_AXES: int = settings.MAX_AXES
    MAX_QUERIES_PER_AXIS: int = settings.MAX_QUERIES_PER_AXIS

    def __init__(self):
        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        prompt_dir = Path("prompts")

        self.axis_parser = PydanticOutputParser(pydantic_object=AxisResponse)
        self.axis_prompt = PromptTemplate(
            template=(prompt_dir / "decompose_query.txt").read_text(encoding="utf-8"),
            input_variables=["topic"],
            partial_variables={
                "format_instructions": self.axis_parser.get_format_instructions()
            },
        )
        self.axis_chain = self.axis_prompt | self.llm | self.axis_parser

        self.evaluate_parser = PydanticOutputParser(
            pydantic_object=EvaluationResponse
        )
        self.evaluate_prompt = PromptTemplate(
            template=(prompt_dir / "evaluate_query.txt").read_text(encoding="utf-8"),
            input_variables=["query", "axis"],
            partial_variables={
                "format_instructions": self.evaluate_parser.get_format_instructions()
            },
        )
        self.evaluate_chain = self.evaluate_prompt | self.llm | self.evaluate_parser

        logger.info("QueryGenerationService initialized")

    def generate_queries(
        self,
        topic: str,
        min_score: int = 3,
        simple_query_threshold: int = 4,
    ) -> List[Dict]:

        start_time = time.time()

        logger.debug(f"Evaluating original topic: {topic}")

        topic_eval = self._evaluate_queries(
            raw_queries=[{"axis": "General", "query": topic}],
            min_score=simple_query_threshold,
        )

        if topic_eval:
            logger.info("Skip decomposition.")
            return topic_eval
        
        axes = self._generate_axes(topic)
        raw_queries = self._flatten_axes(axes)

        final_results = self._evaluate_queries(
            raw_queries=raw_queries,
            min_score=min_score,
        )

        logger.info(
            f"Generated {len(final_results)} queries "
            f"in {time.time() - start_time:.2f}s"
        )

        return final_results

    def _safe_invoke(self, chain, payload: dict, task: str):
        try:
            return chain.invoke(payload)
        except Exception as e:
            logger.error(f"{task} failed: {e}")
            return None

    def _generate_axes(self, topic: str) -> List[AxisModel]:
        logger.debug(f"Generating axes for topic: {topic}")

        result: Optional[AxisResponse] = self._safe_invoke(
            self.axis_chain,
            {"topic": topic},
            task="Axis generation",
        )

        if not result or not result.axes:
            logger.warning("No axes generated")
            return []

        return result.axes[: self.MAX_AXES]

    def _flatten_axes(self, axes: List[AxisModel]) -> List[Dict]:
        raw_queries: List[Dict] = []

        for axis in axes:
            queries = axis.queries[: self.MAX_QUERIES_PER_AXIS]
            for q in queries:
                raw_queries.append(
                    {
                        "axis": axis.axis,
                        "query": q,
                    }
                )

        logger.debug(f"Flattened into {len(raw_queries)} raw queries")
        return raw_queries

    def _evaluate_queries(
        self,
        raw_queries: List[Dict],
        min_score: int,
    ) -> List[Dict]:

        if not raw_queries:
            return []

        final_results: List[Dict] = []

        for rq in raw_queries:
            query_text = rq["query"]
            axis_name = rq["axis"]

            if not isinstance(query_text, str):
                logger.error(f"Invalid query type: {type(query_text)}")
                continue

            eval_result: Optional[EvaluationResponse] = self._safe_invoke(
                self.evaluate_chain,
                {"query": query_text, "axis": axis_name},
                task="Query evaluation",
            )

            if not eval_result or not eval_result.results:
                continue

            for r in eval_result.results:
                if r.keep and r.score >= min_score:
                    final_results.append(
                        {
                            "axis": r.axis or axis_name,  
                            "query": r.query,  
                            "score": r.score,
                            "keep": r.keep,
                        }
                    )

        logger.debug(f"Evaluation kept {len(final_results)} queries")
        return final_results


query_generation_service = QueryGenerationService()