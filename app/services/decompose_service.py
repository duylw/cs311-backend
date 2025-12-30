from typing import List
from pathlib import Path
from loguru import logger
import time
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import settings

load_dotenv()


class QueryGenerationService:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )

        self.parser = JsonOutputParser()

        prompt_dir = Path("prompts")

        self.axis_prompt = PromptTemplate.from_template(
            (prompt_dir / "decompose.txt").read_text(encoding="utf-8")
        )

        self.evaluate_prompt = PromptTemplate.from_template(
            (prompt_dir / "evaluate_query.txt").read_text(encoding="utf-8")
        )

        self.axis_chain = self.axis_prompt | self.llm | self.parser
        self.evaluate_chain = self.evaluate_prompt | self.llm | self.parser

        logger.info("QueryGenerationService initialized")

    def generate_queries(
        self,
        topic: str,
        min_score: int = 3
    ) -> List[dict]:

        start_time = time.time()

        axes_result = self._generate_axes(topic)

        raw_queries = []
        for axis in axes_result["axes"]:
            axis_name = axis["axis"]
            for q in axis.get("example_queries", []):
                raw_queries.append({
                    "axis": axis_name,
                    "query": q
                })

        final_queries = self._evaluate_queries(
            raw_queries=raw_queries,
            min_score=min_score
        )

        logger.info(
            f"Generated {len(final_queries)} queries "
            f"in {time.time() - start_time:.2f}s"
        )

        return final_queries


    def _generate_axes(self, topic: str) -> dict:
        logger.debug(f"Generating research axes for topic: {topic}")
        return self.axis_chain.invoke({"topic": topic})

    def _evaluate_queries(
        self,
        raw_queries: List[dict],
        min_score: int
    ) -> List[dict]:

        logger.debug("Evaluating generated queries")

        result = self.evaluate_chain.invoke({
            "queries": raw_queries
        })

        return [
            r
            for r in result["results"]
            if r["keep"] and r["score"] >= min_score
        ]

query_generation_service = QueryGenerationService()
