"""Composite hybrid evaluator."""

from typing import Optional

from ..models.question import (
    BooleanAnswer,
    CategoricalAnswer,
    NumericAnswer,
    NumericRangeAnswer,
    Question,
)
from ..models.response import LLMResponse
from ..models.result import CompositeEvaluation, JudgeEvaluation, NumericEvaluation
from .llm_judge import LLMJudgeEvaluator
from .numeric import NumericEvaluator


class CompositeEvaluator:
    """Hybrid evaluation combining numeric and LLM-as-judge."""

    def __init__(
        self,
        provider: "OpenRouterProvider",
        judge_model: str,
    ):
        self.numeric_evaluator = NumericEvaluator()
        self.judge_evaluator = LLMJudgeEvaluator(provider, judge_model)

    async def evaluate(
        self,
        question: Question,
        response: LLMResponse,
    ) -> CompositeEvaluation:
        """Perform hybrid evaluation."""
        method = question.evaluation.method

        if method == "exact_match":
            return await self._evaluate_exact_match(question, response)
        elif method == "llm_judge":
            return await self._evaluate_llm_judge(question, response)
        else:  # hybrid
            return await self._evaluate_hybrid(question, response)

    async def _evaluate_exact_match(
        self,
        question: Question,
        response: LLMResponse,
    ) -> CompositeEvaluation:
        """Exact match evaluation."""
        expected = question.expected_answer

        if isinstance(expected, (NumericAnswer, NumericRangeAnswer)):
            numeric_eval = self.numeric_evaluator.evaluate(response, expected)
            score = 1.0 if numeric_eval.correct else 0.0
            return CompositeEvaluation(
                overall_score=score,
                numeric_score=score,
                explanation_score=0.0,
                numeric_evaluation=numeric_eval,
                judge_evaluation=None,
            )

        elif isinstance(expected, CategoricalAnswer):
            # Simple string matching for categorical
            content_lower = response.content.lower()
            correct = expected.value.lower() in content_lower or any(
                alt.lower() in content_lower for alt in expected.alternatives
            )
            return CompositeEvaluation(
                overall_score=1.0 if correct else 0.0,
                numeric_score=1.0 if correct else 0.0,
                explanation_score=0.0,
                numeric_evaluation=None,
                judge_evaluation=None,
            )

        elif isinstance(expected, BooleanAnswer):
            # Boolean matching
            content_lower = response.content.lower()
            if expected.value:
                correct = "yes" in content_lower or "true" in content_lower
            else:
                correct = "no" in content_lower or "false" in content_lower
            return CompositeEvaluation(
                overall_score=1.0 if correct else 0.0,
                numeric_score=1.0 if correct else 0.0,
                explanation_score=0.0,
                numeric_evaluation=None,
                judge_evaluation=None,
            )

        return CompositeEvaluation(
            overall_score=0.0,
            numeric_score=0.0,
            explanation_score=0.0,
            numeric_evaluation=None,
            judge_evaluation=None,
        )

    async def _evaluate_llm_judge(
        self,
        question: Question,
        response: LLMResponse,
    ) -> CompositeEvaluation:
        """LLM-as-judge only evaluation."""
        judge_eval = await self.judge_evaluator.evaluate(question, response)
        return CompositeEvaluation(
            overall_score=judge_eval.score,
            numeric_score=0.0,
            explanation_score=judge_eval.score,
            numeric_evaluation=None,
            judge_evaluation=judge_eval,
        )

    async def _evaluate_hybrid(
        self,
        question: Question,
        response: LLMResponse,
    ) -> CompositeEvaluation:
        """Hybrid numeric + LLM evaluation."""
        expected = question.expected_answer

        # Get numeric evaluation
        numeric_eval: Optional[NumericEvaluation] = None
        numeric_score = 0.0

        if isinstance(expected, (NumericAnswer, NumericRangeAnswer)):
            numeric_eval = self.numeric_evaluator.evaluate(response, expected)
            numeric_score = 1.0 if numeric_eval.correct else 0.0

        # Get judge evaluation
        judge_eval = await self.judge_evaluator.evaluate(question, response)

        # Compute weighted score
        numeric_weight = question.evaluation.numeric_weight
        explanation_weight = question.evaluation.explanation_weight

        overall_score = (
            numeric_weight * numeric_score + explanation_weight * judge_eval.score
        )

        return CompositeEvaluation(
            overall_score=overall_score,
            numeric_score=numeric_score,
            explanation_score=judge_eval.score,
            numeric_evaluation=numeric_eval,
            judge_evaluation=judge_eval,
        )
