"""Composite evaluator with exact match only."""

from ..models.question import (
    BooleanAnswer,
    CategoricalAnswer,
    NumericAnswer,
    Question,
)
from ..models.response import LLMResponse
from ..models.result import CompositeEvaluation, NumericEvaluation
from .numeric import NumericEvaluator


class CompositeEvaluator:
    """Exact match evaluation for benchmark responses."""

    def __init__(self):
        self.numeric_evaluator = NumericEvaluator()

    async def evaluate(
        self,
        question: Question,
        response: LLMResponse,
    ) -> CompositeEvaluation:
        """Perform exact match evaluation."""
        return await self._evaluate_exact_match(question, response)

    async def _evaluate_exact_match(
        self,
        question: Question,
        response: LLMResponse,
    ) -> CompositeEvaluation:
        """Exact match evaluation."""
        expected = question.expected_answer

        if isinstance(expected, NumericAnswer):
            numeric_eval = self.numeric_evaluator.evaluate(response, expected)
            score = 1.0 if numeric_eval.correct else 0.0
            return CompositeEvaluation(
                overall_score=score,
                numeric_score=score,
                numeric_evaluation=numeric_eval,
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
                numeric_evaluation=None,
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
                numeric_evaluation=None,
            )

        return CompositeEvaluation(
            overall_score=0.0,
            numeric_score=0.0,
            numeric_evaluation=None,
        )
