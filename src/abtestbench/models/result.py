"""Evaluation result models."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class NumericEvaluation(BaseModel):
    """Result of numeric answer evaluation."""

    correct: bool
    extracted_value: Optional[float] = None
    expected_value: float
    difference: Optional[float] = None
    within_tolerance: bool


class JudgeEvaluation(BaseModel):
    """Result of LLM-as-judge evaluation."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_concepts_found: list[str] = Field(default_factory=list)
    key_concepts_missing: list[str] = Field(default_factory=list)


class CompositeEvaluation(BaseModel):
    """Combined evaluation result."""

    overall_score: float = Field(ge=0.0, le=1.0)
    numeric_score: float = Field(ge=0.0, le=1.0)
    explanation_score: float = Field(ge=0.0, le=1.0)
    numeric_evaluation: Optional[NumericEvaluation] = None
    judge_evaluation: Optional[JudgeEvaluation] = None


class QuestionResult(BaseModel):
    """Result for a single question."""

    question_id: str
    category: str
    difficulty: str
    success: bool
    response: Optional[Any] = None  # LLMResponse
    evaluation: Optional[CompositeEvaluation] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class ProviderSummary(BaseModel):
    """Summary statistics for a provider/model."""

    provider: str
    model: str
    total_questions: int
    successful: int
    failed: int
    overall_accuracy: float
    numeric_accuracy: float
    explanation_quality: float
    by_category: dict[str, float] = Field(default_factory=dict)
    by_difficulty: dict[str, float] = Field(default_factory=dict)
    total_tokens: int = 0
    total_time_seconds: float = 0.0


class BenchmarkResult(BaseModel):
    """Complete benchmark result."""

    timestamp: datetime
    results: dict[str, list[QuestionResult]]  # model -> results

    def get_summary(self, model: str) -> ProviderSummary:
        """Generate summary statistics for a model."""
        results = self.results.get(model, [])
        successful = [r for r in results if r.success and r.evaluation]

        # Calculate scores
        overall_scores = [r.evaluation.overall_score for r in successful if r.evaluation]
        numeric_scores = [r.evaluation.numeric_score for r in successful if r.evaluation]
        explanation_scores = [r.evaluation.explanation_score for r in successful if r.evaluation]

        # By category
        by_category: dict[str, float] = {}
        categories = set(r.category for r in successful)
        for cat in categories:
            cat_results = [r for r in successful if r.category == cat and r.evaluation]
            if cat_results:
                by_category[cat] = sum(r.evaluation.overall_score for r in cat_results) / len(
                    cat_results
                )

        # By difficulty
        by_difficulty: dict[str, float] = {}
        difficulties = set(r.difficulty for r in successful)
        for diff in difficulties:
            diff_results = [r for r in successful if r.difficulty == diff and r.evaluation]
            if diff_results:
                by_difficulty[diff] = sum(r.evaluation.overall_score for r in diff_results) / len(
                    diff_results
                )

        # Parse provider/model from key
        parts = model.split("/", 1)
        provider = parts[0] if len(parts) > 1 else "unknown"
        model_name = parts[1] if len(parts) > 1 else model

        return ProviderSummary(
            provider=provider,
            model=model_name,
            total_questions=len(results),
            successful=len(successful),
            failed=len(results) - len(successful),
            overall_accuracy=sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            numeric_accuracy=sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0,
            explanation_quality=(
                sum(explanation_scores) / len(explanation_scores) if explanation_scores else 0.0
            ),
            by_category=by_category,
            by_difficulty=by_difficulty,
            total_time_seconds=sum(r.elapsed_seconds for r in results),
        )
