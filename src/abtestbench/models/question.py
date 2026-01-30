"""Question model and schema."""

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


class QuestionTopic(str, Enum):
    """A/B testing question topics."""

    POWER_ANALYSIS = "power_analysis"
    SAMPLE_SIZE = "sample_size"
    SIGNIFICANCE_TESTING = "significance_testing"
    CONFIDENCE_INTERVALS = "confidence_intervals"
    EFFECT_SIZE = "effect_size"


class QuestionDifficulty(str, Enum):
    """Question difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class NumericAnswer(BaseModel):
    """Expected numeric answer with tolerance."""

    type: str = "numeric"
    value: float
    tolerance: float = 0.01
    tolerance_type: str = Field(default="relative", pattern="^(absolute|relative)$")


class CategoricalAnswer(BaseModel):
    """Expected categorical answer."""

    type: str = "categorical"
    value: str
    alternatives: list[str] = Field(default_factory=list)


class BooleanAnswer(BaseModel):
    """Expected boolean answer."""

    type: str = "boolean"
    value: bool


ExpectedAnswer = Union[NumericAnswer, CategoricalAnswer, BooleanAnswer]


class EvaluationConfig(BaseModel):
    """How to evaluate the response."""

    method: str = Field(pattern="^exact_match$")


class Question(BaseModel):
    """A single benchmark question."""

    id: str = Field(pattern="^[a-z0-9_]+$")
    topic: QuestionTopic
    difficulty: QuestionDifficulty
    question: str
    context: Optional[str] = None
    expected_answer: ExpectedAnswer
    evaluation: EvaluationConfig
    source: Optional[str] = None
    requires_code: bool = False


class QuestionSet(BaseModel):
    """A collection of questions."""

    questions: list[Question]

    def filter_by_topic(self, topics: list[str]) -> "QuestionSet":
        """Filter questions by topic."""
        return QuestionSet(
            questions=[q for q in self.questions if q.topic.value in topics]
        )

    def filter_by_difficulty(self, difficulties: list[str]) -> "QuestionSet":
        """Filter questions by difficulty."""
        return QuestionSet(
            questions=[q for q in self.questions if q.difficulty.value in difficulties]
        )

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)
