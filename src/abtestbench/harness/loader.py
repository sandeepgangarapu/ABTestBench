"""Question loader from TOML files."""

import tomllib
from pathlib import Path
from typing import Iterator, Optional

from ..models.question import Question, QuestionSet


class QuestionLoader:
    """Load and validate benchmark questions from individual TOML files."""

    def __init__(self, questions_dir: Path):
        self.questions_dir = Path(questions_dir)

    def load_all(
        self,
        topics: Optional[list[str]] = None,
        difficulties: Optional[list[str]] = None,
    ) -> QuestionSet:
        """Load all questions with optional filtering."""
        questions = []

        for file_path in self.questions_dir.glob("*.toml"):
            question = self._load_file(file_path)
            if question:
                questions.append(question)

        # Apply filters
        if topics:
            questions = [q for q in questions if q.topic.value in topics]
        if difficulties:
            questions = [q for q in questions if q.difficulty.value in difficulties]

        return QuestionSet(questions=questions)

    def _load_file(self, path: Path) -> Optional[Question]:
        """Load a single question from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        if not data:
            return None

        try:
            return Question(**data)
        except Exception as e:
            print(f"Warning: Failed to parse question from {path.name}: {e}")
            return None

    def iter_questions(self, **filters) -> Iterator[Question]:
        """Iterate over questions with optional filtering."""
        question_set = self.load_all(**filters)
        yield from question_set.questions

    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        for question in self.iter_questions():
            if question.id == question_id:
                return question
        return None

    def get_topics(self) -> list[str]:
        """Get all unique topics."""
        questions = self.load_all()
        return list(set(q.topic.value for q in questions.questions))

    def get_statistics(self) -> dict:
        """Get statistics about the question bank."""
        questions = self.load_all()

        by_topic = {}
        by_difficulty = {}

        for q in questions.questions:
            topic = q.topic.value
            diff = q.difficulty.value

            by_topic[topic] = by_topic.get(topic, 0) + 1
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

        return {
            "total": len(questions),
            "by_topic": by_topic,
            "by_difficulty": by_difficulty,
        }
