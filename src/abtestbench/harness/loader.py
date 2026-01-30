"""Question loader from YAML files."""

import json
from pathlib import Path
from typing import Iterator, Optional

import yaml
from jsonschema import ValidationError, validate

from ..models.question import Question, QuestionSet


class QuestionLoader:
    """Load and validate benchmark questions from YAML files."""

    def __init__(
        self,
        questions_dir: Path,
        schema_path: Optional[Path] = None,
    ):
        self.questions_dir = Path(questions_dir)
        self.schema = self._load_schema(schema_path or self.questions_dir / "schema.json")

    def _load_schema(self, path: Path) -> dict:
        """Load the JSON schema for validation."""
        if not path.exists():
            return {}  # Skip validation if no schema

        with open(path) as f:
            return json.load(f)

    def load_all(
        self,
        categories: Optional[list[str]] = None,
        difficulties: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> QuestionSet:
        """Load all questions with optional filtering."""
        questions = []

        for file_path in self.questions_dir.glob("*.yaml"):
            questions.extend(self._load_file(file_path))

        # Apply filters
        if categories:
            questions = [q for q in questions if q.category.value in categories]
        if difficulties:
            questions = [q for q in questions if q.difficulty.value in difficulties]
        if tags:
            questions = [q for q in questions if any(t in q.tags for t in tags)]

        return QuestionSet(questions=questions)

    def _load_file(self, path: Path) -> list[Question]:
        """Load questions from a single YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        if not data or "questions" not in data:
            return []

        questions = []
        for q_data in data.get("questions", []):
            # Validate against schema if available
            if self.schema:
                try:
                    validate(instance=q_data, schema=self.schema)
                except ValidationError as e:
                    print(f"Warning: Invalid question {q_data.get('id')}: {e.message}")
                    continue

            try:
                questions.append(Question(**q_data))
            except Exception as e:
                print(f"Warning: Failed to parse question {q_data.get('id')}: {e}")
                continue

        return questions

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

    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        questions = self.load_all()
        return list(set(q.category.value for q in questions.questions))

    def get_statistics(self) -> dict:
        """Get statistics about the question bank."""
        questions = self.load_all()

        by_category = {}
        by_difficulty = {}

        for q in questions.questions:
            cat = q.category.value
            diff = q.difficulty.value

            by_category[cat] = by_category.get(cat, 0) + 1
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

        return {
            "total": len(questions),
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }
