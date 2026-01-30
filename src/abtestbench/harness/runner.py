"""Main benchmark runner."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import BenchmarkConfig
from ..evaluation.composite import CompositeEvaluator
from ..models.question import Question, QuestionSet
from ..models.response import LLMResponse
from ..models.result import BenchmarkResult, QuestionResult
from ..providers.openrouter import OpenRouterProvider, ToolExecutor
from ..sandbox.docker_runner import PYTHON_TOOL_DEFINITION, DockerSandbox, LocalSandbox, DOCKER_AVAILABLE
from .loader import QuestionLoader


class BenchmarkRunner:
    """Main benchmark orchestrator."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.loader = QuestionLoader(self.config.questions_dir)
        self.provider = OpenRouterProvider(self.config.openrouter)

        # Use Docker sandbox if available, otherwise fall back to local execution
        if DOCKER_AVAILABLE:
            try:
                self.sandbox = DockerSandbox(self.config.sandbox)
                print("Using Docker sandbox for code execution")
            except Exception as e:
                print(f"Docker unavailable ({e}), using local execution")
                self.sandbox = LocalSandbox(self.config.sandbox)
        else:
            print("Docker not available, using local execution")
            self.sandbox = LocalSandbox(self.config.sandbox)

        self.evaluator = CompositeEvaluator(
            provider=self.provider,
            judge_model=self.config.evaluation.judge_model,
        )

        # Set up tool executor
        self.tool_executor = ToolExecutor()
        self.tool_executor.register("execute_python", self.sandbox)

    async def run(
        self,
        models: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
        difficulties: Optional[list[str]] = None,
        question_ids: Optional[list[str]] = None,
    ) -> BenchmarkResult:
        """Run the complete benchmark."""
        # Load questions
        questions = self.loader.load_all(
            categories=categories or self.config.categories,
            difficulties=difficulties or self.config.difficulties,
        )

        if question_ids:
            questions = QuestionSet(
                questions=[q for q in questions.questions if q.id in question_ids]
            )

        if not questions.questions:
            raise ValueError("No questions found matching the specified filters")

        # Get models to test
        models_to_test = models or self.config.models

        print(f"Running benchmark with {len(questions)} questions across {len(models_to_test)} models")

        # Run benchmark for each model
        results: dict[str, list[QuestionResult]] = {}

        for model in models_to_test:
            print(f"\n{'='*60}")
            print(f"Testing model: {model}")
            print(f"{'='*60}")

            results[model] = await self._run_model(model, questions)

        return BenchmarkResult(
            timestamp=datetime.utcnow(),
            results=results,
        )

    async def _run_model(
        self,
        model: str,
        questions: QuestionSet,
    ) -> list[QuestionResult]:
        """Run all questions for a single model."""
        results = []

        # Use semaphore for rate limiting
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests

        async def run_question(question: Question) -> QuestionResult:
            async with semaphore:
                return await self._run_single_question(model, question)

        # Run questions with progress
        for i, question in enumerate(questions.questions, 1):
            print(f"  [{i}/{len(questions)}] {question.id}...", end=" ", flush=True)
            result = await run_question(question)
            results.append(result)

            if result.success and result.evaluation:
                print(f"Score: {result.evaluation.overall_score:.2f}")
            else:
                print(f"Error: {result.error}")

        return results

    async def _run_single_question(
        self,
        model: str,
        question: Question,
    ) -> QuestionResult:
        """Run a single question and evaluate the response."""
        # Build the prompt
        system_prompt = self._get_system_prompt()
        user_message = self._format_question(question)

        messages = [{"role": "user", "content": user_message}]

        start_time = datetime.utcnow()

        try:
            # Get LLM response with tool use if needed
            if question.requires_code:
                response = await self.provider.complete_with_tool_loop(
                    messages=messages,
                    model=model,
                    tools=[PYTHON_TOOL_DEFINITION],
                    tool_executor=self.tool_executor,
                    system_prompt=system_prompt,
                )
            else:
                response = await self.provider.complete(
                    messages=messages,
                    model=model,
                    system_prompt=system_prompt,
                )

            elapsed_time = (datetime.utcnow() - start_time).total_seconds()

            # Evaluate the response
            evaluation = await self.evaluator.evaluate(
                question=question,
                response=response,
            )

            return QuestionResult(
                question_id=question.id,
                category=question.category.value,
                difficulty=question.difficulty.value,
                success=True,
                response=response.model_dump(),
                evaluation=evaluation,
                elapsed_seconds=elapsed_time,
            )

        except Exception as e:
            elapsed_time = (datetime.utcnow() - start_time).total_seconds()
            return QuestionResult(
                question_id=question.id,
                category=question.category.value,
                difficulty=question.difficulty.value,
                success=False,
                error=str(e),
                elapsed_seconds=elapsed_time,
            )

    def _get_system_prompt(self) -> str:
        """Load the system prompt."""
        prompt_path = self.config.prompts_dir / "benchmark_system.txt"
        if prompt_path.exists():
            return prompt_path.read_text()

        return """You are an expert statistician specializing in A/B testing.
Use the execute_python tool for calculations. State your final answer clearly."""

    def _format_question(self, question: Question) -> str:
        """Format a question for the LLM."""
        text = question.question
        if question.context:
            text += f"\n\nAdditional context:\n{question.context}"
        return text


async def run_benchmark(
    models: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """Convenience function to run the benchmark."""
    runner = BenchmarkRunner(config)
    return await runner.run(models=models, categories=categories)
