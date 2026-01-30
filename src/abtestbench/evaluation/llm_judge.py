"""LLM-as-judge evaluator."""

import json
import re
from pathlib import Path
from typing import Optional

from ..models.question import Question
from ..models.response import LLMResponse
from ..models.result import JudgeEvaluation


class LLMJudgeEvaluator:
    """Use an LLM to judge response quality."""

    def __init__(self, provider: "OpenRouterProvider", model: str):
        self.provider = provider
        self.model = model
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the judge prompt template."""
        prompt_path = Path("prompts/judge_prompt.txt")
        if prompt_path.exists():
            return prompt_path.read_text()

        return """You are evaluating an LLM response to an A/B testing statistics question.

## Question
{question}

## Expected Answer
{expected_answer}

## Grading Rubric
{rubric}

## Key Concepts
{key_concepts}

## Response
{response}

## Tool Outputs
{tool_outputs}

Evaluate and respond with JSON only:
```json
{{
    "score": <0.0-1.0>,
    "reasoning": "<explanation>",
    "key_concepts_found": ["concept1"],
    "key_concepts_missing": ["concept2"]
}}
```"""

    async def evaluate(
        self,
        question: Question,
        response: LLMResponse,
    ) -> JudgeEvaluation:
        """Evaluate response using LLM-as-judge."""
        # Format tool outputs
        tool_outputs = "No tool calls made"
        if response.all_tool_results:
            tool_outputs = "\n".join(
                f"Tool: {r.tool_name}\nInput: {r.tool_input}\nOutput: {r.output}"
                for r in response.all_tool_results
            )

        # Build evaluation prompt
        prompt = self._prompt_template.format(
            question=question.question,
            expected_answer=str(question.expected_answer),
            rubric=question.evaluation.rubric or "Evaluate correctness and explanation quality",
            key_concepts=", ".join(question.evaluation.key_concepts or []),
            response=response.content,
            tool_outputs=tool_outputs,
        )

        # Get judge response
        judge_response = await self.provider.complete(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            system_prompt="You are an expert evaluator. Respond with JSON only.",
        )

        return self._parse_judge_response(judge_response.content)

    def _parse_judge_response(self, content: str) -> JudgeEvaluation:
        """Parse the judge's JSON response."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            json_str = json_match.group(0) if json_match else content

        try:
            data = json.loads(json_str)
            return JudgeEvaluation(
                score=float(data.get("score", 0.0)),
                reasoning=data.get("reasoning", ""),
                key_concepts_found=data.get("key_concepts_found", []),
                key_concepts_missing=data.get("key_concepts_missing", []),
            )
        except (json.JSONDecodeError, ValueError):
            return JudgeEvaluation(
                score=0.0,
                reasoning=f"Failed to parse judge response: {content[:200]}",
                key_concepts_found=[],
                key_concepts_missing=[],
            )
