"""Numeric answer evaluator."""

import re
from typing import Optional

from ..models.question import ExpectedAnswer, NumericAnswer, NumericRangeAnswer
from ..models.response import LLMResponse
from ..models.result import NumericEvaluation


class NumericEvaluator:
    """Evaluate numeric answers with tolerance."""

    # Patterns to extract numeric answers from text
    ANSWER_PATTERNS = [
        # Explicit answer statements
        r"(?:final answer|answer is|result is|equals?|=)[:\s]*\$?([+-]?\d+\.?\d*)",
        r"(?:approximately|≈|about)[:\s]*\$?([+-]?\d+\.?\d*)",
        # Bold/emphasized numbers
        r"\*\*\$?([+-]?\d+\.?\d*)\*\*",
        # Numbers at end of response
        r":\s*\$?([+-]?\d+\.?\d*)\s*$",
        # Percentage patterns
        r"([+-]?\d+\.?\d*)\s*%",
        # Sample size patterns
        r"(?:sample size|n\s*=)[:\s]*([+-]?\d+\.?\d*)",
        # p-value patterns
        r"(?:p-value|p\s*=)[:\s]*([+-]?\d+\.?\d*)",
    ]

    def evaluate(
        self,
        response: LLMResponse,
        expected: ExpectedAnswer,
    ) -> NumericEvaluation:
        """Evaluate a numeric response."""
        # Extract numeric value from response
        extracted = self._extract_number(response.content)

        # Also try extracting from tool outputs
        if extracted is None and response.all_tool_results:
            for result in response.all_tool_results:
                extracted = self._extract_number(result.output)
                if extracted is not None:
                    break

        # Handle numeric answer
        if isinstance(expected, NumericAnswer):
            return self._evaluate_numeric(extracted, expected)

        # Handle numeric range answer
        elif isinstance(expected, NumericRangeAnswer):
            return self._evaluate_range(extracted, expected)

        # Unsupported answer type
        return NumericEvaluation(
            correct=False,
            extracted_value=extracted,
            expected_value=0.0,
            difference=None,
            within_tolerance=False,
        )

    def _evaluate_numeric(
        self, extracted: Optional[float], expected: NumericAnswer
    ) -> NumericEvaluation:
        """Evaluate against a single numeric value."""
        if extracted is None:
            return NumericEvaluation(
                correct=False,
                extracted_value=None,
                expected_value=expected.value,
                difference=None,
                within_tolerance=False,
            )

        # Calculate difference based on tolerance type
        if expected.tolerance_type == "relative":
            if expected.value == 0:
                difference = abs(extracted)
            else:
                difference = abs(extracted - expected.value) / abs(expected.value)
            within_tolerance = difference <= expected.tolerance
        else:  # absolute
            difference = abs(extracted - expected.value)
            within_tolerance = difference <= expected.tolerance

        return NumericEvaluation(
            correct=within_tolerance,
            extracted_value=extracted,
            expected_value=expected.value,
            difference=difference,
            within_tolerance=within_tolerance,
        )

    def _evaluate_range(
        self, extracted: Optional[float], expected: NumericRangeAnswer
    ) -> NumericEvaluation:
        """Evaluate against a numeric range."""
        if extracted is None:
            return NumericEvaluation(
                correct=False,
                extracted_value=None,
                expected_value=(expected.min + expected.max) / 2,
                difference=None,
                within_tolerance=False,
            )

        within_range = expected.min <= extracted <= expected.max

        # Calculate distance from range
        if within_range:
            difference = 0.0
        elif extracted < expected.min:
            difference = expected.min - extracted
        else:
            difference = extracted - expected.max

        return NumericEvaluation(
            correct=within_range,
            extracted_value=extracted,
            expected_value=(expected.min + expected.max) / 2,
            difference=difference,
            within_tolerance=within_range,
        )

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a numeric answer from text."""
        if not text:
            return None

        # Try each pattern
        for pattern in self.ANSWER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Return the last match (most likely to be the final answer)
                    return float(matches[-1])
                except ValueError:
                    continue

        # Fallback: find all numbers and return the last one
        numbers = re.findall(r"[+-]?\d+\.?\d*", text)
        if numbers:
            # Filter out very small numbers that are likely not answers
            valid_numbers = []
            for n in numbers:
                try:
                    val = float(n)
                    # Skip numbers that look like dates or versions
                    if 1900 <= val <= 2100 and val == int(val):
                        continue
                    valid_numbers.append(val)
                except ValueError:
                    continue

            if valid_numbers:
                return valid_numbers[-1]

        return None
