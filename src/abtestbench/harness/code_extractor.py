"""Extract Python code blocks from LLM responses."""

import re
from typing import Optional


class CodeExtractor:
    """Extract Python code from markdown code blocks."""

    PATTERN = r'```python\s*\n(.*?)```'

    @staticmethod
    def extract(content: str) -> Optional[str]:
        """Extract the last Python code block from content.

        Returns the last code block since models often refine their code
        in subsequent blocks.
        """
        matches = re.findall(CodeExtractor.PATTERN, content, re.DOTALL)
        return matches[-1].strip() if matches else None
