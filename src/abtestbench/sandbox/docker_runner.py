"""Docker-based sandboxed Python execution."""

import asyncio
import io
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import docker
    from docker.errors import ContainerError, ImageNotFound
    DOCKER_AVAILABLE = True
except Exception:
    DOCKER_AVAILABLE = False

from ..config import SandboxConfig


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str
    error: Optional[str] = None
    timed_out: bool = False


class LocalSandbox:
    """Local Python execution (fallback when Docker unavailable)."""

    WRAPPER_TEMPLATE = '''
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Pre-import allowed modules
import numpy as np
import scipy.stats as stats
from scipy import stats as scipy_stats
import pandas as pd
import math
from statsmodels.stats.power import TTestIndPower, NormalIndPower, tt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        exec("""
{code}
""")
    output = stdout_capture.getvalue()
    if output:
        print(output)
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()

    def _validate_code(self, code: str) -> Optional[ExecutionResult]:
        """Basic code validation for dangerous patterns."""
        dangerous_patterns = [
            "os.system",
            "subprocess",
            "__import__",
            "open(",
            "file(",
            "input(",
            "quit(",
            "exit(",
            "os.remove",
            "os.unlink",
            "shutil.rmtree",
            "import os",
            "from os",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Disallowed operation: {pattern}",
                )

        return None

    async def execute(self, arguments: dict) -> ExecutionResult:
        """Execute Python code locally."""
        code = arguments.get("code", "")

        if not code.strip():
            return ExecutionResult(
                success=False,
                output="",
                error="No code provided",
            )

        # Validate code
        validation_error = self._validate_code(code)
        if validation_error:
            return validation_error

        # Create wrapper script
        wrapper_code = self.WRAPPER_TEMPLATE.format(
            code=code.replace('"""', '\\"\\"\\"'),
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(wrapper_code)
            script_path = Path(f.name)

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._run_script(script_path)
                ),
                timeout=self.config.timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.config.timeout_seconds}s",
                timed_out=True,
            )
        finally:
            script_path.unlink(missing_ok=True)

    def _run_script(self, script_path: Path) -> ExecutionResult:
        """Run the script as a subprocess."""
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            output = result.stdout.strip()
            error = result.stderr.strip() if result.returncode != 0 else None

            return ExecutionResult(
                success=result.returncode == 0,
                output=output,
                error=error,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.config.timeout_seconds}s",
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
            )


class DockerSandbox:
    """Docker-based sandboxed Python execution."""

    DOCKERFILE = """
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy scipy statsmodels pandas
WORKDIR /app
"""

    WRAPPER_TEMPLATE = '''
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Pre-import allowed modules
import numpy as np
import scipy.stats as stats
from scipy import stats as scipy_stats
import pandas as pd
import math
from statsmodels.stats.power import TTestIndPower, NormalIndPower, tt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        exec("""
{code}
""")
    output = stdout_capture.getvalue()
    if output:
        print(output)
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.client = docker.from_env()
        self.image_name = "abtestbench-sandbox:latest"
        self._image_ready = False

    def _ensure_image(self) -> None:
        """Ensure the Docker image exists with required packages."""
        if self._image_ready:
            return

        try:
            self.client.images.get(self.image_name)
            self._image_ready = True
        except ImageNotFound:
            print("Building sandbox Docker image (first run only)...")
            dockerfile_bytes = io.BytesIO(self.DOCKERFILE.encode())
            self.client.images.build(
                fileobj=dockerfile_bytes,
                tag=self.image_name,
                rm=True,
            )
            self._image_ready = True

    def _validate_code(self, code: str) -> Optional[ExecutionResult]:
        """Basic code validation for dangerous patterns."""
        dangerous_patterns = [
            "os.system",
            "subprocess",
            "eval(",
            "exec(",  # except our wrapper
            "__import__",
            "open(",
            "file(",
            "input(",
            "quit(",
            "exit(",
            "os.remove",
            "os.unlink",
            "shutil.rmtree",
            "import os",
            "from os",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Disallowed operation: {pattern}",
                )

        return None

    async def execute(self, arguments: dict) -> ExecutionResult:
        """Execute Python code in Docker sandbox."""
        code = arguments.get("code", "")

        if not code.strip():
            return ExecutionResult(
                success=False,
                output="",
                error="No code provided",
            )

        # Validate code
        validation_error = self._validate_code(code)
        if validation_error:
            return validation_error

        # Ensure image exists
        self._ensure_image()

        # Create wrapper script
        wrapper_code = self.WRAPPER_TEMPLATE.format(
            code=code.replace('"""', '\\"\\"\\"'),
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(wrapper_code)
            script_path = Path(f.name)

        try:
            # Run in container
            result = await self._run_in_container(script_path)
            return result
        finally:
            script_path.unlink(missing_ok=True)

    async def _run_in_container(self, script_path: Path) -> ExecutionResult:
        """Run script in Docker container."""
        try:
            # Run container
            container = self.client.containers.run(
                self.image_name,
                command=["python", "/app/script.py"],
                volumes={str(script_path): {"bind": "/app/script.py", "mode": "ro"}},
                mem_limit=f"{self.config.memory_limit_mb}m",
                network_disabled=True,
                remove=True,
                detach=True,
                stderr=True,
                stdout=True,
            )

            # Wait with timeout
            try:
                exit_code = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: container.wait(timeout=self.config.timeout_seconds)
                    ),
                    timeout=self.config.timeout_seconds + 5,
                )

                logs = container.logs(stdout=True, stderr=True).decode("utf-8")

                return ExecutionResult(
                    success=exit_code.get("StatusCode", 1) == 0,
                    output=logs.strip(),
                    error=None if exit_code.get("StatusCode", 1) == 0 else logs.strip(),
                )

            except asyncio.TimeoutError:
                try:
                    container.kill()
                except Exception:
                    pass
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.config.timeout_seconds}s",
                    timed_out=True,
                )

        except ContainerError as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Container error: {str(e)}",
            )


# Tool definition for LLM tool use
PYTHON_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": """Execute Python code for statistical calculations.

Available libraries: numpy, scipy, statsmodels, pandas, math

Use this tool to:
- Calculate sample sizes and statistical power
- Run statistical tests (t-tests, z-tests, chi-square)
- Compute confidence intervals
- Calculate effect sizes (Cohen's d, Cohen's h)
- Perform numerical computations

The code should print the final result.""",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Print your results.",
                }
            },
            "required": ["code"],
        },
    },
}
