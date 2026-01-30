"""Configuration management for ABTestBench."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class SandboxConfig(BaseSettings):
    """Docker sandbox configuration."""

    timeout_seconds: int = Field(default=30, description="Code execution timeout")
    memory_limit_mb: int = Field(default=512, description="Memory limit for sandbox")
    docker_image: str = Field(default="python:3.11-slim", description="Base Docker image")
    allowed_imports: list[str] = Field(
        default=["numpy", "scipy", "statsmodels", "pandas", "math"],
        description="Allowed Python imports in sandbox",
    )


class OpenRouterConfig(BaseSettings):
    """OpenRouter API configuration."""

    api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="OpenRouter API base URL"
    )
    max_tokens: int = Field(default=4096, description="Max tokens per response")
    temperature: float = Field(default=0.0, description="Sampling temperature")

    model_config = {"env_prefix": "OPENROUTER_", "env_file": ".env", "extra": "ignore"}


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""

    judge_model: str = Field(
        default="tngtech/deepseek-r1t2-chimera:free", description="Model for LLM-as-judge"
    )
    numeric_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for numeric accuracy"
    )
    explanation_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for explanation quality"
    )


class BenchmarkConfig(BaseSettings):
    """Main benchmark configuration."""

    questions_dir: Path = Field(default=Path("questions"), description="Questions directory")
    results_dir: Path = Field(default=Path("results"), description="Results output directory")
    prompts_dir: Path = Field(default=Path("prompts"), description="Prompts directory")

    # Sub-configurations
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Filtering options
    categories: Optional[list[str]] = Field(default=None, description="Filter to categories")
    difficulties: Optional[list[str]] = Field(default=None, description="Filter to difficulties")
    models: list[str] = Field(
        default=[
            "tngtech/deepseek-r1t2-chimera:free",
            "z-ai/glm-4.5-air:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "google/gemma-3-27b-it:free",
        ],
        description="Models to benchmark",
    )

    model_config = {"env_prefix": "ABTESTBENCH_", "env_file": ".env", "extra": "ignore"}
