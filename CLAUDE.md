# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABTestBench is a benchmark system for evaluating LLM performance on A/B testing statistics problems. It uses OpenRouter for model access, Docker-sandboxed code execution, and exact-match evaluation.

## Commands

```bash
# Install dependencies
uv sync

# Run the benchmark
uv run python scripts/run_benchmark.py

# Run tests
uv run pytest tests

# Lint and format
uv run ruff check src/
uv run ruff format src/
```

## Architecture

**Data Flow:**
```
CLI → BenchmarkRunner → QuestionLoader (TOML questions)
                     → OpenRouterProvider (LLM completions)
                     → CodeExtractor → Sandbox (Docker/local Python)
                     → CompositeEvaluator (exact match scoring)
                     → ReportFormatter (JSON/Markdown/CSV)
```

**Key Components:**

- **Config** (`config.py`): Pydantic models for sandbox, API, evaluation, and benchmark settings
- **Models** (`models/`): Data classes for Questions (5 topics, 4 difficulty levels), Responses, Results
- **Questions** (`questions/`): Individual TOML files per question, validated by Pydantic
- **Providers** (`providers/openrouter.py`): OpenAI-compatible API client with httpx
- **Sandbox** (`sandbox/docker_runner.py`): Docker execution with local fallback, safety validation, 30s timeout
- **Evaluation** (`evaluation/`): NumericEvaluator extracts numbers from text; CompositeEvaluator does exact match
- **Harness** (`harness/`): QuestionLoader parses TOML; BenchmarkRunner orchestrates with rate limiting (3 concurrent)
- **Reporting** (`reporting/formatters.py`): Multi-format output generation

## Configuration

- API key: Set `OPENROUTER_API_KEY` in `.env` (see `.env.example`)
- System prompt: `prompts/benchmark_system.txt`

## Code Conventions

- Python 3.11+ with type hints throughout
- Pydantic for data validation
- Async/await for I/O operations
- Line length: 100 characters (ruff)
