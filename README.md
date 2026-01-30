# ABTestBench

A benchmark for evaluating LLM accuracy on A/B testing statistics questions.

## Overview

ABTestBench measures how well language models understand A/B testing statistical concepts:
- **Power Analysis** - Sample size calculation, statistical power
- **Sample Size** - Determining required sample sizes
- **Significance Testing** - p-values, multiple testing corrections
- **Confidence Intervals** - Proportion CIs, difference in means
- **Effect Size** - Cohen's d, Cohen's h, relative lift

LLMs are given access to Python code execution and evaluated using a hybrid scoring system:
- **Numeric accuracy** (70%) - Exact match with tolerance
- **Explanation quality** (30%) - LLM-as-judge evaluation

## Setup

### Prerequisites
- Python 3.11+
- Docker (for sandboxed code execution)
- [OpenRouter API key](https://openrouter.ai/)

### Installation

```bash
# Clone and enter directory
cd ABTestBench

# Install dependencies
uv sync

# Copy environment template and add your API key
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your_key_here
```

## Usage

### Run the benchmark

```bash
# Run all questions on default models
uv run python scripts/run_benchmark.py

# Test specific models
uv run python scripts/run_benchmark.py \
  --models anthropic/claude-3.5-sonnet openai/gpt-4o

# Filter by category
uv run python scripts/run_benchmark.py \
  --categories power_analysis significance_testing

# Filter by difficulty
uv run python scripts/run_benchmark.py \
  --difficulties easy medium

# List available questions
uv run python scripts/run_benchmark.py --list-questions
```

### Output

Results are saved to the `results/` directory in multiple formats:
- `results_YYYYMMDD_HHMMSS.json` - Full detailed results
- `results_YYYYMMDD_HHMMSS.md` - Markdown summary report
- `results_YYYYMMDD_HHMMSS.csv` - CSV for further analysis

## Question Bank

The benchmark includes 15 questions across 5 categories:

| Category | Questions | Topics |
|----------|-----------|--------|
| Power Analysis | 3 | Sample size for power, post-hoc power, MDE |
| Sample Size | 3 | Margin of error, A/B test sizing |
| Significance Testing | 3 | p-values, Bonferroni correction, interpretation |
| Confidence Intervals | 3 | Wald interval, difference in means, relative risk |
| Effect Size | 3 | Cohen's d, Cohen's h, practical significance |

## Adding Questions

Questions are stored in `questions/*.yaml`. Each question includes:

```yaml
- id: power_001
  category: power_analysis
  difficulty: medium
  question: |
    Your question text here...
  expected_answer:
    type: numeric
    value: 3842
    tolerance: 100
    tolerance_type: absolute
  evaluation:
    method: hybrid
    numeric_weight: 0.7
    explanation_weight: 0.3
    key_concepts: ["effect size", "power"]
  requires_code: true
```

## Architecture

```
ABTestBench/
├── src/abtestbench/
│   ├── config.py          # Configuration management
│   ├── models/            # Data models (Question, Response, Result)
│   ├── providers/         # OpenRouter LLM provider
│   ├── sandbox/           # Docker-based Python execution
│   ├── evaluation/        # Numeric + LLM-as-judge evaluators
│   ├── harness/           # Question loader, benchmark runner
│   └── reporting/         # JSON/Markdown/CSV formatters
├── questions/             # YAML question bank
├── prompts/               # System prompts
└── scripts/               # CLI scripts
```

## License

MIT
