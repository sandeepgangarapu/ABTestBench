#!/usr/bin/env python3
"""Main benchmark runner CLI."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abtestbench.config import BenchmarkConfig
from abtestbench.harness.runner import BenchmarkRunner
from abtestbench.reporting.formatters import ReportFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run A/B Testing Statistics Benchmark for LLMs"
    )

    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="Models to test (e.g., anthropic/claude-3.5-sonnet openai/gpt-4o)",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        choices=[
            "power_analysis",
            "sample_size",
            "significance_testing",
            "confidence_intervals",
            "effect_size",
        ],
        help="Filter to specific categories",
    )
    parser.add_argument(
        "--difficulties",
        "-d",
        nargs="+",
        choices=["easy", "medium", "hard", "expert"],
        help="Filter to specific difficulties",
    )
    parser.add_argument(
        "--questions",
        "-q",
        nargs="+",
        help="Specific question IDs to run",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--format",
        "-f",
        nargs="+",
        choices=["json", "markdown", "csv"],
        default=["json", "markdown"],
        help="Output formats",
    )
    parser.add_argument(
        "--list-questions",
        action="store_true",
        help="List available questions and exit",
    )

    return parser.parse_args()


def list_questions():
    """List all available questions."""
    from abtestbench.harness.loader import QuestionLoader

    loader = QuestionLoader(Path("questions"))
    stats = loader.get_statistics()

    print("\nQuestion Bank Statistics:")
    print(f"  Total questions: {stats['total']}")

    print("\nBy Category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat}: {count}")

    print("\nBy Difficulty:")
    for diff in ["easy", "medium", "hard", "expert"]:
        if diff in stats["by_difficulty"]:
            print(f"  {diff}: {stats['by_difficulty'][diff]}")

    print("\nQuestions:")
    for q in loader.iter_questions():
        code_flag = "[code]" if q.requires_code else ""
        print(f"  {q.id} ({q.category.value}, {q.difficulty.value}) {code_flag}")


async def main():
    args = parse_args()

    if args.list_questions:
        list_questions()
        return

    # Create config
    config = BenchmarkConfig()

    if args.models:
        config.models = args.models
    if args.categories:
        config.categories = args.categories
    if args.difficulties:
        config.difficulties = args.difficulties

    # Run benchmark
    runner = BenchmarkRunner(config)

    try:
        result = await runner.run(
            models=args.models,
            categories=args.categories,
            difficulties=args.difficulties,
            question_ids=args.questions,
        )
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        sys.exit(1)

    # Generate reports
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")

    if "json" in args.format:
        json_path = args.output / f"results_{timestamp}.json"
        ReportFormatter.to_json(result, json_path)
        print(f"\nJSON results saved to: {json_path}")

    if "markdown" in args.format:
        md_path = args.output / f"results_{timestamp}.md"
        ReportFormatter.to_markdown(result, md_path)
        print(f"Markdown report saved to: {md_path}")

    if "csv" in args.format:
        csv_path = args.output / f"results_{timestamp}.csv"
        ReportFormatter.to_csv(result, csv_path)
        print(f"CSV results saved to: {csv_path}")

    # Print summary
    ReportFormatter.print_summary(result)


if __name__ == "__main__":
    asyncio.run(main())
