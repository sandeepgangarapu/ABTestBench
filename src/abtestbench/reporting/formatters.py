"""Report formatters for benchmark results."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.result import BenchmarkResult


class ReportFormatter:
    """Generate benchmark reports in various formats."""

    @staticmethod
    def to_json(result: BenchmarkResult, path: Path) -> None:
        """Export results to JSON."""
        data: dict[str, Any] = {
            "timestamp": result.timestamp.isoformat(),
            "summaries": {},
            "detailed_results": {},
        }

        for model in result.results:
            summary = result.get_summary(model)
            data["summaries"][model] = {
                "provider": summary.provider,
                "model": summary.model,
                "overall_accuracy": round(summary.overall_accuracy, 4),
                "numeric_accuracy": round(summary.numeric_accuracy, 4),
                "by_topic": {k: round(v, 4) for k, v in summary.by_topic.items()},
                "by_difficulty": {k: round(v, 4) for k, v in summary.by_difficulty.items()},
                "total_questions": summary.total_questions,
                "successful": summary.successful,
                "failed": summary.failed,
                "total_time_seconds": round(summary.total_time_seconds, 2),
            }

            data["detailed_results"][model] = [
                {
                    "question_id": r.question_id,
                    "topic": r.topic,
                    "difficulty": r.difficulty,
                    "success": r.success,
                    "overall_score": (
                        round(r.evaluation.overall_score, 4) if r.evaluation else None
                    ),
                    "numeric_score": (
                        round(r.evaluation.numeric_score, 4) if r.evaluation else None
                    ),
                    "elapsed_seconds": round(r.elapsed_seconds, 2),
                    "error": r.error,
                }
                for r in result.results[model]
            ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def to_markdown(result: BenchmarkResult, path: Path) -> None:
        """Generate a Markdown report."""
        lines = [
            "# A/B Testing Statistics Benchmark Results",
            f"\nGenerated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "\n## Summary\n",
        ]

        # Summary table
        lines.append("| Model | Overall | Numeric | Questions |")
        lines.append("|-------|---------|---------|-----------|")

        for model in result.results:
            summary = result.get_summary(model)
            lines.append(
                f"| {model} | {summary.overall_accuracy:.1%} | "
                f"{summary.numeric_accuracy:.1%} | "
                f"{summary.successful}/{summary.total_questions} |"
            )

        # By Topic breakdown
        lines.append("\n## Results by Topic\n")

        for model in result.results:
            summary = result.get_summary(model)
            lines.append(f"\n### {model}\n")
            lines.append("| Topic | Accuracy |")
            lines.append("|-------|----------|")
            for topic, acc in sorted(summary.by_topic.items()):
                lines.append(f"| {topic} | {acc:.1%} |")

        # By Difficulty breakdown
        lines.append("\n## Results by Difficulty\n")

        for model in result.results:
            summary = result.get_summary(model)
            lines.append(f"\n### {model}\n")
            lines.append("| Difficulty | Accuracy |")
            lines.append("|------------|----------|")
            for diff in ["easy", "medium", "hard", "expert"]:
                if diff in summary.by_difficulty:
                    lines.append(f"| {diff} | {summary.by_difficulty[diff]:.1%} |")

        # Individual Results
        lines.append("\n## Detailed Results\n")

        for model in result.results:
            lines.append(f"\n### {model}\n")
            lines.append("| Question | Topic | Difficulty | Score | Time |")
            lines.append("|----------|-------|------------|-------|------|")

            for r in result.results[model]:
                score = f"{r.evaluation.overall_score:.2f}" if r.evaluation else "ERROR"
                lines.append(
                    f"| {r.question_id} | {r.topic} | {r.difficulty} | "
                    f"{score} | {r.elapsed_seconds:.1f}s |"
                )

        with open(path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def to_csv(result: BenchmarkResult, path: Path) -> None:
        """Export detailed results to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "question_id",
                    "topic",
                    "difficulty",
                    "overall_score",
                    "numeric_score",
                    "success",
                    "elapsed_seconds",
                    "error",
                ]
            )

            for model in result.results:
                for r in result.results[model]:
                    writer.writerow(
                        [
                            model,
                            r.question_id,
                            r.topic,
                            r.difficulty,
                            r.evaluation.overall_score if r.evaluation else "",
                            r.evaluation.numeric_score if r.evaluation else "",
                            r.success,
                            round(r.elapsed_seconds, 2),
                            r.error or "",
                        ]
                    )

    @staticmethod
    def print_summary(result: BenchmarkResult) -> None:
        """Print a summary to stdout."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        for model in result.results:
            summary = result.get_summary(model)
            print(f"\n{model}:")
            print(f"  Overall Accuracy:    {summary.overall_accuracy:.1%}")
            print(f"  Numeric Accuracy:    {summary.numeric_accuracy:.1%}")
            print(f"  Questions: {summary.successful}/{summary.total_questions}")
            print(f"  Time: {summary.total_time_seconds:.1f}s")

            if summary.by_topic:
                print("\n  By Topic:")
                for topic, acc in sorted(summary.by_topic.items()):
                    print(f"    {topic}: {acc:.1%}")

            if summary.by_difficulty:
                print("\n  By Difficulty:")
                for diff in ["easy", "medium", "hard", "expert"]:
                    if diff in summary.by_difficulty:
                        print(f"    {diff}: {summary.by_difficulty[diff]:.1%}")
