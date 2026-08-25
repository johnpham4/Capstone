"""Profile the diagram generation pipeline using mocked DSL samples.

Usage:
    uv run python profiling/profile_diagram.py
    uv run python profiling/profile_diagram.py --profile      # enable cProfile dump
    uv run python profiling/profile_diagram.py --epochs 3000  # baseline
    uv run python profiling/profile_diagram.py --epochs 800   # optimized
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the backend package root is on sys.path when running this script directly.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from loguru import logger

from profiling.mock_dsls import get_mock_dsls
from src.config.settings import settings
from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.dsl_parser import DSLParser
from src.services.diagram.generation import DiagramService
from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer
from src.services.diagram.optimizer import Optimizer


class StageTimer:
    """Simple context manager to time a code block."""

    def __init__(self, name: str, record: dict[str, float]):
        self.name = name
        self.record = record
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        self.record[self.name] = elapsed_ms
        return False


def profile_single_dsl(
    dsl: str,
    service: DiagramService,
    epochs: int,
    dpi: int = 150,
    use_early_stop: bool = True,
    dtype: str = "float32",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one DSL through all pipeline stages and record timings."""
    task_id = f"profile_{int(time.time() * 1000)}"
    timings: dict[str, float] = {}
    result: dict[str, Any] = {"dsl": dsl, "status": "unknown"}

    try:
        # Stage 1: Parse DSL
        with StageTimer("parse_ms", timings):
            lines = [line.strip() for line in dsl.splitlines() if line.strip()]
            sexprs = DSLParser.parse_sexprs(lines)

        # Stage 2: Build instruction graph
        with StageTimer("build_ms", timings):
            builder = DiagramBuilder(lines)

        # Stage 3: Optimize geometry (PyTorch differentiable solver)
        with StageTimer("optimize_ms", timings):
            optimizer_opts = {
                "epochs": epochs,
                "n_tries": 1,
                "learning_rate": 0.01,
                "seed": 42,
                "dtype": dtype,
                "early_stop_patience": 0 if not use_early_stop else settings.DIAGRAM_OPTIMIZER_EARLY_STOP_PATIENCE,
                "early_stop_min_delta": settings.DIAGRAM_OPTIMIZER_EARLY_STOP_MIN_DELTA,
                "early_stop_min_epochs": settings.DIAGRAM_OPTIMIZER_EARLY_STOP_MIN_EPOCHS,
            }
            optimizer = Optimizer(builder.instructions, optimizer_opts, verbosity=False)
            diagram, final_loss = optimizer.solve_single()

        # Stage 4: Render diagram to PNG
        with StageTimer("render_ms", timings):
            renderer = MatplotlibDiagramRenderer(diagram)
            render_dir = output_dir if output_dir else Path(settings.OUTPUT_DIR)
            render_dir.mkdir(parents=True, exist_ok=True)
            image_path = render_dir / f"{task_id}.png"
            renderer.render(diagram=diagram, show=False, save=True, filename=str(image_path))

        result.update(
            {
                "status": "success",
                "sexpr_count": len(sexprs),
                "instruction_count": len(builder.instructions),
                "point_count": len(diagram.points) if diagram else 0,
                "timings_ms": timings,
                "final_loss": final_loss,
                "warnings": builder.warnings,
            }
        )

    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": str(exc),
                "timings_ms": timings,
            }
        )

    return result


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of a sorted list (0-100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate latency statistics from profiling results."""
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    def collect(stage: str) -> list[float]:
        values = []
        for r in successful:
            t = r.get("timings_ms", {}).get(stage)
            if t is not None:
                values.append(t)
        return values

    stages = ["parse_ms", "build_ms", "optimize_ms", "render_ms"]
    losses = [r.get("final_loss") for r in successful if r.get("final_loss") is not None]
    summary: dict[str, Any] = {
        "total_samples": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "mean_final_loss": round(sum(losses) / len(losses), 6) if losses else None,
        "stage_stats": {},
    }

    for stage in stages:
        vals = collect(stage)
        if vals:
            summary["stage_stats"][stage] = {
                "count": len(vals),
                "min_ms": round(min(vals), 2),
                "max_ms": round(max(vals), 2),
                "mean_ms": round(sum(vals) / len(vals), 2),
                "p50_ms": round(percentile(vals, 50), 2),
                "p95_ms": round(percentile(vals, 95), 2),
                "p99_ms": round(percentile(vals, 99), 2),
            }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print profiling summary."""
    print("\n" + "=" * 70)
    print("Diagram Pipeline Profiling Summary")
    print("=" * 70)
    print(f"Total samples : {summary['total_samples']}")
    print(f"Successful    : {summary['successful']}")
    print(f"Failed        : {summary['failed']}")
    print("-" * 70)

    for stage, stats in summary["stage_stats"].items():
        print(f"\n{stage}:")
        for metric, value in stats.items():
            print(f"  {metric:12s}: {value}")

    optimize = summary["stage_stats"].get("optimize_ms")
    if optimize:
        print(f"\nGeometry optimizer is the dominant cost ({optimize['mean_ms']:.1f} ms mean).")


def run_cprofile_baseline(epochs: int, sample_count: int) -> str:
    """Run cProfile on a single representative DSL and return the top callers."""
    dsls = get_mock_dsls(sample_count)
    service = DiagramService()

    profiler = cProfile.Profile()
    profiler.enable()
    for dsl in dsls:
        try:
            profile_single_dsl(dsl, service, epochs=epochs)
        except Exception:
            pass
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile diagram generation pipeline")
    parser.add_argument("--epochs", type=int, default=3000, help="Optimizer epochs")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI")
    parser.add_argument("--count", type=int, default=20, help="Number of mock DSLs")
    parser.add_argument("--dtype", type=str, default=settings.DIAGRAM_OPTIMIZER_DTYPE, help="Optimizer dtype: float32 or float64")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable plateau early stopping")
    parser.add_argument("--label", type=str, default="run", help="Label for output subfolder and log file")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: None)  # suppress loguru noise during profiling

    dsls = get_mock_dsls(args.count)
    service = DiagramService()

    # Organize rendered diagrams into a labeled subfolder so baseline and
    # optimized runs do not overwrite or mix images.
    image_output_dir = Path(settings.OUTPUT_DIR) / args.label
    image_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Profiling {len(dsls)} mock DSLs with epochs={args.epochs}, dtype={args.dtype}, label='{args.label}' ...")
    print(f"Diagram images will be saved to: {image_output_dir}")

    if args.profile:
        profile_text = run_cprofile_baseline(args.epochs, len(dsls))
        print("\n" + profile_text)

    results: list[dict[str, Any]] = []
    for idx, dsl in enumerate(dsls, 1):
        print(f"  [{idx}/{len(dsls)}] Running sample {idx} ...", end="\r")
        result = profile_single_dsl(
            dsl,
            service,
            epochs=args.epochs,
            dpi=args.dpi,
            use_early_stop=not args.no_early_stop,
            dtype=args.dtype,
            output_dir=image_output_dir,
        )
        results.append(result)

    print(" " * 60, end="\r")

    summary = summarize(results)
    print_summary(summary)

    payload = {"summary": summary, "results": results}

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results written to: {output_path}")

    # Always write a timestamped text log next to the JSON for easy CV reference.
    log_path = image_output_dir / f"{args.label}_profiling_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Diagram Pipeline Profiling Log — label={args.label}\n")
        f.write(f"epochs={args.epochs}, dtype={args.dtype}, early_stop={not args.no_early_stop}, count={args.count}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total samples : {summary['total_samples']}\n")
        f.write(f"Successful    : {summary['successful']}\n")
        f.write(f"Failed        : {summary['failed']}\n")
        f.write(f"Mean final loss: {summary.get('mean_final_loss', 'N/A')}\n\n")
        for stage, stats in summary["stage_stats"].items():
            f.write(f"{stage}:\n")
            for metric, value in stats.items():
                f.write(f"  {metric:12s}: {value}\n")
            f.write("\n")
        f.write("\nPer-sample results:\n")
        for idx, r in enumerate(results, 1):
            f.write(f"\n--- Sample {idx} ---\n")
            f.write(f"DSL: {r['dsl'][:120].replace(chr(10), ' ')}\n")
            f.write(f"Status: {r['status']}\n")
            if r.get("final_loss") is not None:
                f.write(f"Final loss: {r['final_loss']:.6f}\n")
            if r.get("timings_ms"):
                for stage, ms in r["timings_ms"].items():
                    f.write(f"  {stage}: {ms:.2f} ms\n")
            if r.get("error"):
                f.write(f"Error: {r['error']}\n")
    print(f"Text log written to: {log_path}")


if __name__ == "__main__":
    main()
