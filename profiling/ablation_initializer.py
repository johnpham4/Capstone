"""Ablation study: Random vs Smart (Geometry-Aware) initialization.

Compares the differentiable geometry optimizer under two initialization
strategies on an identical constraint graph:

  * smart  : canonical Geometric-Aware templates from Initializer
  * random : every point sampled from U(-1, 1), no geometric prior

Metrics per strategy (across DSLs x seeds):
  * Success Rate (%)      : no exception AND final_loss <= tau AND non-degenerate
  * Avg. Epochs           : mean optimizer iterations to convergence/early-stop
  * Final Loss            : mean (and std) of final optimization loss
  * Degenerate Cases (%)  : runs whose resolved diagram is geometrically invalid

Usage:
  uv run python profiling/ablation_initializer.py --seeds 5 --epochs 1000
  uv run python profiling/ablation_initializer.py --repo quangne/geometry3k8-8-1-1 \
      --split test --field output --max-samples 200 --seeds 5 --epochs 1000 \
      --output profiling/ablation_results.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# --------------------------------------------------------------------------- #
# Serialized child-process startup.
#
# Spawning many workers at once makes them import torch concurrently, and
# torch's shm.dll intermittently fails to initialize (WinError 1114) on
# Windows under that race. Worker processes are marked via the ABL_CHILD env
# var (inherited through spawn); each child then takes turns importing the
# heavy modules while holding an exclusive lock directory.
# --------------------------------------------------------------------------- #
_ABL_LOCK_DIR = _BACKEND_ROOT / "profiling" / ".abl_import_lock"
if os.environ.get("ABL_CHILD") == "1":
    _deadline = time.monotonic() + 180.0
    while True:
        try:
            _ABL_LOCK_DIR.mkdir(parents=True)
            break
        except FileExistsError:
            if time.monotonic() > _deadline:
                break  # proceed anyway rather than deadlock
            time.sleep(0.2)

from loguru import logger

from profiling.mock_dsls import get_mock_dsls
from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.model.entities import Diagram
from src.services.diagram.optimizer import Optimizer

# Keep worker-process output clean (child processes re-import this module).
logger.remove()

# Release the startup lock once this child finished importing heavy modules.
if os.environ.get("ABL_CHILD") == "1":
    try:
        _ABL_LOCK_DIR.rmdir()
    except OSError:
        pass

DEFAULT_TAU = 0.5
AREA_EPS = 1e-3          # min polygon area to be considered non-degenerate
COINCIDENT_EPS = 1e-3    # min distance between any two distinct points


# --------------------------------------------------------------------------- #
# Validity / degeneracy detection
# --------------------------------------------------------------------------- #
def _polygon_area(points: list) -> float:
    """Shoelace area of a 2D polygon given [(x, y), ...]."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def check_diagram_degenerate(diagram: Diagram | None) -> tuple[bool, list[str]]:
    """Return (is_degenerate, reasons) for a resolved diagram.

    A diagram is degenerate if:
      - no points were produced at all, or
      - a declared polygon has (near-)zero area (collapsed/collinear), or
      - two distinct points coincide within COINCIDENT_EPS.
    """
    if diagram is None:
        return True, ["no diagram"]
    if not diagram.points:
        return True, ["no points produced"]

    reasons: list[str] = []

    for tri in diagram.triangles:
        p1, p2, p3 = tri[0], tri[1], tri[2]
        area = _polygon_area([(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)])
        if area < AREA_EPS:
            reasons.append(f"triangle {p1.name}-{p2.name}-{p3.name} area={area:.5f}")

    for quad in diagram.quadrilaterals:
        pts = quad.get('points', [])
        if len(pts) == 4:
            area = _polygon_area([(p.x, p.y) for p in pts])
            if area < AREA_EPS:
                names = "-".join(str(p.name) for p in pts)
                reasons.append(f"quadrilateral {names} area={area:.5f}")

    point_list = list(diagram.points.values())
    for i in range(len(point_list)):
        for j in range(i + 1, len(point_list)):
            a, b = point_list[i], point_list[j]
            d = math.hypot(a.x - b.x, a.y - b.y)
            if d < COINCIDENT_EPS:
                reasons.append(f"coincident points {a.name}~{b.name} d={d:.5f}")

    return bool(reasons), reasons


# --------------------------------------------------------------------------- #
# Single-case runner
# --------------------------------------------------------------------------- #
def _worker_init() -> None:
    """Stagger worker startup to dodge concurrent torch/shm.dll init races."""
    time.sleep(random.uniform(0.5, 4.0))


def run_case(dsl_idx: int, dsl: str, mode: str, seed: int,
             opts: dict[str, Any]) -> dict[str, Any]:
    """Solve one DSL under a given init mode and seed; record metrics."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)

    lines = [ln.strip() for ln in dsl.splitlines() if ln.strip()]
    result: dict[str, Any] = {
        "dsl_idx": dsl_idx,
        "dsl": dsl,
        "mode": mode,
        "seed": seed,
        "status": "ok",
        "final_loss": None,
        "epochs_used": None,
        "epochs_to_tau": None,
        "converged": None,
        "degenerate": None,
        "degenerate_reasons": [],
        "point_count": None,
        "elapsed_seconds": None,
        "image_path": None,
        "error": None,
    }

    t0 = time.perf_counter()
    diagram = None
    try:
        builder = DiagramBuilder(lines)
        optimizer_opts = {
            "epochs": opts["epochs"],
            "n_tries": 1,
            "learning_rate": opts["lr"],
            "seed": seed,
            "dtype": opts["dtype"],
            "init_mode": mode,
            "success_tau": opts["tau"],
            "early_stop_patience": opts["early_stop_patience"],
            "early_stop_min_delta": opts["early_stop_min_delta"],
            "early_stop_min_epochs": opts["early_stop_min_epochs"],
        }
        optimizer = Optimizer(builder.instructions, optimizer_opts, verbosity=False)
        diagram, final_loss = optimizer.solve_single()

        result["final_loss"] = float(final_loss)
        result["epochs_used"] = int(optimizer.epochs_used)
        result["epochs_to_tau"] = int(optimizer.epochs_to_tau) if optimizer.epochs_to_tau is not None else None
        result["converged"] = bool(optimizer.converged)
        result["point_count"] = len(diagram.points) if diagram else 0
        degenerate, reasons = check_diagram_degenerate(diagram)
        result["degenerate"] = degenerate
        result["degenerate_reasons"] = reasons

    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"

    # Render the final diagram (best-effort; never affects metrics).
    render_dir = opts.get("render_dir")
    if render_dir and diagram is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer

            out_dir = Path(render_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{opts.get('tag', 'run')}_{dsl_idx:03d}_{mode}_seed{seed}.png"
            fpath = out_dir / fname
            MatplotlibDiagramRenderer(diagram).render(
                show=False, save=True, filename=str(fpath)
            )
            import matplotlib.pyplot as plt
            plt.close("all")
            result["image_path"] = str(fpath)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Render failed for #{dsl_idx} [{mode}]: {exc}")

    result["elapsed_seconds"] = round(time.perf_counter() - t0, 4)
    return result


def is_success(result: dict[str, Any], tau: float) -> bool:
    if result["status"] != "ok":
        return False
    if result["final_loss"] is None or not math.isfinite(result["final_loss"]):
        return False
    if result["final_loss"] > tau:
        return False
    if result.get("degenerate", True):
        return False
    if not result.get("point_count"):
        return False
    return True


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_hf_dsls(repo: str, split: str, field: str, max_samples: int | None) -> list[str]:
    from datasets import load_dataset

    logger.info(f"Loading {repo} split='{split}' field='{field}' from Hugging Face ...")
    ds = load_dataset(repo, split=split)
    dsls: list[str] = []
    skipped = 0
    for i, sample in enumerate(ds):
        if max_samples is not None and len(dsls) >= max_samples:
            break
        raw = sample.get(field)
        if not isinstance(raw, str) or not raw.strip():
            skipped += 1
            continue
        dsls.append(raw.strip())
    logger.info(f"Loaded {len(dsls)} DSLs (skipped {skipped} empty/non-string rows)")
    return dsls


def load_dsls(args: argparse.Namespace) -> list[str]:
    if args.repo:
        return load_hf_dsls(args.repo, args.split, args.field, args.max_samples)
    dsls = get_mock_dsls(args.max_samples)
    logger.info(f"Using {len(dsls)} built-in mock DSLs")
    return dsls


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate(results: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    total = len(results)
    successes = [r for r in results if is_success(r, tau)]
    failed = [r for r in results if r["status"] != "ok"]
    degenerates = [r for r in results if r.get("degenerate")]

    completed = [r for r in results if r.get("final_loss") is not None]
    losses = [r["final_loss"] for r in completed if math.isfinite(r["final_loss"])]
    epochs = [r["epochs_used"] for r in completed if r.get("epochs_used") is not None]
    # Convergence speed: first epoch where the loss crossed below tau
    # (only among runs that actually reached the threshold).
    reached_tau = [r for r in results if r.get("epochs_to_tau") is not None]
    tau_epochs = [r["epochs_to_tau"] for r in reached_tau]

    loss_mean, loss_std = _mean_std(losses)
    epochs_mean, epochs_std = _mean_std(epochs)
    tau_mean, tau_std = _mean_std(tau_epochs)
    times = [r["elapsed_seconds"] for r in completed if r.get("elapsed_seconds") is not None]
    time_mean, time_std = _mean_std(times)

    return {
        "total_runs": total,
        "successes": len(successes),
        "failed_runs": len(failed),
        "degenerate_runs": len(degenerates),
        "success_rate_pct": round(100.0 * len(successes) / total, 2) if total else 0.0,
        "degenerate_pct": round(100.0 * len(degenerates) / total, 2) if total else 0.0,
        "avg_epochs": round(epochs_mean, 2) if epochs_mean is not None else None,
        "avg_epochs_std": round(epochs_std, 2) if epochs_std is not None else None,
        "avg_epochs_to_tau": round(tau_mean, 2) if tau_mean is not None else None,
        "avg_epochs_to_tau_std": round(tau_std, 2) if tau_std is not None else None,
        "runs_reached_tau": len(reached_tau),
        "final_loss_mean": round(loss_mean, 6) if loss_mean is not None else None,
        "final_loss_std": round(loss_std, 6) if loss_std is not None else None,
        "avg_time_s": round(time_mean, 3) if time_mean is not None else None,
        "avg_time_s_std": round(time_std, 3) if time_std is not None else None,
    }


def latex_table(summaries: dict[str, dict[str, Any]], tau: float) -> str:
    rows = []
    for label, key in (("Random Initialization", "random"), ("Smart Initializer", "smart")):
        s = summaries[key]
        sr = "---" if s["total_runs"] == 0 else f"{s['success_rate_pct']:.1f}"
        # Report epochs-to-tau (convergence speed) when available; otherwise total epochs.
        if s["avg_epochs_to_tau"] is not None:
            ep = f"{s['avg_epochs_to_tau']:.0f} $\\pm$ {s['avg_epochs_to_tau_std']:.0f}"
        else:
            ep = "---" if s["avg_epochs"] is None else f"{s['avg_epochs']:.0f} $\\pm$ {s['avg_epochs_std']:.0f}"
        fl = "---" if s["final_loss_mean"] is None else f"{s['final_loss_mean']:.4f}"
        dg = "---" if s["total_runs"] == 0 else f"{s['degenerate_pct']:.1f}"
        bold = "\\textbf{" if key == "smart" else ""
        bold_end = "}" if key == "smart" else ""
        rows.append(
            f"{label} & {bold}{sr}{bold_end} & {ep} & {fl} & {dg} \\\\"
        )
    table = (
        "\\begin{table}[h]\n"
        "\\caption{Ablation study of the proposed Smart Initializer.}\n"
        "\\label{tab:ablation_initializer}\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Initialization} & "
        "\\textbf{Success Rate (\\%)} & "
        "\\textbf{Avg. Epochs} & "
        "\\textbf{Final Loss} & "
        "\\textbf{Degenerate Cases (\\%)} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return table


# --------------------------------------------------------------------------- #
# Per-diagram report (CSV + HTML gallery for manual inspection)
# --------------------------------------------------------------------------- #
def write_report(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Write report.csv and a side-by-side gallery.html for eyeballing."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda r: (r.get("dsl_idx", 0), r["mode"]))
    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dsl_idx", "mode", "status", "final_loss", "epochs_used",
                    "epochs_to_tau", "solve_time_s", "degenerate_auto_flag",
                    "auto_reasons", "point_count", "image_path", "error"])
        for r in rows:
            w.writerow([
                r.get("dsl_idx"), r["mode"], r["status"],
                r.get("final_loss"), r.get("epochs_used"), r.get("epochs_to_tau"),
                r.get("elapsed_seconds"),
                "" if r.get("degenerate") is None else ("YES" if r["degenerate"] else "no"),
                "; ".join(r.get("degenerate_reasons") or [])[:200],
                r.get("point_count"), r.get("image_path") or "",
                (r.get("error") or "")[:200],
            ])

    # Group by DSL index for the side-by-side gallery.
    by_dsl: dict[int, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_dsl.setdefault(r.get("dsl_idx", 0), {})[r["mode"]] = r

    def _img_tag(path: str | None) -> str:
        if not path or not Path(path).exists():
            return "<div class='missing'>no image</div>"
        b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"<img src='data:image/png;base64,{b64}' loading='lazy'>"

    def _fmt(v: Any, nd: int = 3) -> str:
        if v is None:
            return "&mdash;"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Ablation Gallery</title><style>",
        "body{font-family:Segoe UI,Arial,sans-serif;background:#111;color:#eee;margin:20px}",
        ".card{background:#1c1c1c;border-radius:10px;padding:14px;margin-bottom:24px}",
        "h2{color:#8ab4ff;font-size:18px;margin:0 0 8px}",
        ".grid{display:flex;gap:12px;flex-wrap:wrap}",
        ".pane{flex:1;min-width:340px;background:#242424;border-radius:8px;padding:10px}",
        ".pane img{width:100%;border-radius:6px}",
        ".flag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;margin-left:6px}",
        ".bad{background:#5c1a1a;color:#ff9c9c}.good{background:#123f1e;color:#8ef0a8}",
        ".missing{padding:40px;text-align:center;color:#777}",
        "table{font-size:13px;border-collapse:collapse;width:100%}",
        "td{padding:2px 6px}.k{color:#9aa}</style></head><body>",
        "<h1>Random vs Smart &mdash; per-diagram report</h1>",
    ]
    for idx in sorted(by_dsl):
        modes = by_dsl[idx]
        any_r = next(iter(modes.values()))
        dsl_text = any_r["dsl"].replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f"<div class='card'><h2>#{idx}")
        for mode in ("random", "smart"):
            r = modes.get(mode)
            if not r:
                continue
            flag_cls = "bad" if r.get("degenerate") else "good"
            label = "AUTO-FLAG: degenerate" if r.get("degenerate") else "auto-flag: ok"
            parts.append(
                f"<span class='flag {flag_cls}'>{mode}: {label}</span>")
        parts.append(f"<pre style='color:#bbb;font-size:12px'>{dsl_text}</pre>")
        parts.append("<div class='grid'>")
        for mode in ("random", "smart"):
            r = modes.get(mode)
            parts.append(f"<div class='pane'><b>{mode}</b>")
            if r is None:
                parts.append("<div class='missing'>run missing</div></div>")
                continue
            if r["status"] != "ok":
                parts.append(f"<div class='missing'>FAILED: {r.get('error')}</div></div>")
                continue
            parts.append(_img_tag(r.get("image_path")))
            parts.append("<table>")
            parts.append(f"<tr><td class='k'>final loss</td><td>{_fmt(r.get('final_loss'))}</td>"
                         f"<td class='k'>time</td><td>{_fmt(r.get('elapsed_seconds'))} s</td></tr>")
            parts.append(f"<tr><td class='k'>epochs used</td><td>{_fmt(r.get('epochs_used'), 0)}</td>"
                         f"<td class='k'>epochs to &tau;</td><td>{_fmt(r.get('epochs_to_tau'), 0)}</td></tr>")
            reasons = "; ".join(r.get("degenerate_reasons") or [])
            parts.append(f"<tr><td class='k'>auto reasons</td><td colspan='3'>{reasons or '&mdash;'}</td></tr>")
            parts.append("</table></div>")
        parts.append("</div></div>")
    parts.append("</body></html>")

    html_path = out_dir / "gallery.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Report written:\n  {csv_path}\n  {html_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Random vs Smart initialization ablation")
    parser.add_argument("--repo", type=str, default=None,
                        help="HuggingFace dataset repo (default: use built-in mock DSLs)")
    parser.add_argument("--split", type=str, default="test", help="HF dataset split")
    parser.add_argument("--field", type=str, default="output", help="HF dataset column holding the DSL")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of DSLs")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per DSL")
    parser.add_argument("--epochs", type=int, default=1000, help="Optimizer epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Optimizer learning rate")
    parser.add_argument("--dtype", type=str, default="float32", help="float32 or float64")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Success loss threshold")
    parser.add_argument("--early-stop-patience", type=int, default=150)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--early-stop-min-epochs", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=0, help="Seed offset")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker processes (0 = auto, 1 = sequential)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already present in the raw output file")
    parser.add_argument("--render-dir", type=str, default=None,
                        help="Render each final diagram to PNG under this directory")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: None)  # suppress loguru noise

    dsls = load_dsls(args)
    if not dsls:
        print("No DSLs to evaluate.")
        sys.exit(1)

    modes = ["random", "smart"]
    results: list[dict[str, Any]] = []
    start = time.perf_counter()

    # Write raw results incrementally so a crash never loses completed runs.
    raw_output = None
    if args.output:
        raw_output = Path(args.output).with_suffix(".raw.json")

    if args.resume and raw_output is not None and raw_output.exists():
        try:
            prior = json.loads(raw_output.read_text(encoding="utf-8"))
            results.extend(prior)
            logger.info(f"Resumed with {len(results)} completed runs from {raw_output}")
            print(f"Resumed: {len(results)} runs already done")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not resume from {raw_output}: {exc}")

    def _save_raw() -> None:
        if raw_output is not None:
            raw_output.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Build the full task list: (dsl, mode, seed, opts) per run.
    # Interleave modes/seeds within each DSL so any prefix of completed work
    # stays balanced between the two strategies.
    done_keys = {(r["dsl"], r["mode"], r["seed"]) for r in results}
    run_opts = vars(args)
    if args.render_dir:
        run_opts = {**run_opts, "render_dir": args.render_dir,
                    "tag": Path(args.output).stem if args.output else "ablation"}
    tasks: list[tuple[int, str, str, int, dict[str, Any]]] = []
    for idx, dsl in enumerate(dsls):
        for s in range(args.seeds):
            for mode in modes:
                t = (idx, dsl, mode, args.seed_base + s, run_opts)
                if (t[1], t[2], t[3]) in done_keys:
                    continue
                tasks.append(t)

    if args.workers == 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    else:
        workers = args.workers

    print(f"Running {len(tasks)} runs with {workers} worker(s) ...")
    start_all = time.perf_counter()

    if workers == 1:
        for i, (dsl, mode, seed, opts) in enumerate(tasks, 1):
            results.append(run_case(dsl, mode, seed, opts))
            if i % 10 == 0:
                _save_raw()
                print(f"[{mode:6s}] {i}/{len(tasks)} runs done", end="\r")
    else:
        from concurrent.futures import FIRST_COMPLETED, wait, BrokenExecutor

        os.environ["ABL_CHILD"] = "1"  # inherited by spawned workers
        remaining: list[tuple[str, str, int, dict[str, Any]]] = list(tasks)
        max_pool_attempts = 6

        for attempt in range(1, max_pool_attempts + 1):
            if not remaining:
                break
            pending: dict = {}
            try:
                with ProcessPoolExecutor(max_workers=workers,
                                         initializer=_worker_init) as executor:
                    for task in remaining:
                        pending[executor.submit(run_case, *task)] = task
                    while pending:
                        done_set, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                        for future in done_set:
                            task = pending.pop(future)
                            try:
                                results.append(future.result())
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(f"Run failed ({type(exc).__name__}): {exc}")
                        _save_raw()
                        print(f"{len(results)}/{len(tasks)} runs done", end="\r")
            except (BrokenExecutor, OSError) as exc:
                logger.warning(f"Pool attempt {attempt} broke ({exc})")
                time.sleep(3.0)
            finally:
                # Reconcile: whatever did not produce a result gets re-run,
                # regardless of how this attempt ended.
                done_keys = {(r["dsl"], r["mode"], r["seed"]) for r in results}
                remaining = [t for t in tasks
                             if (t[1], t[2], t[3]) not in done_keys]
                if remaining:
                    logger.warning(f"{len(remaining)} tasks still pending after "
                                   f"attempt {attempt}")

        # Any still-remaining tasks after all attempts: run sequentially.
        for task in remaining:
            results.append(run_case(*task))

    elapsed = time.perf_counter() - start
    print(" " * 60, end="\r")
    print(f"Total runs: {len(results)} in {elapsed:.1f}s")

    summaries = {mode: aggregate([r for r in results if r["mode"] == mode], args.tau)
                 for mode in modes}
    print("\n" + "=" * 60)
    print("Ablation summary (Random vs Smart Initialization)")
    print("=" * 60)
    for mode in modes:
        s = summaries[mode]
        print(f"\n[{mode}]")
        print(f"  Total runs        : {s['total_runs']}")
        print(f"  Success rate      : {s['success_rate_pct']:.2f}% ({s['successes']}/{s['total_runs']})")
        print(f"  Failed (crash)    : {s['failed_runs']}")
        print(f"  Degenerate        : {s['degenerate_pct']:.2f}% ({s['degenerate_runs']})")
        print(f"  Avg epochs        : {s['avg_epochs']} +/- {s['avg_epochs_std']}")
        print(f"  Avg epochs to tau : {s['avg_epochs_to_tau']} +/- {s['avg_epochs_to_tau_std']} (reached: {s['runs_reached_tau']})")
        print(f"  Avg solve time    : {s['avg_time_s']} s +/- {s['avg_time_s_std']}")
        print(f"  Final loss        : {s['final_loss_mean']} +/- {s['final_loss_std']}")

    table = latex_table(summaries, args.tau)
    print("\n" + "-" * 60)
    print("LaTeX table:")
    print(table)

    payload = {
        "args": vars(args),
        "tau": args.tau,
        "elapsed_seconds": round(elapsed, 2),
        "summaries": summaries,
        "results": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to: {out_path}")

        report_dir = Path(args.render_dir) if args.render_dir else out_path.parent / "report"
        write_report(results, report_dir)


if __name__ == "__main__":
    main()