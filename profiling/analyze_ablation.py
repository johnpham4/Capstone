"""Analyze ablation results JSON."""
import json
import math
import statistics
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "profiling/ablation_mock_full.json"
d = json.load(open(path))
res = d["results"]
tau = d.get("tau", 0.5)

for mode in ["random", "smart"]:
    rs = [r for r in res if r["mode"] == mode]
    succ = [
        r for r in rs
        if r["status"] == "ok"
        and r.get("final_loss") is not None
        and math.isfinite(r["final_loss"])
        and r["final_loss"] <= tau
        and not r.get("degenerate")
        and r.get("point_count", 0) > 0
    ]
    ep = [r["epochs_used"] for r in succ if r.get("epochs_used")]
    sloss = [r["final_loss"] for r in succ]
    print(f"== {mode} ==")
    print(f"  successful runs: {len(succ)}/{len(rs)}")
    if len(ep) > 1:
        print(f"  epochs among successes: mean={statistics.mean(ep):.1f} std={statistics.stdev(ep):.1f}")
    else:
        print(f"  epochs: {ep}")
    print(f"  loss among successes: mean={statistics.mean(sloss):.4f}")

    degen = [r for r in rs if r.get("degenerate")]
    c = Counter()
    for r in degen:
        for reason in r["degenerate_reasons"]:
            c[reason.split(" ")[0]] += 1
    print(f"  degenerate runs: {len(degen)}  reasons: {dict(c)}")

    over = [
        r for r in rs
        if r["status"] == "ok"
        and r.get("final_loss") is not None
        and math.isfinite(r["final_loss"])
        and r["final_loss"] > tau
    ]
    print(f"  non-converged (loss>{tau}): {len(over)}")

    # Per-DSL: how many seeds succeeded per DSL
    per_dsl = Counter()
    for r in succ:
        per_dsl[r["dsl"].splitlines()[0][:40]] += 1
    print("  per-DSL success (of 5 seeds):")
    for dsl, n in per_dsl.items():
        print(f"    {n}/5  {dsl}")
    print()