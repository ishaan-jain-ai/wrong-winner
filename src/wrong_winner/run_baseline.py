"""Phase 1: RQ1-RQ5 on MT-Bench using the GPT-4 judgments bundled with the dataset.

Usage:
    python -m wrong_winner.run_baseline            # full run (2,000 reps)
    python -m wrong_winner.run_baseline --reps 200 # quick check

Writes results/baseline.csv (decisions) and results/agreement.csv (RQ6), and
prints a summary for pairs with a clear human winner.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

from .data import load_mtbench, summarise
from .simulate import (
    agreement,
    ground_truth,
    judge_full_data_verdict,
    run_grid,
    simulate_pair,
)

N_GRID = (25, 50, 100, 200)
K_GRID = (10, 25, 50)


def configs():
    for n in N_GRID:
        for rule in ("naive", "significant"):
            # RQ1/RQ3: judge only
            yield {"n": n, "method": "judge", "rule": rule}
            # RQ2: humans on the same n items = pure sampling noise
            yield {"n": n, "method": "human_only", "k": n, "rule": rule}
        # RQ4: judge on n items + k human labels
        for k in K_GRID:
            if k < n:
                yield {"n": n, "method": "ppi", "k": k, "rule": "significant"}
                yield {"n": n, "method": "human_only", "k": k, "rule": "significant"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/baseline.csv"))
    args = parser.parse_args()

    pairs = load_mtbench()
    print("Loaded pairs:")
    for row in summarise(pairs):
        print("  ", row)

    print("\nFull-data verdicts (human vs judge). Amplification = judge gap / human gap from 0.5:")
    amplifications = []
    for p in pairs:
        h, j = ground_truth(p), judge_full_data_verdict(p)
        h_gap, j_gap = h.estimate.value - 0.5, j.estimate.value - 0.5
        same_side = h_gap * j_gap > 0
        amp = abs(j_gap) / abs(h_gap) if h_gap else float("inf")
        if h.winner is not None:
            amplifications.append(amp)
        print(
            f"  {p.name:45s} human={h.winner or 'too_close':9s} ({h.estimate.value:.3f})"
            f"  judge={j.winner or 'too_close':9s} ({j.estimate.value:.3f})"
            f"  amp={amp:5.2f}{'' if same_side else '  OPPOSITE SIDE'}"
        )
    if amplifications:
        amplified = sum(a > 1 for a in amplifications)
        print(
            f"  Clear-winner pairs where the judge's gap is larger than humans': "
            f"{amplified}/{len(amplifications)} (median {median(amplifications):.2f}x)"
        )

    rows = run_grid(pairs, configs(), reps=args.reps, seed=args.seed)

    # RQ5: reuse a correction measured on another pair (each pair -> next pair).
    for i, target in enumerate(pairs):
        source = pairs[(i + 1) % len(pairs)]
        for n in (100, 200):
            for k in (25, 50):
                row = simulate_pair(
                    target, n=n, k=k, method="transport", transport_source=source,
                    reps=args.reps, seed=args.seed,
                )
                row["method"] = f"transport_from:{source.name}"
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # RQ6: does item-level agreement predict decision quality?
    # A "bad call" is either the opposite winner or an overclaim on a too-close pair.
    judge_bad = {
        (r["pair"], r["n"]): r["wrong_rate"] + r["overclaim_rate"]
        for r in rows
        if r["method"] == "judge" and r["rule"] == "significant"
    }
    agree_rows = []
    for p in pairs:
        a = agreement(p)
        a["truth"] = ground_truth(p).winner or "too_close"
        for n in N_GRID:
            a[f"judge_bad_call_rate_n{n}"] = judge_bad[(p.name, n)]
        agree_rows.append(a)
    agree_path = args.out.with_name("agreement.csv")
    with agree_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agree_rows[0]))
        writer.writeheader()
        writer.writerows(agree_rows)

    print("\nRQ6 – agreement vs bad-call rate (judge only, significance rule):")
    print(f"  {'pair':45s} {'truth':9s} {'agree':>6s} {'kappa':>6s} {'r':>6s}  bad@n=50  bad@n=200")
    for a in sorted(agree_rows, key=lambda a: a["kappa"]):
        print(
            f"  {a['pair']:45s} {a['truth']:9s} {a['raw_agreement']:6.2f} "
            f"{a['kappa']:6.2f} {a['pearson_r']:6.2f}  "
            f"{a['judge_bad_call_rate_n50']:8.1%}  {a['judge_bad_call_rate_n200']:9.1%}"
        )
    print(f"Wrote {len(agree_rows)} rows to {agree_path}")

    # Too-close pairs: how often does each method make a confident call humans can't back?
    close = [r for r in rows if r["truth"] == "too_close" and r["rule"] == "significant"]
    if close:
        print("\nToo-close pairs (humans can't separate them), significance rule:")
        print(f"  {'pair':35s} {'method':11s} {'n':>4s} {'k':>4s}  overclaim  no-decision")
        for r in sorted(close, key=lambda r: (r["pair"], r["method"], r["n"], r["k"])):
            method = "transport" if r["method"].startswith("transport") else r["method"]
            print(
                f"  {r['pair']:35s} {method:11s} {r['n']:4d} {r['k']:4d}  "
                f"{r['overclaim_rate']:9.1%}  {r['no_decision_rate']:11.1%}"
            )

    # Summary over pairs with a clear human winner.
    groups = defaultdict(list)
    for r in rows:
        if r["truth"] == "too_close":
            continue
        method = "transport" if r["method"].startswith("transport") else r["method"]
        groups[(method, r["rule"], r["n"], r["k"])].append(r)

    print("\nClear-winner pairs, averaged:")
    print(f"  {'method':11s} {'rule':12s} {'n':>4s} {'k':>4s}  wrong  correct  no-decision")
    for (method, rule, n, k), rs in sorted(groups.items()):
        print(
            f"  {method:11s} {rule:12s} {n:4d} {k:4d}  "
            f"{fmean(r['wrong_rate'] for r in rs):5.1%}  "
            f"{fmean(r['correct_rate'] for r in rs):7.1%}  "
            f"{fmean(r['no_decision_rate'] for r in rs):11.1%}"
        )


if __name__ == "__main__":
    main()
