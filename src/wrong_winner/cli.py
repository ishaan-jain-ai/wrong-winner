"""Command line: `wrong-winner check` and `wrong-winner pick`.

    wrong-winner check verdicts.csv [--labels labels.csv] [--a current --b candidate]
    wrong-winner pick verdicts.csv 25 -o labels.csv

Exit codes for `check`: 0 = the gate passed, 1 = it didn't, 2 = bad input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

from .checker import CLEAR_GAP, GATES, START_LABELS, InputError, check, exit_code, pick, render

CHECK_HELP = """\
Is the judge's verdict backed by evidence? Reads a CSV with one row per test
item and the columns id, judge_a_first, judge_b_first and (optional) human.
A verdict names the winner (a, b, tie, or the systems' names from --a/--b) or,
in the judge columns, its position (first, second). Give both judge columns
when you can: an item whose verdict flips with the answer order counts as a tie.
"""

PICK_HELP = """\
Pick items at random for you to label. Writes their rows without the judge's
verdicts and with an empty human column: fill it in (a, b or tie), then pass the
file to `check --labels`.
"""

GATE_HELP = (
    "better (default): exit 0 only if B is shown better; not-worse: exit 1 only if A is "
    "shown better; none: always exit 0"
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wrong-winner",
        description="Check whether an LLM judge's A/B verdict is backed by evidence.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="check a judge's A/B verdict", description=CHECK_HELP)
    c.add_argument("verdicts", type=Path, help="CSV of judge verdicts, one row per test item")
    c.add_argument("--labels", type=Path, help="CSV with columns id and human (from `pick`)")
    c.add_argument("--a", dest="a_name", default="A", metavar="NAME", help="name of system A (the current one)")
    c.add_argument("--b", dest="b_name", default="B", metavar="NAME", help="name of system B (the candidate)")
    c.add_argument("--gate", choices=GATES, default="better", help=GATE_HELP)
    c.add_argument("--confidence", type=float, default=0.95, help="interval confidence (default 0.95)")
    c.add_argument(
        "--clear-gap", type=float, default=CLEAR_GAP * 100, metavar="POINTS",
        help=f"how far from a tie, in points, a judge-only result must be to count (default {CLEAR_GAP * 100:.0f})",
    )
    c.add_argument("--json", action="store_true", help="print the report as JSON")

    k = sub.add_parser("pick", help="pick items to label", description=PICK_HELP)
    k.add_argument("verdicts", type=Path, help="the CSV you will pass to `check`")
    k.add_argument("n", type=int, nargs="?", default=START_LABELS, help=f"how many (default {START_LABELS})")
    k.add_argument("-o", "--out", type=Path, help="file to write (default: print); never overwritten")
    k.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    k.add_argument("--a", dest="a_name", default="A", metavar="NAME", help="name of system A")
    k.add_argument("--b", dest="b_name", default="B", metavar="NAME", help="name of system B")
    return p


def run_check(args: argparse.Namespace) -> int:
    report = check(
        args.verdicts, args.labels, args.a_name, args.b_name,
        confidence=args.confidence, clear_gap=args.clear_gap / 100,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render(report, source=str(args.verdicts)))
    return exit_code(report, args.gate)


def run_pick(args: argparse.Namespace) -> int:
    if args.out and args.out.exists():
        raise InputError(f"{args.out} already exists; not overwriting it (it may hold your labels)")
    header, rows = pick(args.verdicts, args.n, args.seed, args.a_name, args.b_name)
    out = args.out.open("x", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()
    if args.out:
        print(f"Wrote {len(rows)} items to {args.out}. Fill in the human column (a, b or tie).", file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run_check(args) if args.command == "check" else run_pick(args)
    except (InputError, ValueError) as e:
        print(f"wrong-winner: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
