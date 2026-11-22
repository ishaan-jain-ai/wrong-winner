"""Example files for `wrong-winner check`, and how the check does on Phase 2.

Builds verdict CSVs from the Phase 2 judgments (every answer order a judge saw),
with real LMArena votes as the human labels, then:

1. writes examples/ for one judge (Gemma 4 31B):
   - close_pair.csv / close_pair_labels.csv: ChatGPT-4o (A) vs o3 (B). People see
     a tie; all four judges call o3.
   - clear_pair.csv: Claude 3.7 Sonnet (A) vs Gemini 2.5 Pro (B). People prefer
     Gemini 2.5 Pro.
   The labels are 25 items chosen by `pick` (seed 0), filled in with each item's vote.
2. prints the check's result for every judge and pair, judge alone and with 25
   random labels (200 draws), against the Bradley-Terry truth.

The judge-alone rule (30 points from a tie) was set from these same pairs, so
step 2 is a sanity check, not an independent test.

Run from the repo root:
    PYTHONPATH=src python scripts/make_check_examples.py
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from wrong_winner.checker import Item, check, pick  # noqa: E402
from wrong_winner.figures import short_name  # noqa: E402
from wrong_winner.judge import JUDGMENTS_DIR, read_plan  # noqa: E402
from wrong_winner.run_phase2 import JUDGES, read_pairs, read_verdicts  # noqa: E402

OUT = Path("examples")
EXAMPLE_JUDGE = "gemma-4-31b"
CLOSE = ("chatgpt-4o-latest-20250326", "o3-2025-04-16")
CLEAR = ("claude-3-7-sonnet-20250219", "gemini-2.5-pro")
K, DRAWS = 25, 200
LETTER = {0.0: "a", 0.5: "tie", 1.0: "b"}


def rows_for(pair, plan, verdicts):
    """One row per item of the pair: id, judge_a_first, judge_b_first, human (pair A = model_a)."""
    rows = []
    for item, orders in plan:
        if item.pair != pair.key:
            continue
        same = item.model_a == pair.model_a  # the item's own model_a is the pair's A
        row = {"id": item.id, "judge_a_first": "", "judge_b_first": "", "human": ""}
        for order in orders:
            verdict = verdicts.get((item.id, order))
            if verdict is None:
                continue
            item_a_first = order == "ab"
            pair_a_first = item_a_first == same
            if verdict == "tie":
                winner = "tie"
            else:
                first_won = verdict == "A"
                winner = "a" if first_won == pair_a_first else "b"
            row["judge_a_first" if pair_a_first else "judge_b_first"] = winner
        human_for_a = item.human_score if same else 1.0 - item.human_score
        row["human"] = LETTER[1.0 - human_for_a]  # scores are from B's view
        rows.append(row)
    return rows


def to_items(rows, labelled=()):
    score = {"a": 0.0, "tie": 0.5, "b": 1.0, "": None}
    return [
        Item(
            r["id"], score[r["judge_a_first"]], score[r["judge_b_first"]],
            score[r["human"]] if r["id"] in labelled else None,
        )
        for r in rows
    ]


def write(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truth(pair):
    """The Bradley-Terry winner as a result name, or None for a tie."""
    winner = pair.bt_winner
    return {"A": "a_better", "B": "b_better"}.get(winner)


def make_examples(pairs, plan):
    judge = next(j for j in JUDGES if j.key == EXAMPLE_JUDGE)
    verdicts = read_verdicts(JUDGMENTS_DIR / judge.file)
    OUT.mkdir(exist_ok=True)
    for name, models in (("close_pair", CLOSE), ("clear_pair", CLEAR)):
        pair = pairs[tuple(sorted(models))]
        assert (pair.model_a, pair.model_b) == models, "pair orientation changed"
        rows = rows_for(pair, plan, verdicts)
        path = OUT / f"{name}.csv"
        write(path, ["id", "judge_a_first", "judge_b_first"], rows)
        print(f"wrote {path}: {len(rows)} items, {short_name(pair.model_a)} (A) vs {short_name(pair.model_b)} (B)")
        if name == "close_pair":
            header, picked = pick(path, K)
            votes = {r["id"]: r["human"] for r in rows}
            for r in picked:
                r["human"] = votes[r["id"]]
            write(OUT / "close_pair_labels.csv", header, picked)
            print(f"wrote {OUT / 'close_pair_labels.csv'}: {len(picked)} labels from the Arena votes")


def sweep(pairs, plan):
    judge_alone = defaultdict(Counter)
    with_labels = defaultdict(Counter)
    for judge in JUDGES:
        path = JUDGMENTS_DIR / judge.file
        if not path.exists():
            continue
        verdicts = read_verdicts(path)
        for pair in pairs.values():
            rows = rows_for(pair, plan, verdicts)
            want = truth(pair)
            kind = (pair.group, "people see a tie" if want is None else "people pick a winner")

            def grade(result):
                if result in ("a_better", "b_better"):
                    return "correct winner" if result == want else ("invented winner" if want is None else "wrong winner")
                return result

            judge_alone[kind][grade(check(to_items(rows)).result)] += 1
            rng = random.Random(f"{judge.key}:{pair.name}")
            ids = [r["id"] for r in rows if r["judge_a_first"] or r["judge_b_first"]]
            for _ in range(DRAWS):
                labelled = set(rng.sample(ids, min(K, len(ids))))
                with_labels[kind][grade(check(to_items(rows, labelled)).result)] += 1

    order = ("correct winner", "invented winner", "wrong winner", "not_proven", "no_difference", "too_few_items")
    for title, table, unit in (("Judge alone", judge_alone, "judge x pair results"), (f"With {K} random labels", with_labels, f"checks ({DRAWS} draws each)")):
        print(f"\n{title}")
        for kind in sorted(table):
            total = sum(table[kind].values())
            parts = ", ".join(f"{k} {table[kind][k] / total:.0%}" for k in order if table[kind][k])
            print(f"  {kind[0]} pairs, {kind[1]} ({total} {unit}): {parts}")


def main():
    pairs = read_pairs()
    plan = read_plan()
    make_examples(pairs, plan)
    sweep(pairs, plan)


if __name__ == "__main__":
    main()
