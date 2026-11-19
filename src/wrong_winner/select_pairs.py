"""Phase 2 step 1: choose LMArena model pairs by how far apart humans rate them.

Usage:
    python -m wrong_winner.select_pairs                 # fetches votes on first run (~7 MB)
    python -m wrong_winner.select_pairs --min-votes 120

Writes results/phase2/candidates.csv (every pair with enough votes),
results/phase2/pairs.csv (the chosen pairs) and results/phase2/items.csv (the
votes to judge), and prints a token estimate for judging them. Makes no LLM calls.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Dict, List, Sequence, Tuple

from .arena import DEFAULT_CACHE, Vote, fetch_votes, filter_votes, load_votes, pair_counts
from .stats import mean_ci
from .truth import bootstrap_pairs

# (group, lowest gap, highest gap, pairs wanted). Gaps in points from 50%.
GROUPS: Tuple[Tuple[str, float, float, int], ...] = (
    ("close", 0.0, 3.0, 15),
    ("small", 3.0, 7.0, 15),
    ("clear", 7.0, 100.0, 10),
)
MAX_ITEMS = 150  # votes judged per pair
TEMPLATE_TOKENS = 400  # judge instructions around the prompt and two answers


@dataclass
class Candidate:
    model_a: str
    model_b: str
    votes: int
    bt_p: float
    bt_low: float
    bt_high: float
    bt_gap_points: float
    own_p: float
    own_low: float
    own_high: float
    bt_separable: bool
    own_separable: bool
    same_side: bool  # Bradley–Terry and the pair's own votes favour the same model
    has_claude: bool
    group: str = ""


def group_of(gap_points: float, groups=GROUPS) -> str:
    for name, low, high, _ in groups:
        if low <= gap_points < high or (name == groups[-1][0] and gap_points >= low):
            return name
    return ""


def spread_pick(rows: Sequence[Candidate], k: int) -> List[Candidate]:
    """k rows evenly spaced by gap, so a group covers its whole range."""
    ordered = sorted(rows, key=lambda r: (r.bt_gap_points, r.model_a, r.model_b))
    if k >= len(ordered):
        return list(ordered)
    if k == 1:
        return [ordered[len(ordered) // 2]]
    idx = sorted({round(i * (len(ordered) - 1) / (k - 1)) for i in range(k)})
    return [ordered[i] for i in idx]


def select(candidates: Sequence[Candidate], groups=GROUPS) -> List[Candidate]:
    chosen = []
    for name, _, _, k in groups:
        in_group = [c for c in candidates if group_of(c.bt_gap_points, groups) == name]
        for c in spread_pick(in_group, k):
            c.group = name
            chosen.append(c)
    return chosen


def build_candidates(votes: Sequence[Vote], min_votes: int, reps: int, seed: int) -> List[Candidate]:
    counts = pair_counts(votes)
    pairs = sorted(p for p, n in counts.items() if n >= min_votes)
    own: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for v in votes:
        pair, score = v.canonical()
        own[pair].append(score)
    outcomes = [(v.model_a, v.model_b, v.score) for v in votes]
    out = []
    for t in bootstrap_pairs(outcomes, pairs, reps=reps, seed=seed):
        o = mean_ci(own[(t.model_a, t.model_b)])
        out.append(
            Candidate(
                model_a=t.model_a,
                model_b=t.model_b,
                votes=counts[(t.model_a, t.model_b)],
                bt_p=t.p,
                bt_low=t.low,
                bt_high=t.high,
                bt_gap_points=t.gap_points,
                own_p=o.value,
                own_low=o.low,
                own_high=o.high,
                bt_separable=t.separable,
                own_separable=o.low > 0.5 or o.high < 0.5,
                same_side=(t.p - 0.5) * (o.value - 0.5) > 0,
                has_claude="claude" in (t.model_a + t.model_b).lower(),
            )
        )
    return out


def items_for(votes: Sequence[Vote], chosen: Sequence[Candidate], seed: int) -> Dict[Tuple[str, str], List[Vote]]:
    """Up to MAX_ITEMS votes per chosen pair (a seeded random subset when there are more)."""
    by_pair: Dict[Tuple[str, str], List[Vote]] = defaultdict(list)
    for v in votes:
        by_pair[v.canonical()[0]].append(v)
    rng = random.Random(seed)
    out = {}
    for c in chosen:
        vs = sorted(by_pair[(c.model_a, c.model_b)], key=lambda v: v.id)
        out[(c.model_a, c.model_b)] = vs if len(vs) <= MAX_ITEMS else rng.sample(vs, MAX_ITEMS)
    return out


ITEM_FIELDS = ("id", "group", "model_a", "model_b", "human_score", "tokens")


def write_items(path: Path, items: Dict[Tuple[str, str], List[Vote]], chosen: Sequence[Candidate]) -> None:
    """One row per vote to judge. model_a/model_b are in the order the voter saw them;
    human_score is from model_a's view; tokens is the dataset's count plus the judge instructions."""
    group = {(c.model_a, c.model_b): c.group for c in chosen}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ITEM_FIELDS)
        for pair, vs in items.items():
            for v in sorted(vs, key=lambda v: v.id):
                tokens = v.user_tokens + v.a_tokens + v.b_tokens + TEMPLATE_TOKENS
                writer.writerow([v.id, group[pair], v.model_a, v.model_b, v.score, tokens])


def _write(path: Path, rows: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for r in rows:
            d = asdict(r)
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-votes", type=int, default=100)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/phase2"))
    args = parser.parse_args()

    if not DEFAULT_CACHE.exists():
        print("Fetching the needed LMArena columns (about 7 MB)...")
        fetch_votes()
    all_votes = load_votes()
    votes = filter_votes(all_votes)
    print(f"Votes: {len(all_votes)} in total, {len(votes)} English single-turn")

    candidates = build_candidates(votes, args.min_votes, args.reps, args.seed)
    print(f"Pairs with >= {args.min_votes} votes: {len(candidates)}")
    for name, low, high, k in GROUPS:
        n = sum(group_of(c.bt_gap_points) == name for c in candidates)
        print(f"  {name:5s} ({low:g}-{high:g} pts): {n} available, {min(n, k)} chosen")
    chosen = select(candidates)
    _write(args.out / "candidates.csv", sorted(candidates, key=lambda c: c.bt_gap_points))
    _write(args.out / "pairs.csv", chosen)

    items = items_for(votes, chosen, args.seed)
    write_items(args.out / "items.csv", items, chosen)
    calls_per_judge = 2 * sum(len(v) for v in items.values())  # both answer orders
    tokens = [
        v.user_tokens + v.a_tokens + v.b_tokens + TEMPLATE_TOKENS
        for vs in items.values()
        for v in vs
    ]
    print(f"\nChosen: {len(chosen)} pairs, {sum(len(v) for v in items.values())} items, "
          f"{calls_per_judge} judge calls per judge (both orders)")
    print(f"Input tokens per call (dataset's own counts + {TEMPLATE_TOKENS} for instructions): "
          f"median {median(tokens):.0f}, mean {fmean(tokens):.0f}, max {max(tokens)}")
    print(f"Total input tokens per judge: {2 * sum(tokens) / 1e6:.1f}M")
    print(f"Pairs with a Claude model: {sum(c.has_claude for c in chosen)}")
    disagree = [c for c in chosen if not c.same_side]
    print(f"Pairs where the pair's own votes favour the other model: {len(disagree)}")
    print(f"\nWrote {args.out / 'candidates.csv'}, {args.out / 'pairs.csv'} and {args.out / 'items.csv'}")


if __name__ == "__main__":
    main()
