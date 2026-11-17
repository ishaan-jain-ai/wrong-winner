"""Load public human-judgment datasets into PairData.

MT-Bench human judgments (lmsys/mt_bench_human_judgments, CC-BY-4.0):
  - split "human":     ~3.3k expert pairwise judgments
  - split "gpt4_pair": ~2.4k GPT-4 pairwise judgments of the same model pairs

Expected columns (verify on first download; see `check_schema`):
  question_id, model_a, model_b, winner, judge, turn
where winner is "model_a", "model_b", or a value starting with "tie".
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

from .simulate import PairData

MTBENCH_REPO = "lmsys/mt_bench_human_judgments"
REQUIRED_COLUMNS = ("question_id", "model_a", "model_b", "winner", "turn")


def winner_to_score(winner: str) -> float:
    """Score from model_a's point of view."""
    w = str(winner).strip().lower()
    if w == "model_a":
        return 1.0
    if w == "model_b":
        return 0.0
    if w.startswith("tie"):
        return 0.5
    raise ValueError(f"unrecognised winner value: {winner!r}")


def build_pairs(
    human_rows: Iterable[Mapping], judge_rows: Iterable[Mapping]
) -> List[PairData]:
    """Group rows into canonical (alphabetical) model pairs.

    Rows where the dataset lists the models the other way round have their
    score flipped, so every score is from the alphabetically-first model's view.
    """
    pairs: Dict[Tuple[str, str], PairData] = {}

    def add(rows: Iterable[Mapping], field: str) -> None:
        for row in rows:
            a, b = row["model_a"], row["model_b"]
            score = winner_to_score(row["winner"])
            if a > b:
                a, b = b, a
                score = 1.0 - score
            pair = pairs.setdefault((a, b), PairData(a, b))
            item = (row["question_id"], int(row["turn"]))
            getattr(pair, field).setdefault(item, []).append(score)

    add(human_rows, "human")
    add(judge_rows, "judge")
    return [pairs[key] for key in sorted(pairs)]


def check_schema(rows: List[Mapping]) -> None:
    if not rows:
        raise ValueError("no rows loaded")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"dataset is missing expected columns {missing}; found {sorted(rows[0])}"
        )
    winners = sorted({str(r["winner"]) for r in rows})
    for w in winners:
        winner_to_score(w)  # raises on unexpected values


def load_mtbench(judge_split: str = "gpt4_pair") -> List[PairData]:
    """Download (once, cached by Hugging Face) and build MT-Bench pairs.

    Needs the optional dependency: pip install "wrong-winner[data]"
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            'Loading MT-Bench needs the "datasets" package: pip install -e ".[data]"'
        ) from exc

    ds = load_dataset(MTBENCH_REPO)
    keep = list(REQUIRED_COLUMNS)
    human = [{c: r[c] for c in keep} for r in ds["human"]]
    judge = [{c: r[c] for c in keep} for r in ds[judge_split]]
    check_schema(human)
    check_schema(judge)
    return build_pairs(human, judge)


def summarise(pairs: List[PairData]) -> List[dict]:
    """Per-pair counts, for a quick sanity check after loading."""
    out = []
    for p in pairs:
        out.append(
            {
                "pair": p.name,
                "human_items": len(p.human),
                "human_judgments": sum(len(v) for v in p.human.values()),
                "judge_items": len(p.judge),
                "shared_items": len(p.shared_items()),
            }
        )
    return out
