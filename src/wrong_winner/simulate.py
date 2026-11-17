"""Simulate a team running a small eval, and score its decision against humans.

A *pair* is two systems (A, B) compared on the same items. Each item can have
several human scores and one or more judge scores, all from A's point of view
(A wins = 1, tie = 0.5, B wins = 0).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import fmean
from typing import Dict, Hashable, Iterable, List, Optional

from .stats import Estimate, cohen_kappa, decide, mean_ci, pearson_r, ppi_mean

Item = Hashable

METHODS = ("judge", "human_only", "ppi", "transport")


@dataclass
class PairData:
    model_a: str
    model_b: str
    human: Dict[Item, List[float]] = field(default_factory=dict)
    judge: Dict[Item, List[float]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.model_a} vs {self.model_b}"

    def shared_items(self) -> List[Item]:
        """Items that have both human and judge scores, in a stable order."""
        return sorted(set(self.human) & set(self.judge), key=repr)


@dataclass(frozen=True)
class Truth:
    estimate: Estimate
    winner: Optional[str]  # "A", "B", or None when too close to call


def ground_truth(pair: PairData, confidence: float = 0.95) -> Truth:
    """Human verdict on all available items (item-level mean, then mean over items)."""
    items = [fmean(scores) for scores in pair.human.values() if scores]
    est = mean_ci(items, confidence)
    return Truth(est, decide(est, "significant"))


def _one_eval(
    pair: PairData,
    pool: List[Item],
    n: int,
    k: int,
    method: str,
    rng: random.Random,
    confidence: float,
    transport_source: Optional[PairData],
) -> Estimate:
    sample = rng.choices(pool, k=n)
    judge_scores = [rng.choice(pair.judge[item]) for item in sample]

    if method == "judge":
        return mean_ci(judge_scores, confidence)

    # One random human (one annotator) per labelled item, as a real team would have.
    human_scores = [rng.choice(pair.human[item]) for item in sample[:k]]

    if method == "human_only":
        return mean_ci(human_scores, confidence)

    if method == "ppi":
        return ppi_mean(judge_scores[k:], judge_scores[:k], human_scores, confidence)

    if method == "transport":
        # Correction measured on a *different* pair, then reused here.
        assert transport_source is not None
        src_pool = transport_source.shared_items()
        src_items = rng.choices(src_pool, k=k)
        src_judge = [rng.choice(transport_source.judge[i]) for i in src_items]
        src_human = [rng.choice(transport_source.human[i]) for i in src_items]
        return ppi_mean(judge_scores, src_judge, src_human, confidence)

    raise ValueError(f"unknown method: {method!r}")


def simulate_pair(
    pair: PairData,
    n: int,
    method: str = "judge",
    k: int = 0,
    rule: str = "significant",
    reps: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    transport_source: Optional[PairData] = None,
    truth: Optional[Truth] = None,
) -> dict:
    """Run `reps` simulated evals of size n and tally decisions against the human truth.

    method:
      - "judge":      LLM judge on n items, no human labels.
      - "human_only": one human label on each of k items (set k = n for the
                      pure-sampling-noise baseline).
      - "ppi":        judge on n items, human labels on k of them, PPI correction.
      - "transport":  judge on n items, correction measured on k items of
                      `transport_source` (a different pair).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method in ("human_only", "ppi", "transport") and k < 2:
        raise ValueError(f"method {method!r} needs k >= 2 human labels")
    if method in ("human_only", "ppi") and k > n:
        raise ValueError("k cannot exceed n")
    if method == "transport" and transport_source is None:
        raise ValueError("method 'transport' needs transport_source")

    pool = pair.shared_items()
    if not pool:
        raise ValueError(f"{pair.name}: no items with both human and judge scores")
    truth = truth or ground_truth(pair, confidence)
    rng = random.Random(seed)

    tally = {"correct": 0, "wrong": 0, "overclaim": 0, "no_decision": 0}
    for _ in range(reps):
        est = _one_eval(pair, pool, n, k, method, rng, confidence, transport_source)
        d = decide(est, rule)
        if d is None:
            tally["no_decision"] += 1
        elif truth.winner is None:
            # A confident call on a pair humans cannot separate.
            tally["overclaim"] += 1
        elif d == truth.winner:
            tally["correct"] += 1
        else:
            # The opposite winner to the human verdict.
            tally["wrong"] += 1

    return {
        "pair": pair.name,
        "truth": truth.winner or "too_close",
        "human_win_rate_a": round(truth.estimate.value, 4),
        "n": n,
        "k": k,
        "method": method,
        "rule": rule,
        "reps": reps,
        "correct_rate": tally["correct"] / reps,
        "wrong_rate": tally["wrong"] / reps,
        "overclaim_rate": tally["overclaim"] / reps,
        "no_decision_rate": tally["no_decision"] / reps,
    }


def judge_full_data_verdict(pair: PairData, confidence: float = 0.95) -> Truth:
    """The judge's verdict with all items (isolates judge bias from sampling noise)."""
    items = [fmean(scores) for scores in pair.judge.values() if scores]
    est = mean_ci(items, confidence)
    return Truth(est, decide(est, "significant"))


def score_label(score: float) -> str:
    """Item-level label from a score: "A", "B" or "tie"."""
    if score > 0.5:
        return "A"
    if score < 0.5:
        return "B"
    return "tie"


def agreement(pair: PairData) -> dict:
    """Item-level judge-human agreement on shared items (RQ6).

    Uses each item's mean human score and mean judge score. Raw agreement and
    Cohen's kappa are on A/B/tie labels; Pearson r is on the scores, as many
    guides recommend.
    """
    items = pair.shared_items()
    if not items:
        raise ValueError(f"{pair.name}: no items with both human and judge scores")
    human = [fmean(pair.human[i]) for i in items]
    judge = [fmean(pair.judge[i]) for i in items]
    h_labels = [score_label(s) for s in human]
    j_labels = [score_label(s) for s in judge]
    return {
        "pair": pair.name,
        "items": len(items),
        "raw_agreement": sum(h == j for h, j in zip(h_labels, j_labels)) / len(items),
        "kappa": cohen_kappa(h_labels, j_labels),
        "pearson_r": pearson_r(human, judge),
        "human_win_rate_a": fmean(human),
        "judge_win_rate_a": fmean(judge),
    }


def run_grid(pairs: Iterable[PairData], configs: Iterable[dict], **kwargs) -> List[dict]:
    """Run simulate_pair for every (pair, config) combination."""
    pairs = list(pairs)
    configs = list(configs)
    rows = []
    for pair in pairs:
        truth = ground_truth(pair)
        for cfg in configs:
            rows.append(simulate_pair(pair, truth=truth, **cfg, **kwargs))
    return rows
