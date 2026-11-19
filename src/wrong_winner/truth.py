"""Bradley–Terry ground truth for model pairs (standard library only).

Each pair in the arena has only 100–200 votes, so its own win rate has wide
error bars. Bradley–Terry pools every vote: each model gets a strength s, and
P(A beats B) = s_A / (s_A + s_B). A pair then borrows evidence from games both
models played against others. This assumes strength is transitive; the pair's
own votes are kept as a check.

Ties and "both bad" count as half a win for each side, as in Phase 1.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Outcome = Tuple[str, str, float]  # (model_a, model_b, score from model_a's view)
Pair = Tuple[str, str]


def _aggregate(outcomes: Iterable[Outcome]) -> Tuple[Dict[str, float], Dict[Pair, float]]:
    wins: Dict[str, float] = defaultdict(float)
    games: Dict[Pair, float] = defaultdict(float)
    for a, b, score in outcomes:
        if a == b:
            continue
        wins[a] += score
        wins[b] += 1.0 - score
        games[(a, b) if a < b else (b, a)] += 1.0
    return wins, games


def fit_bradley_terry(
    outcomes: Iterable[Outcome],
    prior: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 10_000,
    start: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Fit strengths with the MM algorithm (Hunter 2004).

    `prior` adds that many tied games between each model and a reference model of
    strength 1, so a model that never lost (or never won) still gets a finite
    strength. With thousands of votes its effect is negligible.
    Strengths are scaled so their geometric mean is 1.
    """
    wins, games = _aggregate(outcomes)
    models = sorted(wins)
    if len(models) < 2:
        raise ValueError("need at least two models")
    opponents: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (a, b), n in games.items():
        opponents[a].append((b, n))
        opponents[b].append((a, n))

    s = {m: (start or {}).get(m, 1.0) for m in models}
    for _ in range(max_iter):
        new = {}
        for m in models:
            denom = sum(n / (s[m] + s[o]) for o, n in opponents[m])
            denom += prior / (s[m] + 1.0)
            new[m] = (wins[m] + prior / 2) / denom
        log_mean = math.fsum(math.log(v) for v in new.values()) / len(new)
        scale = math.exp(log_mean)
        new = {m: v / scale for m, v in new.items()}
        change = max(abs(math.log(new[m] / s[m])) for m in models)
        s = new
        if change < tol:
            return s
    raise RuntimeError(f"Bradley–Terry did not converge in {max_iter} iterations")


def win_prob(strengths: Dict[str, float], a: str, b: str) -> float:
    """Predicted probability that a beats b (ties count half)."""
    return strengths[a] / (strengths[a] + strengths[b])


@dataclass(frozen=True)
class PairTruth:
    model_a: str
    model_b: str
    p: float  # predicted P(model_a beats model_b)
    low: float
    high: float

    @property
    def gap_points(self) -> float:
        """Distance from a coin flip, in percentage points."""
        return abs(self.p - 0.5) * 100

    @property
    def separable(self) -> bool:
        """True when the interval excludes 50%."""
        return self.low > 0.5 or self.high < 0.5


def bootstrap_pairs(
    outcomes: Sequence[Outcome],
    pairs: Sequence[Pair],
    reps: int = 200,
    seed: int = 0,
    confidence: float = 0.95,
    prior: float = 1.0,
) -> List[PairTruth]:
    """Point estimate and percentile interval of P(a beats b) for each pair.

    Resamples votes with replacement and refits. Warm-starts from the full fit,
    so each refit takes few iterations.
    """
    full = fit_bradley_terry(outcomes, prior=prior)
    rng = random.Random(seed)
    draws: Dict[Pair, List[float]] = {p: [] for p in pairs}
    for _ in range(reps):
        sample = rng.choices(outcomes, k=len(outcomes))
        s = fit_bradley_terry(sample, prior=prior, start=full, tol=1e-7)
        for a, b in pairs:
            draws[(a, b)].append(win_prob(s, a, b))
    alpha = (1 - confidence) / 2
    out = []
    for a, b in pairs:
        xs = sorted(draws[(a, b)])
        lo = xs[int(math.floor(alpha * (len(xs) - 1)))]
        hi = xs[int(math.ceil((1 - alpha) * (len(xs) - 1)))]
        out.append(PairTruth(a, b, win_prob(full, a, b), lo, hi))
    return out
