"""Small, dependency-free statistics for eval decisions.

Everything here works on plain Python lists so the core of the study runs
without numpy. Scores are floats in [0, 1]; for pairwise comparisons a score
is from system A's point of view (A wins = 1, tie = 0.5, B wins = 0).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist, StatisticsError, correlation, fmean
from typing import Hashable, Optional, Sequence


def z_value(confidence: float = 0.95) -> float:
    """Two-sided normal critical value, e.g. 1.96 for 95%."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return h


def _beta_inc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution."""
    tail = 0.5 * _beta_inc(df / 2, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


@lru_cache(maxsize=None)
def t_value(confidence: float = 0.95, df: int = 1) -> float:
    """Two-sided Student's t critical value, e.g. 2.262 for 95% with 9 degrees of freedom."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if df < 1:
        raise ValueError("df must be at least 1")
    target = 0.5 + confidence / 2
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < target:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < target else (lo, mid)
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


def critical_value(confidence: float, n: int, t: bool) -> float:
    """t value with n - 1 degrees of freedom when `t` is set (and n >= 2), else z."""
    return t_value(confidence, n - 1) if t and n >= 2 else z_value(confidence)


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    value: float
    low: float
    high: float

    def excludes(self, x: float) -> bool:
        return self.low > x or self.high < x


def _sample_variance(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = fmean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def mean_ci(scores: Sequence[float], confidence: float = 0.95, t: bool = False) -> Estimate:
    """Mean with a CLT interval (Miller 2024, 'Adding Error Bars to Evals').

    Works for binary scores and for pairwise scores with ties (0.5). With `t`,
    the critical value comes from Student's t with n - 1 degrees of freedom,
    which is wider for small n (Phase 1 found the normal one slightly too narrow at n = 10).
    """
    n = len(scores)
    if n == 0:
        raise ValueError("need at least one score")
    m = fmean(scores)
    half = critical_value(confidence, n, t) * math.sqrt(_sample_variance(scores) / n)
    return Estimate(m, m - half, m + half)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Estimate:
    """Wilson score interval for a binomial pass rate.

    Better behaved than the plain CLT interval for small n or rates near 0/1.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = z_value(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Estimate(p, max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Agreement (RQ6)
# ---------------------------------------------------------------------------


def cohen_kappa(labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]) -> float:
    """Chance-corrected agreement between two raters on categorical labels.

    Returns 1.0 for perfect agreement and about 0 for chance-level agreement.
    When both raters use a single identical label throughout (expected
    agreement is 1), kappa is undefined; we return 1.0 if they also agree
    perfectly and nan otherwise.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("need at least one label")
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    expected = sum(count_a[c] * count_b[c] for c in count_a) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else math.nan
    return (observed - expected) / (1 - expected)


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; nan when either input is constant."""
    if len(xs) != len(ys):
        raise ValueError("inputs must be the same length")
    try:
        return correlation(xs, ys)
    except StatisticsError:
        return math.nan


# ---------------------------------------------------------------------------
# Judge error correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeError:
    """How a binary judge behaves relative to human labels."""

    sensitivity: float  # P(judge says pass | human says pass)
    specificity: float  # P(judge says fail | human says fail)

    @property
    def youden(self) -> float:
        """Sensitivity + specificity - 1. 0 means the judge is no better than chance."""
        return self.sensitivity + self.specificity - 1


def estimate_judge_error(
    judge_labels: Sequence[bool], human_labels: Sequence[bool]
) -> JudgeError:
    """Estimate sensitivity and specificity from items labelled by both."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")
    pos = [j for j, h in zip(judge_labels, human_labels) if h]
    neg = [j for j, h in zip(judge_labels, human_labels) if not h]
    if not pos or not neg:
        raise ValueError("need at least one human pass and one human fail")
    return JudgeError(
        sensitivity=sum(pos) / len(pos),
        specificity=sum(1 for j in neg if not j) / len(neg),
    )


class CorrectionRefused(ValueError):
    """Raised when a correction would be unstable or meaningless."""


def rogan_gladen(
    observed_rate: float, error: JudgeError, min_youden: float = 0.1
) -> float:
    """Correct an observed pass rate for a judge's known error rates.

    true_rate = (observed + specificity - 1) / (sensitivity + specificity - 1)

    Refuses when the judge is too close to chance, because the denominator
    then blows up small errors. The result is clipped to [0, 1].

    Caveat (Li 2026, 'Frozen Judges, Moving Agents'): this is only valid if
    the judge's error rates on the data you correct match the rates you
    measured. They often do not transfer across system versions.
    """
    if error.youden <= min_youden:
        raise CorrectionRefused(
            f"judge is too close to chance (Youden index {error.youden:.3f} <= {min_youden})"
        )
    corrected = (observed_rate + error.specificity - 1) / error.youden
    return min(1.0, max(0.0, corrected))


def ppi_mean(
    judge_unlabelled: Sequence[float],
    judge_labelled: Sequence[float],
    human_labelled: Sequence[float],
    confidence: float = 0.95,
    t: bool = False,
) -> Estimate:
    """Prediction-powered estimate of the human mean (Angelopoulos et al., 2023).

    estimate = mean(judge on unlabelled) + mean(human - judge on labelled)

    The second term (the "rectifier") removes the judge's average bias using
    the items that humans also labelled. The interval adds both variances.
    With `t`, the critical value is Student's t with k - 1 degrees of freedom
    (k = human labels), a conservative choice since the rectifier's variance
    dominates when k is small.
    """
    if len(judge_labelled) != len(human_labelled):
        raise ValueError("judge_labelled and human_labelled must be the same length")
    n_lab = len(human_labelled)
    if n_lab == 0:
        raise ValueError("need at least one human-labelled item")
    gaps = [h - j for h, j in zip(human_labelled, judge_labelled)]
    rectifier = fmean(gaps)
    var = _sample_variance(gaps) / n_lab
    if judge_unlabelled:
        base = fmean(judge_unlabelled)
        var += _sample_variance(judge_unlabelled) / len(judge_unlabelled)
    else:
        base = fmean(judge_labelled)
        var += _sample_variance(judge_labelled) / n_lab
    value = base + rectifier
    half = critical_value(confidence, n_lab, t) * math.sqrt(var)
    return Estimate(value, value - half, value + half)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def decide(
    estimate: Estimate, rule: str = "significant", tie: float = 0.5
) -> Optional[str]:
    """Turn an estimate of A's win rate into a ship decision.

    Returns "A", "B" or None (no decision).

    - "naive": whichever side is ahead, however slightly.
    - "significant": only if the interval excludes a tie.
    """
    if rule == "naive":
        if estimate.value > tie:
            return "A"
        if estimate.value < tie:
            return "B"
        return None
    if rule == "significant":
        if estimate.low > tie:
            return "A"
        if estimate.high < tie:
            return "B"
        return None
    raise ValueError(f"unknown rule: {rule!r}")
