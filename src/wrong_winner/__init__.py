"""wrong-winner: how often does a small LLM-judged eval pick the wrong winner?"""

from .checker import InputError, Item, Report, check, pick, read_items, read_labels, render
from .simulate import PairData, agreement, ground_truth, run_grid, simulate_pair
from .stats import (
    CorrectionRefused,
    Estimate,
    JudgeError,
    cohen_kappa,
    decide,
    estimate_judge_error,
    mean_ci,
    pearson_r,
    ppi_mean,
    rogan_gladen,
    wilson_interval,
)

__all__ = [
    "CorrectionRefused",
    "Estimate",
    "InputError",
    "Item",
    "JudgeError",
    "PairData",
    "Report",
    "agreement",
    "check",
    "cohen_kappa",
    "decide",
    "estimate_judge_error",
    "ground_truth",
    "mean_ci",
    "pearson_r",
    "pick",
    "ppi_mean",
    "read_items",
    "read_labels",
    "render",
    "rogan_gladen",
    "run_grid",
    "simulate_pair",
    "wilson_interval",
]

__version__ = "0.1.0"
