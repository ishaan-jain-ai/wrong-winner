"""Phase 2: score the judges' verdicts on LMArena pairs against the human truth.

Usage:
    python -m wrong_winner.run_phase2              # full run (2,000 reps)
    python -m wrong_winner.run_phase2 --reps 200   # quick check

Reads results/phase2/pairs.csv, plan.csv and judgments/*.jsonl, and writes to
results/phase2/analysis/:

- verdicts.csv  per judge and pair: the judge's win rate on all its items, its
                call, and whether humans back it (main truth: Bradley-Terry;
                check: the pair's own votes)
- grid.csv      simulated evals (judge only, humans only, PPI, PPI with a
                correction from another pair) per judge, pair, n and k
- position.csv  RQ9: how often each judge changes its verdict when the answers swap
- family.csv    RQ7: does a judge favour a model from its own family
- summary.csv   headline numbers per judge (RQ8 reads down the size ladder)

Each item is scored once, in its planned answer order (a fixed coin flip), as a
team running one call per item would. The 100 items shown both ways feed RQ9 only.
Every interval here uses Student's t; Phase 1 used the normal one.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .judge import JUDGMENTS_DIR, PLAN_CSV, Item, load_done, orders_for, read_plan, verdict_score
from .simulate import PairData, agreement, score_label, simulate_decisions, tally
from .stats import Estimate, _sample_variance, cohen_kappa, decide, mean_ci, wilson_interval

PAIRS_CSV = Path("results/phase2/pairs.csv")
OUT_DIR = Path("results/phase2/analysis")
GROUPS = ("close", "clear")

N_GRID = (25, 50, 80, 200)  # 80 = every item of a close pair; larger n resamples them
K_GRID = (10, 25, 50)
TRANSPORT = ((80, 25), (80, 50), (200, 25), (200, 50))
HUMANS = "humans"  # judge column for the humans-only baseline

PairKey = Tuple[str, str]


@dataclass(frozen=True)
class JudgeInfo:
    key: str
    label: str
    file: str
    size: str
    params_b: Optional[float]  # total parameters in billions, for ordering the ladder
    family: Optional[str]
    reasoning: bool


JUDGES = (
    JudgeInfo("qwen3-8b", "Qwen3 8B", "ollama_qwen3-8b-q4_K_M.jsonl", "8B", 8, "qwen", False),
    JudgeInfo("gemma-4-31b", "Gemma 4 31B", "nvidia_google_gemma-4-31b-it.jsonl", "31B", 31, "google", False),
    JudgeInfo(
        "nemotron-3-ultra", "Nemotron 3 Ultra", "nvidia_nvidia_nemotron-3-ultra-550b-a55b.jsonl",
        "550B (55B active)", 550, "nvidia", False,
    ),
    JudgeInfo(
        "mistral-large-4", "Mistral Large 4", "nvidia_mistralai_mistral-large-4-1t-a52b.jsonl",
        "about 1T (52B active)", 1000, "mistral", True,
    ),
    JudgeInfo(
        "gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "gemini_gemini-3.5-flash-lite.jsonl",
        "not published", None, "google", False,
    ),
)

# Model-name prefix -> family, for RQ7. First match wins.
FAMILY_PREFIXES = (
    ("qwen", "qwen"), ("gemini", "google"), ("gemma", "google"), ("mistral", "mistral"),
    ("magistral", "mistral"), ("claude", "anthropic"), ("chatgpt", "openai"), ("gpt", "openai"),
    ("o1", "openai"), ("o3", "openai"), ("o4", "openai"), ("llama", "meta"),
    ("deepseek", "deepseek"), ("command", "cohere"), ("grok", "xai"), ("amazon", "amazon"),
    ("nvidia", "nvidia"), ("nemotron", "nvidia"),
)


def family(model: str) -> str:
    name = model.lower()
    for prefix, fam in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return fam
    return "other"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairInfo:
    model_a: str
    model_b: str
    group: str
    bt: Estimate  # Bradley-Terry win rate of model_a (main truth)
    own: Estimate  # the pair's own votes (robustness check)

    @property
    def key(self) -> PairKey:
        return tuple(sorted((self.model_a, self.model_b)))  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return f"{self.model_a} vs {self.model_b}"

    @property
    def bt_winner(self) -> Optional[str]:
        return decide(self.bt)

    @property
    def own_winner(self) -> Optional[str]:
        return decide(self.own)


def read_pairs(path: Path = PAIRS_CSV, groups: Sequence[str] = GROUPS) -> Dict[PairKey, PairInfo]:
    out: Dict[PairKey, PairInfo] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] not in groups:
                continue
            p = PairInfo(
                r["model_a"], r["model_b"], r["group"],
                Estimate(float(r["bt_p"]), float(r["bt_low"]), float(r["bt_high"])),
                Estimate(float(r["own_p"]), float(r["own_low"]), float(r["own_high"])),
            )
            out[p.key] = p
    return out


def primary_order(item_id: str, orders: Sequence[str], seed: int = 0) -> str:
    """The one order an item is scored in: its only order, or the coin flip for two-way items."""
    return orders[0] if len(orders) == 1 else orders_for(item_id, "one", seed)[0]


def oriented(score: float, item: Item, pair: PairInfo) -> float:
    """Score from the pair's model_a view (items store it from their own model_a's view)."""
    return score if item.model_a == pair.model_a else 1.0 - score


def read_verdicts(path: Path) -> Dict[Tuple[str, str], str]:
    """(item id, order) -> latest readable verdict ("A", "B" or "tie")."""
    return {key: rec["verdict"] for key, rec in load_done(path).items()}


def build_pairs(
    plan: Sequence[Tuple[Item, Tuple[str, ...]]],
    pairs: Dict[PairKey, PairInfo],
    verdicts: Optional[Dict[Tuple[str, str], str]] = None,
    seed: int = 0,
) -> Dict[PairKey, PairData]:
    """Human and judge scores per pair, from the pair's model_a view.

    With no verdicts, the judge slot holds the human scores, so the humans-only
    baseline can run through the same simulation.
    """
    data = {key: PairData(p.model_a, p.model_b) for key, p in pairs.items()}
    for item, orders in plan:
        pair = pairs.get(item.pair)
        if pair is None:
            continue
        pd = data[item.pair]
        human = oriented(item.human_score, item, pair)
        pd.human[item.id] = [human]
        if verdicts is None:
            pd.judge[item.id] = [human]
            continue
        order = primary_order(item.id, orders, seed)
        verdict = verdicts.get((item.id, order))
        if verdict is not None:
            pd.judge[item.id] = [oriented(verdict_score(verdict, order), item, pair)]
    return data


# ---------------------------------------------------------------------------
# Full-data verdicts
# ---------------------------------------------------------------------------


def outcome(call: Optional[str], truth: Optional[str]) -> str:
    """How a confident call compares with the human truth."""
    if call is None:
        return "no_call"
    if truth is None:
        return "overclaim"
    return "correct" if call == truth else "wrong"


def points(p: float) -> float:
    """Win rate as points from a tie: 0.55 -> +5.0."""
    return round((p - 0.5) * 100, 2)


def verdict_rows(judge: str, pairs: Dict[PairKey, PairInfo], data: Dict[PairKey, PairData]) -> List[dict]:
    rows = []
    for key, p in sorted(pairs.items(), key=lambda kv: (kv[1].group, kv[0])):
        pd = data[key]
        items = pd.shared_items()
        if len(items) < 2:
            continue
        judge_scores = [pd.judge[i][0] for i in items]
        human_scores = [pd.human[i][0] for i in items]
        j = mean_ci(judge_scores, t=True)
        h = mean_ci(human_scores, t=True)
        call = decide(j)
        agree = agreement(pd)
        rows.append({
            "judge": judge,
            "group": p.group,
            "pair": p.name,
            "items": len(items),
            "human_bt": round(p.bt.value, 4),
            "human_bt_low": round(p.bt.low, 4),
            "human_bt_high": round(p.bt.high, 4),
            "truth_bt": p.bt_winner or "too_close",
            "human_own": round(p.own.value, 4),
            "truth_own": p.own_winner or "too_close",
            "human_items": round(h.value, 4),
            "judge_win_rate_a": round(j.value, 4),
            "judge_low": round(j.low, 4),
            "judge_high": round(j.high, 4),
            "judge_call": call or "none",
            "outcome_bt": outcome(call, p.bt_winner),
            "outcome_own": outcome(call, p.own_winner),
            "human_gap_points": points(p.bt.value),
            "judge_gap_points": points(j.value),
            "same_side": (j.value - 0.5) * (p.bt.value - 0.5) > 0,
            "raw_agreement": round(agree["raw_agreement"], 4),
            "kappa": round(agree["kappa"], 4),
            "best_judge_weight": round(best_judge_weight(human_scores, judge_scores), 4),
        })
    return rows


def best_judge_weight(human: Sequence[float], judge: Sequence[float]) -> float:
    """PPI++'s weight on the judge when the judge has scored many more items than humans:
    Cov(human, judge) / Var(judge), the slope of human score on judge score. Not clipped,
    so a negative value shows a judge that leans against the voters."""
    var_judge = _sample_variance(judge)
    if var_judge == 0:
        return 0.0
    mh, mj = fmean(human), fmean(judge)
    cov = sum((h - mh) * (j - mj) for h, j in zip(human, judge)) / (len(human) - 1)
    return cov / var_judge


# ---------------------------------------------------------------------------
# Simulated evals
# ---------------------------------------------------------------------------


def judge_configs() -> Iterable[dict]:
    for n in N_GRID:
        for rule in ("naive", "significant"):
            yield {"method": "judge", "n": n, "k": 0, "rule": rule}
        for k in K_GRID:
            if k < n:
                yield {"method": "ppi", "n": n, "k": k, "rule": "significant"}
                yield {"method": "ppi_tuned", "n": n, "k": k, "rule": "significant"}
    for n, k in TRANSPORT:
        yield {"method": "transport", "n": n, "k": k, "rule": "significant"}


def human_configs() -> Iterable[dict]:
    for n in N_GRID:
        for rule in ("naive", "significant"):
            yield {"method": "human_only", "n": n, "k": n, "rule": rule}
    for k in K_GRID:
        yield {"method": "human_only", "n": k, "k": k, "rule": "significant"}


def transport_sources(pairs: Dict[PairKey, PairInfo]) -> Dict[PairKey, PairKey]:
    """Each pair -> the next pair in the same group (sorted, wrapping round)."""
    out: Dict[PairKey, PairKey] = {}
    for group in GROUPS:
        keys = sorted(k for k, p in pairs.items() if p.group == group)
        for i, key in enumerate(keys):
            out[key] = keys[(i + 1) % len(keys)]
    return out


def grid_rows(
    judge: str,
    pairs: Dict[PairKey, PairInfo],
    data: Dict[PairKey, PairData],
    configs: Sequence[dict],
    reps: int,
    seed: int,
) -> List[dict]:
    sources = transport_sources(pairs)
    rows = []
    for key, p in sorted(pairs.items(), key=lambda kv: (kv[1].group, kv[0])):
        pd = data[key]
        pool = len(pd.shared_items())
        if pool < 2:
            continue
        for cfg in configs:
            source = data[sources[key]] if cfg["method"] == "transport" else None
            decisions = simulate_decisions(
                pd, cfg["n"], cfg["method"], cfg["k"], cfg["rule"], reps, seed,
                transport_source=source, t=True,
            )
            bt, own = tally(decisions, p.bt_winner), tally(decisions, p.own_winner)
            rows.append({
                "judge": judge,
                "group": p.group,
                "pair": p.name,
                "pool": pool,
                "method": cfg["method"],
                "n": cfg["n"],
                "k": cfg["k"],
                "rule": cfg["rule"],
                "reps": reps,
                "extrapolated": cfg["n"] > pool,
                "source": source.name if source else "",
                "call_a_rate": decisions.get("A", 0) / reps,
                "call_b_rate": decisions.get("B", 0) / reps,
                "no_decision_rate": decisions.get(None, 0) / reps,
                "truth_bt": p.bt_winner or "too_close",
                "correct_bt": bt["correct_rate"],
                "wrong_bt": bt["wrong_rate"],
                "overclaim_bt": bt["overclaim_rate"],
                "truth_own": p.own_winner or "too_close",
                "correct_own": own["correct_rate"],
                "wrong_own": own["wrong_rate"],
                "overclaim_own": own["overclaim_rate"],
            })
    return rows


def _grid_task(args) -> List[dict]:
    return grid_rows(*args)


# ---------------------------------------------------------------------------
# RQ9: position bias
# ---------------------------------------------------------------------------


def position_row(
    judge: str, plan: Sequence[Tuple[Item, Tuple[str, ...]]], verdicts: Dict[Tuple[str, str], str], seed: int = 0
) -> dict:
    """Verdict changes on items shown both ways, and how often the first answer wins."""
    counts: Counter = Counter()
    for item, orders in plan:
        first = verdicts.get((item.id, primary_order(item.id, orders, seed)))
        if first is not None:
            counts[f"primary_{first}"] += 1
        if len(orders) != 2:
            continue
        v_ab, v_ba = verdicts.get((item.id, "ab")), verdicts.get((item.id, "ba"))
        if v_ab is None or v_ba is None:
            counts["both_missing"] += 1
            continue
        counts["both"] += 1
        if verdict_score(v_ab, "ab") == verdict_score(v_ba, "ba"):
            continue
        counts["changed"] += 1
        if v_ab == v_ba == "A":
            counts["first_both_times"] += 1
        elif v_ab == v_ba == "B":
            counts["second_both_times"] += 1
        else:
            counts["one_tie"] += 1
    decisive = counts["primary_A"] + counts["primary_B"]
    primary = decisive + counts["primary_tie"]
    changed = wilson_interval(counts["changed"], counts["both"]) if counts["both"] else None
    first = wilson_interval(counts["primary_A"], decisive) if decisive else None
    return {
        "judge": judge,
        "items_both_ways": counts["both"],
        "items_missing_an_order": counts["both_missing"],
        "changed": counts["changed"],
        "changed_rate": round(changed.value, 4) if changed else "",
        "changed_low": round(changed.low, 4) if changed else "",
        "changed_high": round(changed.high, 4) if changed else "",
        "first_both_times": counts["first_both_times"],
        "second_both_times": counts["second_both_times"],
        "one_tie": counts["one_tie"],
        "primary_calls": primary,
        "tie_rate": round(counts["primary_tie"] / primary, 4) if primary else "",
        "first_answer_rate": round(first.value, 4) if first else "",
        "first_answer_low": round(first.low, 4) if first else "",
        "first_answer_high": round(first.high, 4) if first else "",
    }


# ---------------------------------------------------------------------------
# RQ7: same-family preference
# ---------------------------------------------------------------------------


def family_rows(
    infos: Sequence[JudgeInfo],
    pairs: Dict[PairKey, PairInfo],
    judged: Dict[str, Dict[PairKey, PairData]],
) -> List[dict]:
    """Does a judge favour its own family's model more than the other judges do?

    Per item, from the family model's view: self_minus_others = this judge's
    score minus the mean score of the other judges (not from the same family) on
    that item; judge_minus_humans = this judge's score minus the human vote.
    Rows per pair, plus a pooled row per judge ("all") over every item.
    """
    rows = []
    for info in infos:
        if info.family is None or info.key not in judged:
            continue
        others = [o.key for o in infos if o.key in judged and o.key != info.key and o.family != info.family]
        pooled_self: List[float] = []
        pooled_human: List[float] = []
        pooled_pairs = 0
        for key, p in sorted(pairs.items(), key=lambda kv: (kv[1].group, kv[0])):
            fa, fb = family(p.model_a) == info.family, family(p.model_b) == info.family
            if fa == fb:
                continue  # no model from the family, or both
            to_fam = (lambda s: s) if fa else (lambda s: 1.0 - s)
            mine = judged[info.key][key]
            self_diffs, human_diffs = [], []
            for item in mine.shared_items():
                other = [judged[o][key].judge[item][0] for o in others if item in judged[o][key].judge]
                if not other:
                    continue
                score = to_fam(mine.judge[item][0])
                self_diffs.append(score - fmean(to_fam(s) for s in other))
                human_diffs.append(score - to_fam(mine.human[item][0]))
            if len(self_diffs) < 2:
                continue
            pooled_self += self_diffs
            pooled_human += human_diffs
            pooled_pairs += 1
            rows.append(_family_row(info, p.group, p.name, p.model_a if fa else p.model_b, 1, self_diffs, human_diffs))
        if pooled_pairs:
            rows.append(_family_row(info, "all", "all", info.family, pooled_pairs, pooled_self, pooled_human))
    return rows


def _family_row(info: JudgeInfo, group: str, pair: str, model: str, n_pairs: int,
                self_diffs: List[float], human_diffs: List[float]) -> dict:
    s, h = mean_ci(self_diffs, t=True), mean_ci(human_diffs, t=True)
    return {
        "judge": info.key,
        "family": info.family,
        "group": group,
        "pair": pair,
        "family_model": model,
        "pairs": n_pairs,
        "items": len(self_diffs),
        "self_minus_others_points": round(s.value * 100, 2),
        "self_minus_others_low": round(s.low * 100, 2),
        "self_minus_others_high": round(s.high * 100, 2),
        "judge_minus_humans_points": round(h.value * 100, 2),
        "judge_minus_humans_low": round(h.low * 100, 2),
        "judge_minus_humans_high": round(h.high * 100, 2),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _grid_mean(rows: List[dict], judge: str, group: str, method: str, n: int, k: int, rule: str, field: str) -> Optional[float]:
    vals = [
        r[field] for r in rows
        if r["judge"] == judge and r["group"] == group and r["method"] == method
        and r["n"] == n and r["k"] == k and r["rule"] == rule
    ]
    return fmean(vals) if vals else None


def unsupported(row: dict, truth: str = "bt") -> float:
    """Share of evals that make a confident call humans don't back (overclaim + wrong)."""
    return row[f"overclaim_{truth}"] + row[f"wrong_{truth}"]


def summary_rows(
    infos: Sequence[JudgeInfo],
    pairs: Dict[PairKey, PairInfo],
    verdicts: List[dict],
    grid: List[dict],
    position: List[dict],
    judged: Dict[str, Dict[PairKey, PairData]],
) -> List[dict]:
    for r in grid:
        r["unsupported_bt"] = unsupported(r, "bt")
        r["unsupported_own"] = unsupported(r, "own")
    pos = {r["judge"]: r for r in position}
    rows = []
    for info in [i for i in infos if i.key in judged] + [None]:
        key = info.key if info else HUMANS
        v_close = [v for v in verdicts if v["judge"] == key and v["group"] == "close"]
        v_clear = [v for v in verdicts if v["judge"] == key and v["group"] == "clear"]
        row = {
            "judge": key,
            "label": info.label if info else "Humans only (one vote per item)",
            "size": info.size if info else "",
            "reasoning": info.reasoning if info else "",
            "close_pairs": len(v_close),
            "close_unsupported_all_items_bt": sum(v["outcome_bt"] in ("overclaim", "wrong") for v in v_close),
            "close_unsupported_all_items_own": sum(v["outcome_own"] in ("overclaim", "wrong") for v in v_close),
            "close_wrong_all_items_bt": sum(v["outcome_bt"] == "wrong" for v in v_close),
            "close_mean_abs_gap": round(fmean(abs(v["judge_gap_points"]) for v in v_close), 2) if v_close else "",
            "close_mean_abs_human_bt_gap": round(fmean(abs(v["human_gap_points"]) for v in v_close), 2) if v_close else "",
            "clear_pairs": len(v_clear),
            "clear_correct_bt": sum(v["outcome_bt"] == "correct" for v in v_clear),
            "clear_wrong_bt": sum(v["outcome_bt"] == "wrong" for v in v_clear),
            "clear_same_side": sum(bool(v["same_side"]) for v in v_clear),
        }
        if info:
            close_items = [
                (pd.human[i][0], pd.judge[i][0])
                for pk, pd in judged[key].items() if pairs[pk].group == "close"
                for i in pd.shared_items()
            ]
            h_labels = [score_label(h) for h, _ in close_items]
            j_labels = [score_label(j) for _, j in close_items]
            row.update({
                "close_kappa": round(cohen_kappa(h_labels, j_labels), 4) if close_items else "",
                "position_changed_rate": pos.get(key, {}).get("changed_rate", ""),
                "first_answer_rate": pos.get(key, {}).get("first_answer_rate", ""),
                "tie_rate": pos.get(key, {}).get("tie_rate", ""),
            })
        method = "judge" if info else "human_only"
        for n in N_GRID:
            k = 0 if info else n
            u = _grid_mean_unsupported(grid, key, method, n, k)
            row[f"close_unsupported_n{n}"] = round(u, 4) if u is not None else ""
            row[f"close_pairs_majority_unsupported_n{n}"] = _majority(grid, key, method, n, k)
        if info:
            for n in (80, 200):
                for method in ("ppi", "ppi_tuned"):
                    for k in K_GRID:
                        u = _grid_mean_unsupported(grid, key, method, n, k)
                        row[f"close_unsupported_{method}_n{n}_k{k}"] = round(u, 4) if u is not None else ""
                for n_t, k_t in TRANSPORT:
                    if n_t == n:
                        u = _grid_mean_unsupported(grid, key, "transport", n, k_t)
                        row[f"close_unsupported_transport_n{n}_k{k_t}"] = round(u, 4) if u is not None else ""
            # Power check: does a corrected eval still find the winner on clear pairs?
            c = _grid_mean(grid, key, "clear", "judge", 80, 0, "significant", "correct_bt")
            row["clear_correct_judge_n80"] = round(c, 4) if c is not None else ""
            for method in ("ppi", "ppi_tuned"):
                for k in K_GRID:
                    c = _grid_mean(grid, key, "clear", method, 80, k, "significant", "correct_bt")
                    row[f"clear_correct_{method}_n80_k{k}"] = round(c, 4) if c is not None else ""
            # PPI++'s weight on the judge when it scores far more items than humans (1 = full trust).
            for group, vs in (("close", v_close), ("clear", v_clear)):
                row[f"{group}_mean_best_judge_weight"] = round(fmean(v["best_judge_weight"] for v in vs), 4) if vs else ""
        else:
            for k in K_GRID:
                u = _grid_mean_unsupported(grid, key, "human_only", k, k)
                row[f"close_unsupported_k{k}"] = round(u, 4) if u is not None else ""
                c = _grid_mean(grid, key, "clear", "human_only", k, k, "significant", "correct_bt")
                row[f"clear_correct_k{k}"] = round(c, 4) if c is not None else ""
        rows.append(row)
    return rows


def _grid_mean_unsupported(grid: List[dict], judge: str, method: str, n: int, k: int) -> Optional[float]:
    return _grid_mean(grid, judge, "close", method, n, k, "significant", "unsupported_bt")


def _majority(grid: List[dict], judge: str, method: str, n: int, k: int) -> int:
    return sum(
        1 for r in grid
        if r["judge"] == judge and r["group"] == "close" and r["method"] == method
        and r["n"] == n and r["k"] == k and r["rule"] == "significant" and r["unsupported_bt"] > 0.5
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for r in rows:
        fields += [f for f in r if f not in fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _pct(x) -> str:
    return f"{x:5.1%}" if isinstance(x, float) else "    -"


def print_summary(summary: List[dict], position: List[dict], fam: List[dict]) -> None:
    def ks(r: dict, fmt: str) -> str:
        return " ".join(_pct(r.get(fmt.format(k=k))) for k in K_GRID)

    print("\nClose pairs: confident calls humans don't back (main truth: Bradley-Terry); simulated evals of 80 items")
    print(f"  {'':22s} {'all items':>9s} {'judge':>6s} {'n=200*':>6s}  {'PPI k=10/25/50':>17s}  "
          f"{'PPI++ k=10/25/50':>17s}  {'reused k=50':>11s}")
    for r in summary:
        if r["judge"] == HUMANS:
            print(f"  {'humans only':22s} {r['close_unsupported_all_items_bt']:>2d} of {r['close_pairs']:<3d} "
                  f"{_pct(r['close_unsupported_n80']):>6s} {_pct(r['close_unsupported_n200']):>6s}  "
                  f"humans alone, k=10/25/50: {ks(r, 'close_unsupported_k{k}')}")
            continue
        print(
            f"  {r['label']:22s} {r['close_unsupported_all_items_bt']:>2d} of {r['close_pairs']:<3d} "
            f"{_pct(r['close_unsupported_n80']):>6s} {_pct(r['close_unsupported_n200']):>6s}  "
            f"{ks(r, 'close_unsupported_ppi_n80_k{k}'):>17s}  {ks(r, 'close_unsupported_ppi_tuned_n80_k{k}'):>17s}  "
            f"{_pct(r.get('close_unsupported_transport_n80_k50')):>11s}"
        )
    print("  all items: pairs where the call on all 80 items isn't backed by humans.")
    print("  Other columns: share of simulated evals making such a call, averaged over close pairs.")
    print("  * n=200 resamples the 80 items, so it treats them as the whole population.")

    print("\nClear pairs (control): call on all 20 items vs the human winner; share of simulated "
          "80-item evals that find it (resampled from 20 items)")
    for r in summary:
        label = "humans only" if r["judge"] == HUMANS else r["label"]
        line = (f"  {label:22s} correct {r['clear_correct_bt']:2d}, wrong {r['clear_wrong_bt']:2d}, "
                f"same side {r['clear_same_side']:2d} of {r['clear_pairs']}")
        if r["judge"] != HUMANS:
            line += (f";  judge {_pct(r['clear_correct_judge_n80'])}, PPI {ks(r, 'clear_correct_ppi_n80_k{k}')}, "
                     f"PPI++ {ks(r, 'clear_correct_ppi_tuned_n80_k{k}')}")
        else:
            line += f";  humans alone k=10/25/50 {ks(r, 'clear_correct_k{k}')}"
        print(line)

    print("\nPPI++ weight on the judge with many unlabelled items (1 = trust fully, 0 = ignore), mean over pairs")
    for r in summary:
        if r["judge"] != HUMANS:
            print(f"  {r['label']:22s} close {r['close_mean_best_judge_weight']:+.2f}, "
                  f"clear {r['clear_mean_best_judge_weight']:+.2f}")

    print("\nRQ9 position bias (100 close items shown both ways)")
    for r in position:
        print(f"  {r['judge']:22s} changed {r['changed']:3d}/{r['items_both_ways']:<3d} "
              f"({_pct(r['changed_rate'])}); first answer both times {r['first_both_times']:3d}, "
              f"second {r['second_both_times']:3d}, one tie {r['one_tie']:3d}; "
              f"first answer wins {_pct(r['first_answer_rate'])} of decisive calls")

    print("\nRQ7 same-family preference (pooled; points toward the family model)")
    for r in fam:
        if r["pair"] == "all":
            print(f"  {r['judge']:22s} {r['pairs']} pairs, {r['items']} items: vs other judges "
                  f"{r['self_minus_others_points']:+.1f} [{r['self_minus_others_low']:+.1f}, {r['self_minus_others_high']:+.1f}], "
                  f"vs humans {r['judge_minus_humans_points']:+.1f} [{r['judge_minus_humans_low']:+.1f}, {r['judge_minus_humans_high']:+.1f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=5, help="processes for the simulation grid")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    pairs = read_pairs()
    plan = read_plan(PLAN_CSV)
    infos = [i for i in JUDGES if (JUDGMENTS_DIR / i.file).exists()]
    print(f"Pairs: {len(pairs)} ({sum(p.group == 'close' for p in pairs.values())} close); "
          f"plan items: {len(plan)}; judges: {', '.join(i.label for i in infos)}")

    all_verdicts = {i.key: read_verdicts(JUDGMENTS_DIR / i.file) for i in infos}
    judged = {i.key: build_pairs(plan, pairs, all_verdicts[i.key], args.seed) for i in infos}
    humans = build_pairs(plan, pairs)
    for i in infos:
        n = sum(len(pd.shared_items()) for pd in judged[i.key].values())
        print(f"  {i.label}: {n} of {len(plan)} items scored in their planned order")

    verdicts = [row for i in infos for row in verdict_rows(i.key, pairs, judged[i.key])]
    verdicts += verdict_rows(HUMANS, pairs, humans)  # the votes' own call, as a baseline
    tasks = [(i.key, pairs, judged[i.key], list(judge_configs()), args.reps, args.seed) for i in infos]
    tasks.append((HUMANS, pairs, humans, list(human_configs()), args.reps, args.seed))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        grid = [row for rows in pool.map(_grid_task, tasks) for row in rows]
    position = [position_row(i.key, plan, all_verdicts[i.key], args.seed) for i in infos]
    fam = family_rows(infos, pairs, judged)
    summary = summary_rows(infos, pairs, verdicts, grid, position, judged)

    for name, rows in (("verdicts", verdicts), ("grid", grid), ("position", position),
                       ("family", fam), ("summary", summary)):
        write_csv(args.out / f"{name}.csv", rows)
        print(f"Wrote {len(rows):5d} rows to {args.out / f'{name}.csv'}")
    print_summary(summary, position, fam)


if __name__ == "__main__":
    main()
