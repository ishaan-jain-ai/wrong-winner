"""Is an LLM judge's A/B verdict backed by evidence? A check for one comparison.

Typical use: you changed a prompt or model (A = current, B = candidate), ran a
test set through both, and asked an LLM judge which answer is better. This
module turns those verdicts, plus optional human labels on a random sample of
items, into a plain result: B is better, A is better, no measurable
difference, or not proven.

Input is one row per test item:

    id, judge_a_first, judge_b_first, human

- judge_a_first: the judge's verdict when A's answer was shown first.
- judge_b_first: the judge's verdict when B's answer was shown first.
  Either column may be blank on a row; give both when you can.
- human: your own label for the item (optional, blank when not labelled).

A verdict names the winning system ("a", "b", "tie") or its position
("first", "second"). Scores are from B's point of view: B better = 1, tie =
0.5, A better = 0. An item judged both ways scores the average, so a verdict
that flips with the answer order counts as a tie.

The rules come from Phase 2 of this study (PHASE2.md): four open judges on 25
LMArena pairs. They are defaults, not laws: your judge and task may differ, and
human labels are the real check.
"""

from __future__ import annotations

import csv
import math
import random
import shlex
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .stats import Estimate, cohen_kappa, mean_ci, ppi_lambda, ppi_mean, wilson_interval

ID, A_FIRST, B_FIRST, HUMAN = "id", "judge_a_first", "judge_b_first", "human"
JUDGE_COLUMNS = (A_FIRST, B_FIRST)
TIE_WORDS = ("tie", "draw", "equal", "same")

# Phase 2: on pairs people saw as ties, judges' win rates went up to 26 points
# from a tie (76/24); on clear pairs they ran 12.5–50 points from a tie. A judge
# result only counts on its own when it is further out than any tie produced.
CLEAR_GAP = 0.30
TIE_GAP_SEEN = 26  # points: the furthest a judge got from a tie on a pair people saw as a tie
MIN_ITEMS = 10  # fewer judged items than this: no result
MIN_LABELS = 10  # fewer labels than this: they don't decide the result
START_LABELS = 25  # how many labels to suggest first
CHEAP_DECISION = 200  # labels needed above this: suggest deciding on cost instead
MAX_LABELS_SHOWN = 1000

RESULTS = ("b_better", "a_better", "no_difference", "not_proven", "too_few_items")
GATES = ("better", "not-worse", "none")

Pathish = Union[str, Path]


class InputError(ValueError):
    """A problem with an input file, worded for the person who made it."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def parse_verdict(value: str, column: str, a_name: str = "A", b_name: str = "B") -> Optional[float]:
    """Score from B's point of view (B better = 1, tie = 0.5, A better = 0); None when blank.

    Accepts "a"/"b", the systems' names, "tie" (or draw/equal/same) and, in the
    judge columns, "first"/"second" for the answer's position.
    """
    v = value.strip().lower()
    if not v:
        return None
    if v in TIE_WORDS:
        return 0.5
    if v in ("a", a_name.strip().lower()):
        return 0.0
    if v in ("b", b_name.strip().lower()):
        return 1.0
    if column in JUDGE_COLUMNS and v in ("first", "second"):
        b_shown_first = column == B_FIRST
        return 1.0 if (v == "first") == b_shown_first else 0.0
    words = f"a, b, {a_name}, {b_name}, tie" + (", first, second" if column in JUDGE_COLUMNS else "")
    raise InputError(f"can't read {value!r} in column {column!r} (expected one of: {words}, or blank)")


@dataclass(frozen=True)
class Item:
    """One test item. Scores are from B's point of view; None = not given."""

    id: str
    a_first: Optional[float] = None  # judge, A's answer shown first
    b_first: Optional[float] = None  # judge, B's answer shown first
    human: Optional[float] = None

    @property
    def judge(self) -> Optional[float]:
        """The judge's score, averaged over the orders it saw."""
        scores = [s for s in (self.a_first, self.b_first) if s is not None]
        return fmean(scores) if scores else None

    @property
    def both_orders(self) -> bool:
        return self.a_first is not None and self.b_first is not None


def _rows(path: Path) -> Tuple[List[str], List[dict]]:
    """Header and rows of a CSV file, with header names stripped and lower-cased."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = [h.strip().lower() for h in (reader.fieldnames or [])]
            rows = [{h.strip().lower(): (v or "") for h, v in r.items() if h is not None} for r in reader]
    except FileNotFoundError:
        raise InputError(f"no such file: {path}") from None
    return header, rows


def _ids(header: Sequence[str], rows: Sequence[dict], path: Path) -> List[str]:
    """Each row's id, or its row number (1, 2, ...) when the file has no id column."""
    if ID not in header:
        return [str(i) for i in range(1, len(rows) + 1)]
    ids = [r[ID].strip() for r in rows]
    for line, item_id in enumerate(ids, start=2):
        if not item_id:
            raise InputError(f"{path}, line {line}: empty id")
    seen = set()
    for line, item_id in enumerate(ids, start=2):
        if item_id in seen:
            raise InputError(f"{path}, line {line}: id {item_id!r} appears more than once")
        seen.add(item_id)
    return ids


def _parse(row: dict, column: str, path: Path, line: int, a_name: str, b_name: str) -> Optional[float]:
    try:
        return parse_verdict(row.get(column, ""), column, a_name, b_name)
    except InputError as e:
        raise InputError(f"{path}, line {line}: {e}") from None


def read_items(path: Pathish, a_name: str = "A", b_name: str = "B") -> List[Item]:
    """Items from a verdicts CSV (column names are case-insensitive)."""
    path = Path(path)
    header, rows = _rows(path)
    if not any(c in header for c in JUDGE_COLUMNS):
        raise InputError(
            f"{path}: needs a {A_FIRST!r} or {B_FIRST!r} column (found: {', '.join(header) or 'nothing'})"
        )
    items = []
    for line, (item_id, row) in enumerate(zip(_ids(header, rows, path), rows), start=2):
        items.append(
            Item(
                item_id,
                _parse(row, A_FIRST, path, line, a_name, b_name),
                _parse(row, B_FIRST, path, line, a_name, b_name),
                _parse(row, HUMAN, path, line, a_name, b_name),
            )
        )
    return items


def read_labels(path: Pathish, a_name: str = "A", b_name: str = "B") -> Dict[str, float]:
    """id -> human score from a labels CSV (columns id and human; blank = not labelled)."""
    path = Path(path)
    header, rows = _rows(path)
    missing = [c for c in (ID, HUMAN) if c not in header]
    if missing:
        raise InputError(f"{path}: needs columns 'id' and 'human' (missing: {', '.join(missing)})")
    labels = {}
    for line, (item_id, row) in enumerate(zip(_ids(header, rows, path), rows), start=2):
        score = _parse(row, HUMAN, path, line, a_name, b_name)
        if score is not None:
            labels[item_id] = score
    return labels


def add_labels(items: Sequence[Item], labels: Dict[str, float]) -> List[Item]:
    """Items with human labels filled in from `labels` (id -> score)."""
    known = {it.id for it in items}
    unknown = sorted(set(labels) - known)
    if unknown:
        shown = ", ".join(repr(i) for i in unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        raise InputError(f"{len(unknown)} labelled id(s) are not in the verdicts file: {shown}")
    out = []
    for it in items:
        if it.id not in labels:
            out.append(it)
            continue
        if it.human is not None and it.human != labels[it.id]:
            raise InputError(f"item {it.id!r} has a different human label in each file")
        out.append(Item(it.id, it.a_first, it.b_first, labels[it.id]))
    return out


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """Everything the check found. Win rates are B's, with ties counting half."""

    a_name: str
    b_name: str
    items: int  # items with at least one judge verdict
    both_orders: int
    a_first_only: int
    b_first_only: int
    judge: Estimate
    flips: int  # items judged both ways whose verdict changed with the order
    flip_rate: Optional[Estimate]
    first_answer_rate: Optional[Estimate]  # decisive verdicts on items judged both ways
    labels: int
    result: str
    basis: str  # "labels" or "judge"
    clear_gap: float
    confidence: float
    humans: Optional[Estimate] = None
    combined: Optional[Estimate] = None  # PPI++: judge and labels together
    judge_weight: Optional[float] = None
    agreed: Optional[int] = None
    kappa: Optional[float] = None
    labels_needed: Optional[int] = None
    first_wins: int = 0  # first-answer wins among `decisive` verdicts on items judged both ways
    decisive: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def decided(self) -> Estimate:
        """The estimate the result rests on."""
        return self.combined if self.basis == "labels" and self.combined else self.judge

    def to_dict(self) -> dict:
        def clean(x):
            if isinstance(x, float) and math.isnan(x):
                return None
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [clean(v) for v in x]
            return x

        return clean(asdict(self))


def side(score: float) -> str:
    """"b", "a" or "tie" for a score from B's point of view."""
    return "b" if score > 0.5 else "a" if score < 0.5 else "tie"


def call(est: Estimate) -> str:
    """The result an interval supports on its own."""
    if est.low > 0.5:
        return "b_better"
    if est.high < 0.5:
        return "a_better"
    return "no_difference"


def mean_interval(scores: Sequence[float], confidence: float = 0.95) -> Estimate:
    """Mean with a Student's t interval, or a Wilson interval when every score is 0 or every
    score is 1 (where the t interval would have zero width)."""
    est = mean_ci(scores, confidence, t=True)
    if est.low == est.high and est.value in (0.0, 1.0):
        return wilson_interval(round(est.value * len(scores)), len(scores), confidence)
    return est


def labels_needed(est: Estimate, k: int) -> Optional[int]:
    """Roughly how many labels would make the interval exclude a tie, if the gap is as estimated.

    Interval width shrinks with the square root of the labels. None when the
    estimate sits exactly on a tie.
    """
    gap = abs(est.value - 0.5)
    if gap == 0:
        return None
    half = (est.high - est.low) / 2
    return max(k + 1, math.ceil(k * (half / gap) ** 2))


def check(
    items: Union[Sequence[Item], Pathish],
    labels: Union[Dict[str, float], Pathish, None] = None,
    a_name: str = "A",
    b_name: str = "B",
    confidence: float = 0.95,
    clear_gap: float = CLEAR_GAP,
) -> Report:
    """Check an A/B comparison. `items` and `labels` may be CSV paths (see the module docstring)."""
    if isinstance(items, (str, Path)):
        items = read_items(items, a_name, b_name)
    if isinstance(labels, (str, Path)):
        labels = read_labels(labels, a_name, b_name)
    if labels:
        items = add_labels(items, labels)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not 0 < clear_gap <= 0.5:
        raise ValueError("clear_gap must be between 0 and 0.5")

    judged = [it for it in items if it.judge is not None]
    if not judged:
        raise InputError("no row has a judge verdict")
    notes: List[str] = []
    if len(judged) < len(items):
        lost = [it for it in items if it.judge is None]
        lost_labels = sum(it.human is not None for it in lost)
        notes.append(
            f"{len(lost)} row(s) had no judge verdict and were left out"
            + (f", {lost_labels} of them labelled." if lost_labels else ".")
        )

    judge = mean_interval([it.judge for it in judged], confidence)

    # Answer order
    both = [it for it in judged if it.both_orders]
    a_only = sum(it.a_first is not None and it.b_first is None for it in judged)
    b_only = sum(it.b_first is not None and it.a_first is None for it in judged)
    flips = sum(it.a_first != it.b_first for it in both)
    flip_rate = wilson_interval(flips, len(both), confidence) if both else None
    first_wins = decisive = 0
    for it in both:
        if it.a_first != 0.5:  # A shown first: A winning (0) is the first answer winning
            decisive += 1
            first_wins += it.a_first == 0.0
        if it.b_first != 0.5:
            decisive += 1
            first_wins += it.b_first == 1.0
    first_answer_rate = wilson_interval(first_wins, decisive, confidence) if decisive else None
    single = a_only + b_only
    if not both:
        notes.append(
            "The judge saw each item in one order only. Run both orders: in Phase 2 the first "
            "answer shown won 58.5–68.5% of decisive verdicts."
        )
    if single >= MIN_ITEMS and abs(a_only / single - 0.5) > 0.1:
        favoured, share = (a_name, a_only / single) if a_only > b_only else (b_name, b_only / single)
        notes.append(
            f"{share:.0%} of the items judged in one order showed {favoured} first, which tilts the "
            f"judge toward {favoured} if it favours the first answer."
        )

    # Human labels
    labelled = [it for it in judged if it.human is not None]
    k = len(labelled)
    report = Report(
        a_name, b_name, len(judged), len(both), a_only, b_only, judge, flips, flip_rate,
        first_answer_rate, k, result="", basis="judge", clear_gap=clear_gap, confidence=confidence,
        first_wins=first_wins, decisive=decisive, notes=notes,
    )
    if k >= 2:
        human = [it.human for it in labelled]
        judge_lab = [it.judge for it in labelled]
        judge_unlab = [it.judge for it in judged if it.human is None]
        report.humans = mean_interval(human, confidence)
        report.judge_weight = ppi_lambda(judge_unlab, judge_lab, human)
        report.combined = ppi_mean(judge_unlab, judge_lab, human, confidence, t=True, lam=report.judge_weight)
        if report.combined.low == report.combined.high:  # only when the labels are all the same
            report.combined = report.humans
        judge_sides, human_sides = [side(s) for s in judge_lab], [side(s) for s in human]
        report.agreed = sum(j == h for j, h in zip(judge_sides, human_sides))
        report.kappa = cohen_kappa(judge_sides, human_sides)
        if side(judge.value) != "tie" and side(report.humans.value) not in ("tie", side(judge.value)):
            notes.append("The judge and your labels lean in opposite directions.")
    if 0 < k < MIN_LABELS:
        notes.append(
            f"Only {k} label(s): at least {MIN_LABELS} are needed before labels decide the result. "
            f"Label {START_LABELS} if you can."
        )

    # The result
    if k >= MIN_LABELS:
        report.basis = "labels"
        report.result = call(report.combined)
        if report.result == "no_difference":
            report.labels_needed = labels_needed(report.combined, k)
    elif len(judged) < MIN_ITEMS:
        report.result = "too_few_items"
    else:
        judged_call = call(judge)
        if judged_call == "no_difference" or abs(judge.value - 0.5) >= clear_gap:
            report.result = judged_call
        else:
            report.result = "not_proven"
    return report


def exit_code(report: Report, gate: str = "better") -> int:
    """0 = pass, 1 = fail. "better": pass only if B is shown better. "not-worse": fail only
    if A is shown better. "none": always pass."""
    if gate == "none":
        return 0
    if gate == "better":
        return 0 if report.result == "b_better" else 1
    if gate == "not-worse":
        return 1 if report.result == "a_better" else 0
    raise ValueError(f"gate must be one of {GATES}")


# ---------------------------------------------------------------------------
# Picking items to label
# ---------------------------------------------------------------------------


def pick(path: Pathish, n: int, seed: int = 0, a_name: str = "A", b_name: str = "B") -> Tuple[List[str], List[dict]]:
    """A random sample of n judged items to label, as (header, rows), in file order.

    The judge's verdicts are left out, so they can't sway the labeller, and an
    empty `human` column is added. Rows keep every other column (the prompt and
    answers, if the file has them). Files without an id column get row numbers
    as ids, the same ones `read_items` uses.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    path = Path(path)
    judged_ids = {it.id for it in read_items(path, a_name, b_name) if it.judge is not None}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        original = [h for h in (reader.fieldnames or []) if h is not None]
        rows = list(reader)
    lowered = [h.strip().lower() for h in original]
    keep = [h for h, low in zip(original, lowered) if low not in JUDGE_COLUMNS + (HUMAN, ID)]
    id_column = next((h for h, low in zip(original, lowered) if low == ID), None)
    numbered = [
        (r[id_column].strip() if id_column else str(i), r) for i, r in enumerate(rows, start=1)
    ]
    candidates = [(i, item_id, r) for i, (item_id, r) in enumerate(numbered) if item_id in judged_ids]
    chosen = sorted(random.Random(seed).sample(candidates, min(n, len(candidates))))
    header = [ID] + keep + [HUMAN]
    out = [{ID: item_id, **{h: r.get(h, "") for h in keep}, HUMAN: ""} for _, item_id, r in chosen]
    return header, out


# ---------------------------------------------------------------------------
# Plain-language report
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _interval(est: Estimate, confidence: float) -> str:
    return f"{confidence:.0%} interval {est.low * 100:.1f}–{est.high * 100:.1f}%"


def render(report: Report, source: str = "verdicts.csv") -> str:
    """The report as text for a terminal."""
    r = report
    a, b = r.a_name, r.b_name
    lines = [f"wrong-winner check: {a} vs {b}, {r.items} items", ""]

    lines += [
        "Judge alone",
        f"  {b} wins {_pct(r.judge.value)} (ties count half; {_interval(r.judge, r.confidence)})",
        "  That interval covers sampling noise, not the judge's own bias.",
    ]
    if r.both_orders:
        lines += _wrap(
            f"Swapping the answers changed the verdict on {r.flips} of {r.both_orders} items judged "
            f"both ways ({r.flip_rate.value:.0%}; close pairs in Phase 2: 26–51%)."
        )
    if r.decisive:
        lines += _wrap(
            f"The first answer shown won {r.first_wins} of {r.decisive} decisive verdicts on those "
            f"items ({r.first_answer_rate.value:.0%})."
        )
    if r.a_first_only or r.b_first_only:
        lines += _wrap(
            f"Judged in one order only: {r.a_first_only} items with {a} first, {r.b_first_only} with {b} first."
        )

    if r.humans:
        lines += [
            "",
            f"Your labels ({r.labels} items)",
            f"  {b} wins {_pct(r.humans.value)} ({_interval(r.humans, r.confidence)})",
        ]
        kappa = "" if r.kappa is None or math.isnan(r.kappa) else f" (kappa {r.kappa:.2f})"
        lines += _wrap(f"The judge agreed with you on {r.agreed} of {r.labels} items{kappa}.")
        lines += _wrap(
            f"Judge and labels combined (PPI++, judge weight {r.judge_weight:.2f}): "
            f"{b} wins {_pct(r.combined.value)} ({_interval(r.combined, r.confidence)})"
        )
        lines += _wrap("This assumes the labelled items were picked at random (`wrong-winner pick` does that).")

    lines += ["", f"Result: {headline(r)}"]
    for paragraph in explanation(r, source):
        # Commands stay on one line so they can be copied.
        lines += [f"    {paragraph[2:]}"] if paragraph.startswith("$ ") else _wrap(paragraph)
    if r.notes:
        lines += ["", "Notes"]
        for note in r.notes:
            lines += _wrap(note, bullet=True)
    return "\n".join(lines)


def _wrap(text: str, bullet: bool = False, width: int = 88) -> List[str]:
    first, rest = ("  - ", "    ") if bullet else ("  ", "  ")
    return textwrap.wrap(text, width, initial_indent=first, subsequent_indent=rest, break_on_hyphens=False)


def headline(r: Report) -> str:
    if r.result in ("b_better", "a_better"):
        winner = r.b_name if r.result == "b_better" else r.a_name
        if r.basis == "labels":
            return f"{winner.upper()} IS BETTER"
        wide = abs(r.judge.value - 0.5) * 100 > TIE_GAP_SEEN
        return f"{winner.upper()} IS BETTER (judge alone{', wide margin' if wide else ''})"
    return {
        "no_difference": "NO MEASURABLE DIFFERENCE",
        "not_proven": "NOT PROVEN",
        "too_few_items": "TOO FEW ITEMS",
    }[r.result]


def explanation(r: Report, source: str) -> List[str]:
    a, b = r.a_name, r.b_name
    names = "" if (a, b) == ("A", "B") else f" --a {shlex.quote(a)} --b {shlex.quote(b)}"
    file = shlex.quote(source)
    next_step = [
        f"Next, label {START_LABELS} items picked at random, without looking at the judge's verdicts "
        "(fill in the human column with a, b or tie), then check again:",
        f"$ wrong-winner pick {file} {START_LABELS} -o labels.csv{names}",
        f"$ wrong-winner check {file} --labels labels.csv{names}",
    ]
    gap_points = abs(r.decided.value - 0.5) * 100
    if r.result in ("b_better", "a_better") and r.basis == "labels":
        return [f"Backed by your {r.labels} labels: the interval excludes a tie."]
    if r.result in ("b_better", "a_better"):
        wide = gap_points > TIE_GAP_SEEN
        evidence = (
            "and gaps this wide came from pairs people could clearly separate."
            if wide
            else "so a gap this size could still be a tie."
        )
        return [
            f"The judge is {gap_points:.0f} points from a tie. In Phase 2, judges never went further "
            f"than {TIE_GAP_SEEN} points from a tie on pairs people saw as ties, {evidence}",
            f"{START_LABELS} random labels would {'confirm' if wide else 'check'} it.",
        ]
    if r.result == "not_proven":
        winner = b if r.judge.value > 0.5 else a
        return [
            f"The judge favours {winner} by {gap_points:.0f} points and its interval excludes a tie, "
            f"but that is not enough. In Phase 2, on pairs people saw as ties, judges went up to "
            f"{TIE_GAP_SEEN} points from a tie and named a winner on 5–7 of 15 pairs.",
            "More test items won't fix this; human labels will.",
            *next_step,
        ]
    if r.result == "too_few_items":
        return [f"Only {r.items} items have a judge verdict; at least {MIN_ITEMS} are needed."]
    # no_difference
    if r.basis == "judge":
        return [
            f"Even the judge can't separate {a} and {b} on these items.",
            *next_step,
        ]
    out = [f"With {r.labels} labels, {a} and {b} are too close to tell apart."]
    if r.labels_needed is None:
        out.append("Your labels show no gap at all.")
    elif r.labels_needed <= MAX_LABELS_SHOWN:
        out.append(f"If the gap is really {gap_points:.1f} points, roughly {r.labels_needed} labels would settle it.")
    else:
        out.append(f"A gap of {gap_points:.1f} points would take more than {MAX_LABELS_SHOWN:,} labels to settle.")
    if r.labels_needed is None or r.labels_needed > CHEAP_DECISION:
        out.append("A gap this small may not be worth chasing: decide on cost, latency or safety instead.")
    return out
