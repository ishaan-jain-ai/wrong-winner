"""Figures for the Phase 2 results.

    python -m wrong_winner.figures              # reads results/phase2/analysis/*.csv
    python -m wrong_winner.figures --out DIR

Needs the optional extra: pip install -e ".[figures]" (matplotlib). Every number comes from
the analysis CSVs that run_phase2 writes, so the figures rebuild without the raw data or the
judges. Each figure is written as PNG (for posts) and SVG (for the README):

1. close_pairs      every close pair: the judges' 80-item verdicts against all Arena votes
2. more_items       unsupported calls as simulated evals grow from 25 to 80 items
3. human_labels     what k human labels buy (PPI, PPI++, the labels alone, a reused correction)
4. position_bias    verdicts that change when the two answers swap places
5. leave_one_out    the headline with each model left out in turn
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Sequence, Tuple

ANALYSIS_DIR = Path("results/phase2/analysis")
OUT_DIR = Path("results/phase2/figures")

SOURCE_NOTE = (
    "Data: lmarena-ai/arena-human-preference-140k (CC BY 4.0), English single-turn votes. "
    "Analysis: wrong-winner, Phase 2."
)

# Light theme. Categorical slots 1-5 of the dataviz reference palette, in order, which pass
# the colour-vision checks as a set. Humans are always blue and judges always orange.
THEME = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "humans": "#2a78d6",
    "judge": "#eb6834",
    "ppi": "#1baf7a",
    "ppi_tuned": "#eda100",
    "transport": "#e87ba4",
}

LINE = 1.5  # points: about 2 px on screen
MARKER = 6.0  # points: about 8 px across

SHORT_NAMES = {
    "amazon.nova-pro-v1:0": "Nova Pro",
    "chatgpt-4o-latest-20250326": "ChatGPT-4o",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-3-7-sonnet-20250219-thinking-32k": "Claude 3.7 Sonnet (thinking)",
    "claude-opus-4-20250514": "Claude Opus 4",
    "claude-opus-4-20250514-thinking-16k": "Claude Opus 4 (thinking)",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-sonnet-4-20250514-thinking-32k": "Claude Sonnet 4 (thinking)",
    "command-a-03-2025": "Command A",
    "deepseek-r1-0528": "DeepSeek R1",
    "deepseek-v3-0324": "DeepSeek V3",
    "gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-pro-preview-03-25": "Gemini 2.5 Pro (preview)",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 mini",
    "grok-3-mini-beta": "Grok 3 Mini",
    "grok-3-preview-02-24": "Grok 3",
    "llama-4-maverick-03-26-experimental": "Llama 4 Maverick (exp.)",
    "mistral-medium-2505": "Mistral Medium 3",
    "o3-2025-04-16": "o3",
    "o4-mini-2025-04-16": "o4-mini",
    "qwen3-235b-a22b-no-thinking": "Qwen3 235B",
}


def short_name(model: str) -> str:
    return SHORT_NAMES.get(model, model)


def pair_label(pair: str) -> str:
    a, b = pair.split(" vs ")
    return f"{short_name(a)} vs {short_name(b)}"


def spread_labels(values: Sequence[float], min_gap: float) -> List[float]:
    """Positions for labels next to `values`: at least `min_gap` apart, same order.

    Labels that would collide are grouped and centred on their values, so each one stays
    as close to its mark as the spacing allows (a leader line then joins label and mark).
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: List[Tuple[float, List[int]]] = []  # (first position, members low to high)
    for i in order:
        clusters.append((values[i], [i]))
        while len(clusters) > 1:
            (start1, members1), (start2, members2) = clusters[-2], clusters[-1]
            if start1 + len(members1) * min_gap <= start2 + 1e-12:
                break
            merged = members1 + members2
            centre = fmean(values[j] for j in merged)
            clusters[-2:] = [(centre - (len(merged) - 1) * min_gap / 2, merged)]
    out = [0.0] * len(values)
    for start, members in clusters:
        for rank, j in enumerate(members):
            out[j] = start + rank * min_gap
    return out


# ---------------------------------------------------------------- data


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def judge_rows(summary: List[dict]) -> List[dict]:
    """Judges in the summary's order (by size), with a display label."""
    out = []
    for row in summary:
        if row["judge"] == "humans":
            continue
        label = row["label"] + (" (reasoning)" if row["reasoning"] == "True" else "")
        out.append({**row, "display": label})
    return out


def humans_row(summary: List[dict]) -> dict:
    return next(row for row in summary if row["judge"] == "humans")


def judge_mean(judges: List[dict], column: str) -> float:
    return fmean(float(row[column]) for row in judges)


# ---------------------------------------------------------------- drawing helpers


def _rc() -> dict:
    t = THEME
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "figure.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.75,
        "axes.labelcolor": t["ink2"],
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": t["axis"],
        "ytick.color": t["axis"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 6,
        "grid.color": t["grid"],
        "grid.linewidth": 0.75,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "svg.fonttype": "path",
        "svg.hashsalt": "wrong-winner",  # stable ids, so a rebuild doesn't change the SVGs
        "legend.frameon": False,
    }


def _canvas(plt, width, height, ncols=1, *, left, right, top, bottom, gap=0.0, sharey=False):
    """A figure with `ncols` panels; margins and the gap between panels are in inches."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=sharey, squeeze=False)
    panel_w = (width - left - right - (ncols - 1) * gap) / ncols
    fig.subplots_adjust(
        left=left / width, right=1 - right / width,
        top=1 - top / height, bottom=bottom / height, wspace=gap / panel_w,
    )
    return fig, list(axes[0])


def _header(fig, title: str, subtitle: str) -> None:
    width, height = fig.get_size_inches()
    t = THEME
    fig.text(0.3 / width, 1 - 0.25 / height, title, fontsize=15, fontweight="bold",
             color=t["ink"], va="top")
    fig.text(0.3 / width, 1 - 0.62 / height, subtitle, fontsize=10, color=t["ink2"],
             va="top", linespacing=1.45)


def _footer(fig, note: str = "") -> None:
    width, height = fig.get_size_inches()
    text = (note + "\n" if note else "") + SOURCE_NOTE
    fig.text(0.3 / width, 0.18 / height, text, fontsize=7.8, color=THEME["muted"],
             va="bottom", linespacing=1.5)


def _panel_title(ax, title: str, detail: str = "", y: float = 1.0) -> None:
    ax.text(0, y, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=THEME["ink"], va="bottom", ha="left")
    if detail:
        ax.text(0, y - 0.005, detail, transform=ax.transAxes, fontsize=9, color=THEME["ink2"],
                va="top", ha="left")


def _percent_axis(axis, ticks: Sequence[float]) -> None:
    axis.set_ticks(list(ticks))
    axis.set_ticklabels([f"{v:.0%}" for v in ticks])


def _dot(ax, x, y, color, *, filled=True, marker="o", size=MARKER, z=4, ring=True):
    """A marker with a surface-coloured ring (filled) or a hollow one (not filled)."""
    # A ring is drawn half inside the marker, so grow a ringed marker to keep its fill size.
    ringed = filled and ring
    ax.plot([x], [y], marker=marker, markersize=size + (LINE if ringed else 0),
            linestyle="none", zorder=z,
            markerfacecolor=color if filled else THEME["surface"],
            markeredgecolor=THEME["surface"] if filled else color,
            markeredgewidth=LINE if (ring or not filled) else 0, clip_on=False)


def _label_with_value(ax, x, y, label, value_text, *, fontsize=9):
    """'Label  42%' with the value in bold primary ink, left-aligned at (x, y) in data units."""
    t = THEME
    name = ax.text(x, y, label + "  ", color=t["ink2"], fontsize=fontsize, va="center",
                   ha="left", clip_on=False)
    ax.annotate(value_text, xy=(1, 0.5), xycoords=name, va="center", ha="left",
                fontsize=fontsize, color=t["ink"], fontweight="bold", annotation_clip=False)


def _line_labels(ax, x_end, x_text, values, labels, *, min_gap, fmt="{:.0%}"):
    t = THEME
    positions = spread_labels(values, min_gap)
    for value, pos, label in zip(values, positions, labels):
        if abs(pos - value) > min_gap * 0.05:
            ax.plot([x_end, x_text], [value, pos], color=t["muted"], lw=0.6,
                    clip_on=False, zorder=1, solid_capstyle="butt")
        _label_with_value(ax, x_text, pos, label, fmt.format(value))


# ---------------------------------------------------------------- figure 1


def fig_close_pairs(plt, verdicts: List[dict], summary: List[dict]):
    from matplotlib.lines import Line2D

    t = THEME
    judges = judge_rows(summary)
    close = [r for r in verdicts if r["group"] == "close"]
    by_key = {(r["judge"], r["pair"]): r for r in close}
    pairs = sorted({r["pair"] for r in close})
    first = judges[0]["judge"]

    def calls(pair):
        return sum(by_key[(j["judge"], pair)]["judge_call"] != "none" for j in judges)

    def spread(pair):
        return fmean(abs(float(by_key[(j["judge"], pair)]["judge_win_rate_a"]) - 0.5)
                     for j in judges)

    def ordered(group):
        return sorted(group, key=lambda p: (-calls(p), -spread(p), pair_label(p)))

    tie = ordered([p for p in pairs if by_key[(first, p)]["truth_bt"] == "too_close"])
    won = ordered([p for p in pairs if by_key[(first, p)]["truth_bt"] != "too_close"])

    rows: List[Tuple[float, str]] = []  # (y, pair) from the top
    headings: List[Tuple[float, str]] = []
    y = 0.0
    for title, group in (
        (f"All Arena votes: too close to call ({len(tie)} pairs)", tie),
        (f"All Arena votes: a narrow winner ({len(won)} pairs)", won),
    ):
        if not group:
            continue
        headings.append((y, title))
        y += 1
        for pair in group:
            rows.append((y, pair))
            y += 1
        y += 0.6
    y_max = y - 0.6

    row_h = 0.27
    plot_h = (y_max + 0.8) * row_h
    top, bottom = 2.45, 1.35
    height = top + plot_h + bottom
    panel_w, gap, left, right = 2.35, 0.3, 3.0, 0.35
    width = left + len(judges) * panel_w + (len(judges) - 1) * gap + right
    fig, axes = _canvas(plt, width, height, len(judges), left=left, right=right, top=top,
                        bottom=bottom, gap=gap, sharey=True)

    _header(
        fig,
        "On close pairs, LLM judges pick winners that people don't",
        "15 close LMArena pairs, 80 items each, every item judged once. Dots show the win rate "
        "of the first-named model with its 95% interval.\nA judge names a winner when its "
        "interval clears 50%. People's verdict comes from every Arena vote on the pair.",
    )
    handles = [
        Line2D([], [], color=t["humans"], lw=LINE + 1, marker="D", markersize=MARKER - 1,
               markeredgewidth=0, label="All Arena votes on the pair (Bradley-Terry)"),
        Line2D([], [], color=t["judge"], lw=LINE, marker="o", markersize=MARKER + LINE,
               markeredgecolor=t["surface"], markeredgewidth=LINE,
               label="Judge on the 80 items: names a winner"),
        Line2D([], [], color=t["judge"], lw=LINE, marker="o", markersize=MARKER,
               markerfacecolor=t["surface"], markeredgecolor=t["judge"], markeredgewidth=LINE,
               label="Judge: no call"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.3 / width, 1 - 1.3 / height),
               ncol=3, fontsize=9, labelcolor=t["ink2"], handlelength=2.2, columnspacing=2.2,
               borderaxespad=0, borderpad=0)

    for ax, judge in zip(axes, judges):
        unsupported = sum(
            by_key[(judge["judge"], p)]["outcome_bt"] in ("overclaim", "wrong") for p in pairs
        )
        _panel_title(ax, judge["display"], f"{unsupported} of {len(pairs)} calls people don't back",
                     y=1.0 + 0.5 / plot_h)
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(y_max + 0.2, -0.6)
        _percent_axis(ax.xaxis, (0.25, 0.5, 0.75))
        ax.grid(axis="x")
        ax.axvline(0.5, color=t["axis"], lw=1.0, zorder=1)
        ax.set_axisbelow(True)
        for y_row, pair in rows:
            r = by_key[(judge["judge"], pair)]
            low, high = float(r["judge_low"]), float(r["judge_high"])
            ax.plot([low, high], [y_row, y_row], color=t["judge"], lw=LINE, zorder=2)
            h_low, h_high = float(r["human_bt_low"]), float(r["human_bt_high"])
            ax.plot([h_low, h_high], [y_row, y_row], color=t["humans"], lw=LINE + 1, zorder=3)
            _dot(ax, float(r["human_bt"]), y_row, t["humans"], marker="D", size=MARKER - 1, z=3,
                 ring=False)
            _dot(ax, float(r["judge_win_rate_a"]), y_row, t["judge"],
                 filled=r["judge_call"] != "none", z=4)

    axes[0].set_yticks([y_row for y_row, _ in rows])
    axes[0].set_yticklabels([pair_label(p) for _, p in rows])
    for y_head, title in headings:
        axes[0].annotate(title, xy=(0, y_head), xycoords=("axes fraction", "data"),
                         xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                         fontsize=9, fontweight="bold", color=t["ink"], annotation_clip=False)
    fig.text((left + (width - left - right) / 2) / width, (bottom - 0.4) / height,
             "Win rate of the first-named model (50% = tie)", ha="center", va="top",
             fontsize=9.5, color=t["ink2"])
    _footer(fig, "Rows are sorted by how many judges name a winner. Every call on the 12 "
                 "too-close pairs is one people don't back; on the 3 narrow pairs, every call "
                 "agrees with them.")
    return fig


# ---------------------------------------------------------------- figure 2


def fig_more_items(plt, summary: List[dict]):
    t = THEME
    judges = judge_rows(summary)
    humans = humans_row(summary)
    sizes = (25, 50, 80)
    width, height = 9.0, 6.4
    fig, (ax,) = _canvas(plt, width, height, left=0.85, right=3.0, top=1.55, bottom=1.25)
    _header(
        fig,
        "More items make the judges surer of winners people don't back",
        "Simulated evals on the 15 close pairs, 2,000 per pair and size. Share of evals whose "
        "95% interval\nnames a winner that the Arena votes don't support.",
    )
    ax.set_xlim(22, 83)
    ax.set_ylim(0, 0.5)
    ax.set_xticks(sizes)
    ax.set_xlabel("Items in the eval, each judged once")
    _percent_axis(ax.yaxis, (0, 0.1, 0.2, 0.3, 0.4, 0.5))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.axhline(0.05, color=t["axis"], lw=1.0, zorder=1)
    ax.text(23, 0.044, "5%: what a 95% interval allows on an exact tie", fontsize=8.5,
            color=t["muted"], va="top")

    ends, labels = [], []
    for judge in judges:
        ys = [float(judge[f"close_unsupported_n{n}"]) for n in sizes]
        ax.plot(sizes, ys, color=t["judge"], lw=LINE, zorder=3)
        for x, y in zip(sizes, ys):
            _dot(ax, x, y, t["judge"])
        ends.append(ys[-1])
        labels.append(judge["display"])
    ys = [float(humans[f"close_unsupported_n{n}"]) for n in sizes]
    ax.plot(sizes, ys, color=t["humans"], lw=LINE, zorder=3)
    for x, y in zip(sizes, ys):
        _dot(ax, x, y, t["humans"], marker="D", size=MARKER - 1)
    ends.append(ys[-1])
    labels.append("Human votes on the same items")
    _line_labels(ax, 80, 85, ends, labels, min_gap=0.024)
    _footer(fig, "Evals are drawn from each pair's 80 judged items. The human line takes the "
                 "one Arena vote on each item; it rises too because, for a few pairs,\nthose "
                 "80 votes lean further from a tie than all of the pair's votes do. Larger evals "
                 "would resample the same 80 items, so they aren't shown.")
    return fig


# ---------------------------------------------------------------- figure 3


def fig_human_labels(plt, summary: List[dict]):
    t = THEME
    judges = judge_rows(summary)
    humans = humans_row(summary)
    ks = (10, 25, 50)
    width, height = 13.2, 6.6
    fig, (ax_close, ax_clear) = _canvas(plt, width, height, 2, left=0.85, right=2.55, top=1.95,
                                        bottom=1.45, gap=2.35)
    _header(
        fig,
        "A few human labels stop the invented winners, but the judge adds little on top",
        "Simulated 80-item evals in which k items also carry one person's vote; average over the "
        "4 judges. PPI corrects the judge's score\nwith the labelled items (prediction-powered "
        "inference); PPI++ also tunes how much to trust the judge.",
    )

    def series(ax, xs, ys, color, marker="o"):
        ax.plot(xs, ys, color=color, lw=LINE, zorder=3)
        for x, y in zip(xs, ys):
            _dot(ax, x, y, color, marker=marker, size=MARKER - (1 if marker == "D" else 0))

    def judge_alone(ax, value):
        ax.plot([ks[0], ks[-1]], [value, value], color=t["judge"], lw=LINE, zorder=3)
        _dot(ax, ks[-1], value, t["judge"])

    for ax in (ax_close, ax_clear):
        ax.set_xlim(7, 52)
        ax.set_xticks(ks)
        ax.set_xlabel("Items also labelled by a person (k)")
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    # Close pairs: unsupported calls (lower is better).
    ax = ax_close
    ax.set_ylim(0, 0.5)
    _percent_axis(ax.yaxis, (0, 0.1, 0.2, 0.3, 0.4, 0.5))
    _panel_title(ax, "Close pairs: evals naming a winner people don't back",
                 "Lower is better", y=1.0 + 0.5 / (height - 1.95 - 1.45))
    jl = judge_mean(judges, "close_unsupported_n80")
    tr = [judge_mean(judges, f"close_unsupported_transport_n80_k{k}") for k in (25, 50)]
    pp = [judge_mean(judges, f"close_unsupported_ppi_n80_k{k}") for k in ks]
    pt = [judge_mean(judges, f"close_unsupported_ppi_tuned_n80_k{k}") for k in ks]
    hu = [float(humans[f"close_unsupported_k{k}"]) for k in ks]
    judge_alone(ax, jl)
    series(ax, (25, 50), tr, t["transport"])
    series(ax, ks, pp, t["ppi"])
    series(ax, ks, pt, t["ppi_tuned"])
    series(ax, ks, hu, t["humans"], marker="D")
    _line_labels(
        ax, 50, 54, [jl, tr[-1], pt[-1], hu[-1], pp[-1]],
        ["Judge alone, no labels", "Correction from another pair", "PPI++",
         "The k labels alone", "PPI"],
        min_gap=0.026,
    )

    # Clear pairs: finding the real winner (higher is better).
    ax = ax_clear
    ax.set_ylim(0, 1.0)
    _percent_axis(ax.yaxis, (0, 0.2, 0.4, 0.6, 0.8, 1.0))
    _panel_title(ax, "Clear pairs: evals naming the real winner",
                 "Higher is better", y=1.0 + 0.5 / (height - 1.95 - 1.45))
    jl = judge_mean(judges, "clear_correct_judge_n80")
    pp = [judge_mean(judges, f"clear_correct_ppi_n80_k{k}") for k in ks]
    pt = [judge_mean(judges, f"clear_correct_ppi_tuned_n80_k{k}") for k in ks]
    hu = [float(humans[f"clear_correct_k{k}"]) for k in ks]
    judge_alone(ax, jl)
    series(ax, ks, pp, t["ppi"])
    series(ax, ks, pt, t["ppi_tuned"])
    series(ax, ks, hu, t["humans"], marker="D")
    _line_labels(ax, 50, 54, [jl, pt[-1], hu[-1], pp[-1]],
                 ["Judge alone, no labels", "PPI++", "The k labels alone", "PPI"],
                 min_gap=0.052)
    _footer(fig, "The k labels alone: an eval of just the k labelled items, no judge. Correction "
                 "from another pair: PPI with the labels taken from a different pair (close pairs "
                 "only).\nClear pairs have 20 judged items each, so their 80-item evals resample "
                 "them. Per-judge numbers are in summary.csv.")
    return fig


# ---------------------------------------------------------------- figure 4


def fig_position_bias(plt, position: List[dict], summary: List[dict]):
    t = THEME
    judges = judge_rows(summary)
    by_judge = {r["judge"]: r for r in position}
    judges = [j for j in judges if j["judge"] in by_judge]
    width, height = 12.0, 4.35
    fig, (ax_swap, ax_first) = _canvas(plt, width, height, 2, left=2.3, right=0.6, top=1.75,
                                       bottom=0.95, gap=0.9, sharey=True)
    _header(
        fig,
        "Swap the two answers and a quarter to half of the verdicts change",
        "Mostly toward whichever answer is shown first. 95% intervals.",
    )
    ys = list(range(len(judges)))
    panel_top = 1.0 + 0.5 / (height - 1.75 - 0.95)
    for ax, col, low, high, title, detail, xlim, ticks, ref in (
        (ax_swap, "changed_rate", "changed_low", "changed_high",
         "Verdict changes when the answers swap", "100 close-pair items shown both ways",
         (0, 0.8), (0, 0.2, 0.4, 0.6, 0.8), None),
        (ax_first, "first_answer_rate", "first_answer_low", "first_answer_high",
         "First answer wins", "Share of decisive verdicts on all 1,400 items",
         (0.4, 0.8), (0.4, 0.5, 0.6, 0.7, 0.8), 0.5),
    ):
        ax.set_xlim(*xlim)
        _percent_axis(ax.xaxis, ticks)
        ax.set_ylim(len(judges) - 0.4, -0.6)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        _panel_title(ax, title, detail, y=panel_top)
        if ref is not None:
            ax.axvline(ref, color=t["axis"], lw=1.0, zorder=1)
            ax.annotate("no preference", xy=(ref, len(judges) - 0.4), xytext=(4, 2),
                        textcoords="offset points", fontsize=8.5, color=t["muted"],
                        va="bottom", ha="left")
        for y, judge in zip(ys, judges):
            r = by_judge[judge["judge"]]
            ax.plot([float(r[low]), float(r[high])], [y, y], color=t["judge"], lw=LINE, zorder=2)
            _dot(ax, float(r[col]), y, t["judge"])
            ax.annotate(f"{float(r[col]):.0%}", xy=(float(r[high]), y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        fontweight="bold", color=t["ink"])
    ax_swap.set_yticks(ys)
    ax_swap.set_yticklabels([j["display"] for j in judges])
    _footer(fig)
    return fig


# ---------------------------------------------------------------- figure 5


def fig_leave_one_out(plt, loo: List[dict], summary: List[dict]):
    t = THEME
    judges = judge_rows(summary)
    rows = [(j["judge"], j["display"], t["judge"], "o") for j in judges]
    rows.append(("humans", "Human votes on the same items", t["humans"], "D"))
    full = {r["judge"]: r for r in loo if r["left_out"] == "(none)"}
    runs: Dict[str, List[dict]] = {}
    for r in loo:
        if r["left_out"] != "(none)":
            runs.setdefault(r["judge"], []).append(r)
    n_models = len({r["left_out"] for r in loo if r["left_out"] != "(none)"})

    width, height = 12.0, 4.6
    fig, axes = _canvas(plt, width, height, 2, left=2.45, right=0.6, top=1.75, bottom=0.95,
                        gap=0.9, sharey=True)
    _header(
        fig,
        "No single model drives the result",
        f"Each of the {n_models} models in the close pairs left out in turn, with all of its "
        f"pairs. Line: range over the {n_models} runs; dot: all 15 close pairs.",
    )
    panel_top = 1.0 + 0.5 / (height - 1.75 - 0.95)
    for ax, col, title, detail in (
        (axes[0], "unsupported_share", "Pairs where the call on all items isn't backed by people",
         "Share of the remaining close pairs"),
        (axes[1], "sim_unsupported_n80", "Simulated 80-item evals making such a call",
         "Average over the remaining close pairs"),
    ):
        ax.set_xlim(0, 0.6)
        _percent_axis(ax.xaxis, (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        _panel_title(ax, title, detail, y=panel_top)
        for y, (key, _, color, marker) in enumerate(rows):
            values = [float(r[col]) for r in runs.get(key, [])]
            if values and max(values) > min(values):
                ax.plot([min(values), max(values)], [y, y], color=color, lw=LINE + 1.5,
                        alpha=0.35, zorder=2, solid_capstyle="round")
            value = float(full[key][col])
            _dot(ax, value, y, color, marker=marker,
                 size=MARKER - (1 if marker == "D" else 0))
            if not values:
                span = ""
            elif max(values) == min(values):
                span = f"{values[0]:.0%} in every run"
            else:
                span = f"{min(values) * 100:.0f}\u2013{max(values):.0%}"
            ax.annotate(span, xy=(max(values + [value]), y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=t["ink2"])
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[1] for r in rows])
    _footer(fig)
    return fig


# ---------------------------------------------------------------- main


FIGURES = ("close_pairs", "more_items", "human_labels", "position_bias", "leave_one_out")


def build_all(analysis_dir: Path = ANALYSIS_DIR, out_dir: Path = OUT_DIR,
              formats: Sequence[str] = ("png", "svg"), dpi: int = 200) -> List[Path]:
    """Draw every figure from the analysis CSVs; return the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = read_csv(analysis_dir / "summary.csv")
    verdicts = read_csv(analysis_dir / "verdicts.csv")
    position = read_csv(analysis_dir / "position.csv")
    loo = read_csv(analysis_dir / "leave_one_out.csv")
    makers = {
        "close_pairs": lambda: fig_close_pairs(plt, verdicts, summary),
        "more_items": lambda: fig_more_items(plt, summary),
        "human_labels": lambda: fig_human_labels(plt, summary),
        "position_bias": lambda: fig_position_bias(plt, position, summary),
        "leave_one_out": lambda: fig_leave_one_out(plt, loo, summary),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with matplotlib.rc_context(_rc()):
        for number, name in enumerate(FIGURES, start=1):
            fig = makers[name]()
            for fmt in formats:
                path = out_dir / f"fig{number}_{name}.{fmt}"
                fig.savefig(path, dpi=dpi, metadata={"Software": None} if fmt == "png" else
                            {"Date": None, "Creator": None})
                written.append(path)
            plt.close(fig)
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    for path in build_all(args.analysis, args.out, dpi=args.dpi):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
