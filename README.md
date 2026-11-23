# wrong-winner

**How often does your LLM eval pick the wrong winner?**

Teams decide whether to ship a prompt or model change by running a small eval
(often 50–200 examples) graded by an LLM judge, usually once. This project measures
how often that process picks a different winner than human raters would, and
how many human labels it takes to fix it.

> **Write-up:** [Your LLM judge says v2 wins. It might be a tie.](https://ishaan-jain-ai.github.io/llm-judge-ties/), an 8-minute read with the charts.
>
> **Status:** Phase 1 (MT-Bench, GPT-4 judge) is done: results in [FINDINGS.md](FINDINGS.md).
> Phase 2 (four current judges on close LMArena comparisons) has results and figures in [PHASE2.md](PHASE2.md).
> To check your own A/B eval, see [Check your own A/B eval](#check-your-own-ab-eval).
> Design, research questions and related work: [STUDY.md](STUDY.md).

## What it measures

For each pair of systems with human and LLM-judge verdicts on the same items:

1. **Wrong-winner rate**: the eval picks a winner, but humans (on all the data) disagree.
2. **Noise vs bias**: humans on the same small sample, versus the judge.
3. **Significance rule**: only ship when the 95% interval excludes a tie.
4. **Human-label budget**: prediction-powered inference (PPI) with k human labels.
5. **Reused corrections**: what happens when a correction from one pair is reused on another.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[data,dev]"
python -m wrong_winner.run_baseline --reps 200   # quick check
python -m wrong_winner.run_baseline              # full run
python scripts/worked_example.py                  # one comparison, step by step
```

The full run writes `results/baseline.csv` (every simulated setting) and
`results/agreement.csv` (judge–human agreement per pair). Both are committed, so
the numbers in [FINDINGS.md](FINDINGS.md) can be checked without rerunning.

Phase 1 uses the GPT-4 judgments bundled with
[MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
(CC-BY-4.0), so it needs no API key.

## Check your own A/B eval

You changed a prompt or a model, and an LLM judge says the new version wins.
`wrong-winner check` says whether that verdict is backed by evidence, using what this
study found. It makes no LLM calls and has no dependencies.

```bash
pip install -e .                                   # from a clone
wrong-winner check verdicts.csv --a current --b candidate
```

Without installing: `PYTHONPATH=src python -m wrong_winner check ...`

**Input:** a CSV with one row per test item.

| Column | What it holds |
|---|---|
| `id` | a unique id (optional; row numbers are used without it) |
| `judge_a_first` | the judge's verdict when A's answer was shown first |
| `judge_b_first` | the judge's verdict when B's answer was shown first |
| `human` | your own label (optional) |

A verdict is `a`, `b`, `tie`, or the names given to `--a` and `--b`. In the judge columns,
`first` and `second` work too. Either judge column may be blank on a row, but give both when
you can: a verdict that flips with the answer order counts as a tie.

**Results:**

- **B is better / A is better:** backed by your labels, or by a judge-only win rate more
  than 30 points from a tie (80/20).
- **Not proven:** the judge leans one way, even with an interval that excludes a tie, but by
  no more than judges managed on pairs people saw as ties. More items won't fix this.
- **No measurable difference:** says roughly how many labels would settle it, or that the gap
  is too small to chase and the choice can rest on cost, latency or safety.
- **Too few items:** fewer than 10 judged.

**Adding human labels:**

```bash
wrong-winner pick verdicts.csv 25 -o labels.csv    # random items, judge's verdicts hidden
# fill in the human column: a, b or tie
wrong-winner check verdicts.csv --labels labels.csv
```

With 10 or more labels, the result rests on PPI++: the judge and your labels together, with
the judge weighted by how well it tracks your labels. In Phase 2 that weight was 0.06–0.29,
so the labels did most of the work.

**In CI:** exit code 0 means the gate passed, 1 that it didn't, 2 bad input. `--gate better`
(the default) passes only when B is shown better; `--gate not-worse` fails only when A is
shown better. `--json` prints the full report.

**From Python:**

```python
from wrong_winner import check, render

report = check("verdicts.csv", labels="labels.csv", a_name="current", b_name="candidate")
report.result   # "b_better", "a_better", "no_difference", "not_proven" or "too_few_items"
print(render(report))
```

**Try it on real data.** [`examples/`](examples/) holds Gemma 4 31B's verdicts on two Phase 2
pairs, with real LMArena votes as the labels (rebuilt by `scripts/make_check_examples.py`):

```bash
wrong-winner check examples/close_pair.csv --a ChatGPT-4o --b o3
# NOT PROVEN: the judge gives o3 74%; people see a tie
wrong-winner check examples/close_pair.csv --labels examples/close_pair_labels.csv --a ChatGPT-4o --b o3
# NO MEASURABLE DIFFERENCE: 25 votes put o3 at 56% (95% interval 40–72%)
wrong-winner check examples/clear_pair.csv --a "Claude 3.7 Sonnet" --b "Gemini 2.5 Pro"
# GEMINI 2.5 PRO IS BETTER (judge alone, wide margin)
```

**Where the rules come from, and their limits:**

- **The 30-point rule:** in Phase 2, judges' win rates went up to 26 points from a tie on
  pairs people saw as ties, and 12.5–50 points on pairs people could clearly separate.
- **Run over every Phase 2 result** (`scripts/make_check_examples.py`): judging alone, the check
  named no winner on any of the 48 judge-and-pair results where people see a tie, and the
  right winner on 24 of 40 clear-pair results, never the wrong one. With 25 random labels it
  invented a winner on 6% of checks on ties, close to the 5% a 95% interval allows. The rule
  was set from these same pairs, so this is a sanity check, not an independent test.
- **Scope:** four open judges, pairwise preference, English single-turn chat. Your judge or
  task may behave differently; human labels are the real check.
- **PPI++ assumes the labelled items are a random sample:** use `pick`, and label without
  looking at the judge's verdicts.

Example data: [lmarena-ai/arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
(CC BY 4.0), item ids and verdicts only.

## Use the statistics in your own evals

The core has no dependencies:

```python
from wrong_winner import mean_ci, decide, ppi_mean

est = mean_ci(win_scores)              # A wins = 1, tie = 0.5, B wins = 0
decide(est, rule="significant")        # "A", "B", or None = not enough evidence

# Judge on all items, humans on a few: correct the judge's bias
est = ppi_mean(judge_unlabelled, judge_labelled, human_labelled)
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # no install needed
```

## License

Code: MIT. Data: see each dataset's own license.
