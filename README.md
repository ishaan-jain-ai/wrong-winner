# wrong-winner

**How often does your LLM eval pick the wrong winner?**

Teams decide whether to ship a prompt or model change by running a small eval
(often 50–200 examples) graded by an LLM judge, usually once. This project measures
how often that process picks a different winner than human raters would, and
how many human labels it takes to fix it.

> **Status:** Phase 1 (MT-Bench, GPT-4 judge) is done: results in [FINDINGS.md](FINDINGS.md).
> Phase 2 (current judges on close LMArena comparisons) is planned in [PHASE2.md](PHASE2.md).
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
