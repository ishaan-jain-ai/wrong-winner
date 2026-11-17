# Study design: How often does your LLM eval pick the wrong winner?

**Status:** draft, 17 Nov 2026. Target publish date: 30 Nov 2026.

## The question in one sentence

When a team compares two versions of an LLM feature with a small LLM-judged eval
(25–200 examples, one judge, one run), how often does the eval pick a different
winner than human raters would, and how many human labels does it take to fix that?

## Why this is worth doing (gap in existing work)

| Existing work | What it covers | What it leaves open |
|---|---|---|
| Miller, *Adding Error Bars to Evals* (arXiv 2411.00640) | Standard errors, paired tests, power analysis | No LLM judge error, no human ground truth |
| *Signal and Noise* (arXiv 2508.13144) | Decision accuracy vs benchmark size | Benchmarks with fixed answers, not LLM judges |
| Anand, *Ask Which, Not How Good* (arXiv 2609.27787) | Minimum detectable difference for LLM-judged benchmarks, judge self-consistency | Does not measure wrong-winner rate against humans; no judge-error correction; no human-label budget |
| Li, *Frozen Judges, Moving Agents* (arXiv 2609.34198) | Judge calibration fails to transfer across agent versions | Agent benchmarks; not practitioner-sized chat evals |
| *Reliability without Validity* (arXiv 2606.19544) | Judge–human agreement for 21 judges | Agreement, not decisions |

Our angle is **decisions, at practitioner scale, against human ground truth**, plus a
concrete answer to "how many human labels do I need?"

## Research questions

- **RQ1 – Wrong-winner rate.** At n = 25, 50, 100, 200 items, how often does a single
  LLM judge pick the wrong winner between two systems, relative to the full-data human verdict?
- **RQ2 – Noise vs bias.** How much of that error is sampling noise (humans on the same
  small sample also get it wrong) versus judge bias (the judge is wrong even on all data)?
- **RQ3 – Does a significance rule help?** If teams only "ship" when the 95% interval
  excludes a tie, how much does the wrong-winner rate drop, and how often do they get "no decision"?
- **RQ4 – Human-label budget.** If k of the n items also get a human label and we use
  prediction-powered inference (PPI) to correct the judge, how does the wrong-winner rate
  fall as k grows? Is PPI better than just using the k human labels alone?
- **RQ5 – When correction hurts.** If the correction is estimated on one model pair and
  reused on another (what teams do in practice), does it make decisions worse?
- **RQ6 – Is "judge agreement > X" enough?** Popular guides tell teams to trust a judge
  once it agrees with humans above a threshold (e.g. "Pearson r > 0.7"). Does a judge's
  item-level agreement (raw agreement, Cohen's kappa, Pearson r) actually predict how
  often it picks the right winner? In particular: how often do judges that *pass* the
  usual thresholds still pick the wrong winner on close comparisons?

## Data

| Dataset | Use | License |
|---|---|---|
| `lmsys/mt_bench_human_judgments` (Hugging Face): ~3.3k expert pairwise judgments of 6 models on 80 questions (2 turns), plus ~2.4k GPT-4 pairwise judgments of the same pairs | Main study. 15 model pairs, each with human and judge verdicts on the same items. Phase 1 needs **no API spend**, because GPT-4 judgments ship with the dataset. | CC-BY-4.0 (per dataset card) |
| `nvidia/HelpSteer2` (+ preference annotations) | Phase 2: judge bias (position, length) at larger scale, with modern judges | CC-BY-4.0 (per dataset card) |

Still to verify before use: exact column names and values on first download, and
the license of any Chatbot Arena data if added later.

## Method

Each comparison item is a (question, turn) for a model pair (A, B). Scores are from
A's point of view: A wins = 1, tie = 0.5, B wins = 0.

1. **Ground truth.** For each pair, average the human score over all items. The pair
   has a *clear winner* if the 95% interval excludes 0.5; otherwise it is reported
   separately as *too close to call* (and any confident decision there counts as overclaiming).
2. **Simulated team eval.** Draw n items with replacement (the "super-population" view in
   Miller 2024). The judge scores them. Apply a decision rule:
   - *naive*: pick whichever side scores higher;
   - *significant*: pick a winner only if the 95% interval excludes 0.5, otherwise "no decision".
3. **Human-label correction (RQ4).** For k of the n items, draw one random human
   judgment (one annotator, as a team would have). Estimate the win rate with
   PPI (Angelopoulos et al., 2023): judge mean on unlabelled items plus the mean
   human-minus-judge gap on labelled items. Compare against "humans only on k items".
4. **Transport (RQ5).** Estimate the human-minus-judge gap on pair X, apply it to pair Y.
5. **Agreement vs decisions (RQ6).** For each pair, compute item-level judge–human
   agreement: raw agreement and Cohen's kappa on labels (A / B / tie, from the item's
   mean score) and Pearson r on item scores. Plot each pair's agreement against its
   wrong-winner rate at each n, split by how close the pair is (human win rate near
   0.5 vs far from it). Phase 2 adds more judges so there are more points per plot.
6. Repeat 2,000 times per (pair, n, k, rule). Report:
   - **wrong-winner rate**: decided, but wrong;
   - **correct rate**: decided and right;
   - **no-decision rate**;
   - with intervals across pairs.

Binary pass/fail evals use Rogan–Gladen correction instead of PPI (also implemented),
with a refusal when the judge is too close to chance (sensitivity + specificity − 1 ≤ 0.1).

## Phases and timeline

| Dates | Work |
|---|---|
| 17–19 Nov | Stats module + tests (done in scaffold), data loader, verify dataset schema |
| 20–22 Nov | Phase 1: RQ1–RQ3 on MT-Bench with bundled GPT-4 judgments (no API cost) |
| 23–25 Nov | RQ4–RQ6; optional Phase 2 with 1–2 current judges on a sample (small API budget) |
| 26–27 Nov | Figures + the "cheat sheet" (decision table: n × effect size × labels) |
| 28–30 Nov | Blog post, README, one-command reproduction, launch |

## Deliverables

1. A blog post with 3–4 charts and one headline number per RQ, including
   "agreement vs wrong-winner rate" (RQ6) as the lead chart if the result is clear.
2. A one-page cheat sheet: "for an expected gap of X points, use n items and k human labels."
3. This repo: `python -m wrong_winner.run_baseline` reproduces every number.

## Threats to validity (say these up front)

- MT-Bench has only 6 (older) models and 80 questions; findings may not transfer to modern models.
- GPT-4 pairwise judgments are from 2023; Phase 2 checks whether a current judge changes the picture.
- Bootstrap resampling of 160 items per pair treats them as a population; rare failure modes are under-represented.
- Expert human raters disagree too; we report human–human agreement as the ceiling.
