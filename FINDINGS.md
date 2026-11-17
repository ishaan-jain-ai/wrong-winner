# Phase 1 findings (MT-Bench, GPT-4 judge)

Run: 17 Nov 2026, `python -m wrong_winner.run_baseline` (2,000 repetitions, seed 0).
Data: MT-Bench human judgments (3,355 expert votes) and the bundled GPT-4 pairwise
judgments (2,400), 6 models, 15 model pairs, 99–146 shared items per pair.

## Headline

**The LLM judge exaggerates differences. That is harmless when there is a real gap,
and harmful when there isn't. More eval data makes it worse, not better.
A handful of human labels on the same comparison fixes it; labels from a different
comparison do not.**

## 1. The judge amplifies every gap

On all 14 pairs where humans see a clear winner, the judge picks the same winner
and makes the gap **bigger** (median 1.31×; range 1.05×–2.65×).
Example: humans give gpt-4 a 61.5% win rate over gpt-3.5-turbo, the judge gives it 80.6%.

GPT-4 also flipped its verdict when the answers were swapped on 380 of 2,400 items (16%),
which the dataset records as "tie (inconsistent)".

## 2. When the gap is real and large, small evals are fine

Across the 14 clear-winner pairs (human gaps of 8.6 to 39 points), the judge-only eval
never picked the opposite winner, at any size from 25 to 200 items.
With the significance rule, it reached the right decision 93% of the time at n = 25 and
98% at n = 50.

## 3. When there is no real gap, the judge invents one

claude-v1 vs gpt-3.5-turbo: humans are split (50.9%, interval includes 50%).
The judge says claude-v1 wins 62.8% (13.7× amplification).

| Eval size (n) | Judge only: confident call humans can't back | Humans only (same n) |
|---|---|---|
| 25 | 54% | 6% |
| 50 | 80% | 6% |
| 100 | 98% | 5% |
| 200 | 100% | 8% |

Adding more items makes the judge's bias look *more* significant. A team running a
"proper" 200-item eval with error bars would ship with full confidence on a difference
humans can't see.

## 4. Ten human labels fix it, but only from the same comparison

Judge on n items plus k human labels, corrected with prediction-powered inference (PPI):

| Setup on the too-close pair | Overclaim rate |
|---|---|
| Judge only, n = 100 | 98% |
| PPI, n = 100, k = 10 | 8% |
| PPI, n = 100, k = 50 | 6% |
| Correction reused from another pair, n = 100, k = 50 | 74% |

The reused correction (measured on claude-v1 vs gpt-4) left the overclaim rate almost
unchanged. This matches the "transported calibration" failure in Li 2026
(*Frozen Judges, Moving Agents*), here on chat evals.

## 5. The price of the fix is more "no decision"

On clear-winner pairs, PPI almost never picked a wrong winner (at most 0.1%), but it decided less often:

| n | Judge only (significance rule) | PPI, k = 10 | PPI, k = 50 |
|---|---|---|---|
| 100 | 99.8% correct | 58% correct, 42% no decision | 78% correct, 22% no decision |
| 200 | 100% correct | 59% correct, 41% no decision | 83% correct, 17% no decision |

The trade-off: the judge alone is decisive but can't tell a real gap from an
invented one; PPI is honest but needs more labels to be decisive.

## 6. Item-level agreement doesn't predict decision quality

Cohen's kappa between judge and humans was low on every pair (0.09–0.44), including
pairs where every decision was right. The one pair where the judge misled had a
middling kappa (0.28). A "judge agreement" score alone would not have flagged it.

## Limits (state these in the write-up)

- **One too-close pair.** Findings 3–4 rest on a single comparison. Phase 2 needs
  more close comparisons before any "X% of evals" claim.
- 2023 models and a 2023 GPT-4 judge.
- "Too close" means the human interval on ~128 items includes 50%; humans may
  still differ slightly.
- With k = 10, the simple interval is a little too narrow (humans-only overclaims
  ~8–9% against a nominal 5%); a t-based or Wilson-style interval would fix this.

## Next (Phase 2)

Get many close comparisons, the kind teams actually face (prompt tweaks, model
upgrades with 1–5 point gaps):
- run a current judge on human-preference data that has close model pairs, and/or
- create close pairs deliberately (same model, two prompt variants) and collect a
  small set of human labels.
