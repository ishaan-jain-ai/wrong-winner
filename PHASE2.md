# Phase 2 plan: many close comparisons, current judges

**Status:** plan, 17 Nov 2026. Phase 1 results: [FINDINGS.md](FINDINGS.md).

## Why Phase 2

Phase 1's most important finding (a judge confidently picks a winner humans can't
separate, and more data makes it worse) rests on **one** close pair. MT-Bench compares
2023 models with large gaps. Teams usually compare versions 1–5 points apart.
Phase 2 tests the finding on dozens of close comparisons with current judges, so the
post can say "in X of Y close comparisons" instead of "in one example".

## Data: LMArena human preferences (140k)

[`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)

| Fact | Value (checked 17 Nov 2026 from the Hugging Face API) |
|---|---|
| License | CC-BY-4.0 |
| Rows | 135,634 votes (Apr–Jul 2025) |
| Models | 53 (Claude 4, Gemini 2.5, o3, GPT-4.1, DeepSeek, Qwen 3, Llama 4, …) |
| Votes | model_a 48.6k, model_b 49.8k, tie 21.5k, both_bad 15.8k |
| Columns | `model_a`, `model_b`, `winner`, `conversation_a`, `conversation_b`, `language`, `is_code`, `category_tag`, … |
| Size | about 3 GB (we read only the columns we need) |
| Language | 52% English (71k) |

**Pairs with enough votes (estimate):** if the arena paired models at random, about
324 model pairs would have 150+ votes and 139 would have 200+. The arena doesn't pair
uniformly, so exact counts are step 1. Frontier models are close in strength, so many
of these pairs should be close.

**Why it fits the study:** each vote is one person judging one prompt, which is exactly
the "one annotator per item" setup the simulation already uses.

### Ground truth

Each pair has only about 150–360 single-person votes, so its own human interval is
roughly ±7 points: too wide to call close pairs reliably. Two ground truths:

1. **Main:** a Bradley–Terry model fitted on all 136k votes (the standard arena method).
   This gives every pair a much tighter predicted win rate. It assumes model strength is
   transitive; that's a stated assumption.
2. **Robustness check:** the pair's own votes (the Phase 1 method). Report where the two disagree.

Ties and "both bad" count as 0.5.

## Pair selection

From pairs with ≥150 English, single-turn votes, choose 40 by human gap (Bradley–Terry):

| Group | Human gap | Pairs |
|---|---|---|
| Close | < 3 points | 15 |
| Small | 3–7 points | 15 |
| Clear | > 7 points | 10 (control, like Phase 1) |

Up to 150 items per pair = **6,000 items**. Each item is judged in both orders
(position-bias check), so **12,000 judge calls per judge**.

## Judges

| Judge | Why | Model ID |
|---|---|---|
| Claude Haiku 5.5 | Cheap; what many teams would use for bulk grading | `claude-haiku-5-5` |
| Claude Sonnet 5.5 | Stronger; tests whether a better judge exaggerates less | `claude-sonnet-5-5` |

- Same prompt for both, verdict only (`A`, `B`, `tie`) via structured outputs, thinking off,
  so the judge behaves like a typical cheap grading setup.
- Run through the **Message Batches API** (50% cheaper, no rate-limit juggling).
  Every result is cached to disk, so reruns cost nothing.
- **Self-preference:** the arena includes Claude models. Pairs with a Claude model are
  analysed separately (new RQ7 below). Optional: a judge from another provider, if available.

## Cost estimate

Assumptions: ~2,000 input tokens per call (prompt + two responses), 150–500 output
tokens. Step 2 measures real token counts on a sample before anything large runs.
Prices from the current Anthropic price list.

| Judge | Price per 1M tokens (in / out) | Standard | With Batch API (50% off) |
|---|---|---|---|
| Haiku 5.5 | $0.10 / $0.50 | $3–6 | **$2–3** |
| Sonnet 5.5 | $2 / $10 | $66–108 | **$33–54** |

**Total with Batches: about $35–60.** Cheaper option: Haiku on all 40 pairs, Sonnet on the 15 close pairs only (about $15–25).

## New research questions (on top of RQ1–RQ6)

- **RQ7 – Same-family bias:** does a Claude judge exaggerate gaps more when a Claude model is in the pair?
- **RQ8 – Judge strength:** does Sonnet exaggerate less than Haiku? Does it overclaim less on close pairs?
- **RQ9 – Position bias now:** how often do current judges flip when answers swap order (GPT-4 in 2023: 16%)?

Headline metric: **share of close pairs where a judge-only 200-item eval confidently picks a winner the human verdict doesn't support**, and how far 10/25/50 human labels bring it down.

## Code changes needed

1. `data.py`: Arena loader that reads only the needed columns, filters English and single-turn, and builds `PairData`.
2. `truth.py`: Bradley–Terry fit (standard library only, iterative updates) with bootstrap intervals.
3. `judge.py`: Anthropic SDK batch runner with disk cache, both orders, structured verdicts, token logging.
4. `stats.py`: t-based interval for small k (Phase 1 showed the normal interval is slightly too narrow at k = 10).
5. `run_phase2.py`: pair selection, judging, simulation grid, results CSVs.
6. Tests for each with synthetic data, so no API calls in tests.

## Timeline

| Dates | Work |
|---|---|
| 18 Nov | Exact pair counts, Bradley–Terry truth, pair selection (no API spend) |
| 19 Nov | Judge runner; 50-item pilot (under $1) to check prompt, token counts and parsing |
| 20–21 Nov | Full batch runs; simulation grid |
| 22–24 Nov | Analysis, RQ7–RQ9, figures |
| 25–27 Nov | Blog post + cheat sheet |
| 28–30 Nov | README, reproduction check, launch |

## What you need to do

1. **Get an Anthropic API key** at platform.claude.com (no key is set on this machine)
   and add about $60 of credit. Set it yourself in your terminal:
   `export ANTHROPIC_API_KEY=...` (don't paste it into chat).
2. **Approve spend at each step:** the pilot first (under $1), then the full runs.

## Risks

- **Arena voters are noisy and anonymous**, and judge prompts differ from what a crowd values
  (style, length). That's partly the point: the study measures agreement with real users.
- **Bradley–Terry truth** assumes transitivity; the per-pair check covers this.
- **Long responses** (reasoning models, code) may push tokens above 2,000 per call; the pilot measures this before the full run.
- **Responses are from 2025 models** judged by 2026 judges; state this in the write-up.
