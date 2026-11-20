# Phase 2 plan: many close comparisons, current judges

**Status:** step 1 done on 17 Nov 2026: 40 pairs chosen, see [Step 1 results](#step-1-results-17-nov-2026).
On 18 Nov Phase 2 switched to a [lean design](#lean-design-18-nov): about 1,500 calls per judge
and free judges from 8B to about 1 trillion parameters, so it costs nothing. **Four judges
finished between 18 and 20 Nov** (see [Judge runs](#judge-runs-1820-nov)); the optional Gemini
judge hasn't run (no key yet). Analysis started on 20 Nov.
Phase 1 results: [FINDINGS.md](FINDINGS.md).

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
| Size | 1.61 GB in 7 parquet files. Step 1 reads only the columns it needs: 7.4 MB. The prompts and answers for the chosen votes mean reading 692 MB, because those votes are spread over all 140 row groups; only the chosen rows are kept |
| Language | 52% English (71k) |

**Pairs with enough votes (counted in step 1):** keeping English, single-turn votes
leaves 60,956. Of the pairs they cover, 113 have 100+ votes, 11 have 150+ and none has
200+. (The earlier estimate of 324 pairs with 150+ assumed all 136k votes and random pairing.)

**Why it fits the study:** each vote is one person judging one prompt, which is exactly
the "one annotator per item" setup the simulation already uses.

### Ground truth

Each chosen pair has only 100–154 single-person votes, so its own human interval is
about ±7.5 points: too wide to call close pairs reliably. Two ground truths:

1. **Main:** a Bradley–Terry model (the standard arena method) fitted on the 60,956
   English single-turn votes, the same kind of prompt the judges will see. Pooling every
   vote makes each pair's interval about 4 points wide instead of 15. It assumes model
   strength is transitive; that's a stated assumption.
2. **Robustness check:** the pair's own votes (the Phase 1 method). Report where the two disagree.

Ties and "both bad" count as 0.5.

## Pair selection

From pairs with ≥100 English, single-turn votes, choose 40 by human gap (Bradley–Terry).
Within each group, pairs are spread evenly across the gap range.

| Group | Human gap | Pairs |
|---|---|---|
| Close | < 3 points | 15 |
| Small | 3–7 points | 15 |
| Clear | > 7 points | 10 (control, like Phase 1) |

Up to 150 items per pair: **4,601 items**, listed in
[`results/phase2/items.csv`](results/phase2/items.csv). The [lean design](#lean-design-18-nov)
judges 1,400 of them, from 25 of the 40 pairs.

## Step 1 results (17 Nov 2026)

Run: `python -m wrong_winner.select_pairs` (200 bootstrap repetitions, seed 0). No LLM calls.
Outputs: [`results/phase2/candidates.csv`](results/phase2/candidates.csv) (all 113 pairs
with 100+ votes) and [`results/phase2/pairs.csv`](results/phase2/pairs.csv) (the 40 chosen).

| Stage | Count |
|---|---|
| All votes | 135,634 |
| English, single-turn | 60,956 (53 models, 1,202 pairs) |
| Pairs with 100+ of those votes | 113: 15 close, 36 small, 62 clear |
| Chosen | 40 pairs, 25 models, 4,601 items |

| Group | Pairs | Gap (Bradley–Terry) | Votes per pair | Interval includes 50% | Own votes alone pick a winner | With a Claude model |
|---|---|---|---|---|---|---|
| Close | 15 (all available) | 0.3–2.0 points | 100–154 | 12 | 1 | 4 |
| Small | 15 of 36 | 3.0–6.9 points | 100–145 | 0 | 5 | 4 |
| Clear | 10 of 62 | 7.2–24.1 points | 100–148 | 0 | 8 | 6 |

- **Changed from the plan: 100+ votes, not 150+.** Only 11 pairs reach 150. Phase 1's
  pairs had 99–146 items each, so 100+ is comparable.
- **Changed from the plan: Bradley–Terry uses English single-turn votes, not all 136k.**
  On the 113 candidate pairs, the two fits differ by a median of 1.0 point (at most 3.6).
- **12 of the 15 close pairs are ties** in Phase 1's sense (the interval includes 50%).
  The other 3 have small but real gaps (1.9–2.0 points).
- **The two truths point different ways in 10 of the 40 pairs** (5 close, 5 small), but
  in none of the 10 do the pair's own votes exclude 50%, so they never confidently
  disagree. One close pair to watch: claude-sonnet-4 vs command-a, where Bradley–Terry
  gives 48.7% but the pair's own 129 votes give 40.3% [33.2, 47.4]. Step 3 reports
  results under both truths.
- **Some models appear often:** qwen3-235b-a22b-no-thinking is in 11 of the 40 pairs and
  gemini-2.5-flash in 8. Step 3 checks that the findings hold with each model left out.

## Lean design (18 Nov)

The first plan judged all 4,601 items in both answer orders: about 9,200 calls per judge, which
meant paying for Claude (about $30–46) or running one small local judge for about 36 hours.
To show the effect and its direction, far fewer calls are enough, and the saving buys a ladder
of bigger judges at no cost.

- **Spend the calls where the claim is:** the claim is about close pairs, so they get most of
  the items. The clear pairs are only a control (judges should agree with humans there).
- **One answer order per item,** picked by a fixed coin flip, so position bias evens out.
  A 100-item subset is shown both ways to measure that bias.
- **Items up to 4,000 tokens** (dataset count plus instructions). Every close pair still has
  80+ such items, and the cap keeps calls quick and inside free-tier limits.
- **Grow only where needed:** verdicts are saved, so items can be added later to a pair that
  comes out borderline, or the small-gap group added back (about 300 calls per judge).

| Part | Pairs | Items | Calls per judge |
|---|---|---|---|
| Close pairs (the claim) | 15 | 80 each | 1,200 |
| Clear pairs (control) | 10 | 20 each | 200 |
| Position-bias check | close | 100 of the above, shown both ways | +100 |
| **Total** | **25** | **1,400** | **1,500** (2.75M input tokens) |

Run: `python -m wrong_winner.judge --make-plan` (seed 0) writes
[`results/phase2/plan.csv`](results/phase2/plan.csv), which every judge follows. Items with
an image or an empty answer are left out (3 rows have an empty answer). With 80 items, a close
pair's judge win rate is known to about ±10 points; pooled over the 15 close pairs, about ±2.5.
That supports conclusions per group and direction per pair, not strong claims about single pairs.

## Judges: a free ladder from 8B to about 1 trillion parameters

A Claude Pro subscription doesn't cover API use, and scripting Claude Code at this scale would
run into the plan's usage limits, so the judges are free models:

| Judge | Size | Runs on | Thinking | Same-family pairs (of 25) |
|---|---|---|---|---|
| Qwen3 8B, 4-bit (`qwen3:8b-q4_K_M`) | 8B | This Mac, through Ollama | Off | Qwen: 8 |
| Gemma 4 31B (`google/gemma-4-31b-it`) | 31B | NVIDIA's free endpoints | None | Gemini: 9 |
| Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`) | 550B (55B active) | NVIDIA's free endpoints | Off | None |
| Mistral Large 4 (`mistralai/mistral-large-4-1t-a52b`) | About 1T (52B active) | NVIDIA's free endpoints | **On; can't be switched off there** | Mistral: 4 |
| Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | Not published | Google's free tier | Minimal | Gemini: 9 |

- **Changed on 18 Nov:** DeepSeek V4.1 Flash, in the first version of this table, isn't
  available to this NVIDIA key (it isn't in the key's model list). Nemotron 3 Ultra replaces it,
  which keeps the size ladder (8B, 31B, 550B, about 1T) but loses the 2 DeepSeek pairs for RQ7.
- Same instructions for every judge, model names hidden, verdict only (`A`, `B` or `tie`) as
  JSON, so the judges behave like a typical cheap grading setup.
- Qwen: thinking off, temperature 0, verdict held to a JSON schema, 16k-token context window.
- NVIDIA models: temperature 0 and `enable_thinking: false` through NVIDIA's OpenAI-style chat
  API. Nemotron honours it (8 output tokens per verdict); Gemma 4 doesn't think anyway.
  **Mistral Large 4 reasons on every call** on NVIDIA's endpoint (500–1,000 reasoning tokens;
  `reasoning_effort: "none"` is refused and `"low"` still reasons), so it is the one reasoning
  judge and is labelled that way in every result. Its output limit is 4,096 tokens to leave room
  (raised later for the calls that needed more; see [Judge runs](#judge-runs-1820-nov)).
  NVIDIA's docs don't mention JSON schemas, so replies are read leniently (JSON, a code fence,
  or a bare A / B / tie, after any reasoning); unreadable replies are retried on the next run.
- Gemini: thinking at its minimum (Flash-Lite can't switch it off fully), default temperature,
  through Google's Interactions API, verdict held to a JSON schema.
- Every verdict is saved the moment it arrives (`results/phase2/judgments/<judge>.jsonl`), so a
  run can stop at any time and resume without repeating a call. Calls run in a fixed shuffled
  order, so even a partial run covers every pair.
- **NVIDIA's free tier:** about 40 requests a minute and 10,000 a day. NVIDIA doesn't publish the
  list of free models; the key's own model list and a 5-call test per judge (18 Nov) confirmed
  the three used here. Calls wait in NVIDIA's queue (Gemma 4 about 50–115 seconds, Mistral
  Large 4 about 45–470 seconds, Nemotron about 1 second), so the runner keeps many calls in
  flight at once (16–24 workers) and caps each judge at 12 requests a minute, about 36 in all.
  Needs a free NVIDIA Developer account with phone verification. NVIDIA records prompts and
  answers and may use them to improve its models.
- **Gemini's free tier:** limits are per Google Cloud project and shown only in AI Studio.
  Third-party readings in October 2026 put Flash-Lite at about 500 requests a day. Google may use
  free-tier prompts and answers to improve its products, and people may review them.
- The arena data is already public (CC-BY-4.0), so nothing private goes to either provider.
- The runner paces hosted calls (thread-safe, across all workers), retries short-term errors
  and stops cleanly at a daily limit; rerunning the same command carries on.

## Run time and cost

**Cost: $0.** Measured on 18 Nov:

| Judge | Time for 1,500 calls |
|---|---|
| Qwen3 8B (this Mac) | About 4.5 hours (about 10.5 seconds per call) |
| The three NVIDIA judges, side by side | About 2–2.5 hours, set by the 12-a-minute cap per judge |
| Gemini 3.5 Flash-Lite | About 3 days at about 500 a day |

Pilot (20 local calls) and 5-call tests per NVIDIA judge: every verdict readable, except Mistral
replies cut off at the first test's 1,024-token limit (now 4,096).

## Judge runs (18–20 Nov)

| Judge | Readable verdicts (of 1,500) | Missing |
|---|---|---|
| Qwen3 8B | 1,500 (plus 15 pilot calls, not used) | 0 |
| Gemma 4 31B | 1,498 | 2 |
| Nemotron 3 Ultra 550B | 1,500 | 0 |
| Mistral Large 4 (reasoning) | 1,497 | 3 |
| Gemini 3.5 Flash-Lite | Not run (no key yet) | |

- **NVIDIA's free tier stopped runs with streaks of 429 errors** (rate limited after 8 retries).
  Rerunning straight away worked; waiting an hour wasn't needed.
- **Mistral Large 4 sometimes reasons far longer than 4,096 tokens.** 25 calls needed more (up to
  14,521). Non-streamed replies that long never arrive from NVIDIA's free tier: the connection
  times out or is dropped. Two runner options fixed it (19–20 Nov): `--max-tokens` raises the
  limit and scales the timeout, and `--stream` asks for a streamed reply, which does arrive.
  Mistral's last 5 calls ran with `--max-tokens 32768 --stream`: 2 returned a verdict (after
  17,474 and 23,350 tokens); **3 used all 32,768 tokens without deciding** and are recorded as
  no verdict. All 3 are close pairs that people rated a tie.
- **The Mac sleeps with the lid closed,** even with the runner's "don't sleep" request, which
  slowed the overnight run. Keep the lid open during long runs.
- Missing verdicts are left out of that judge's analysis (each judge's pair then has 79 or 19
  items instead of 80 or 20), not counted as ties.

## First results (20 Nov)

Run: `python -m wrong_winner.run_phase2` (2,000 repetitions, seed 0, about 3 minutes). Outputs in
[`results/phase2/analysis/`](results/phase2/analysis/). Main truth: Bradley-Terry. Every result
below also holds with the pair's own votes as the truth.

**How the analysis is set up**
- Each item is scored once, in its planned answer order, as a team making one call per item would.
  The 100 items shown both ways feed RQ9 only.
- All intervals use Student's t.
- **The real eval is 80 items** (every judged item of a close pair). Simulated evals of 200 items
  have to resample those 80, which treats them as the whole population and inflates confident
  calls: even humans-only evals then back an unsupported winner on 27% of runs. So the headline
  uses 80 items, and n = 200 is reported only as an extrapolation.

**Close pairs (15): judges invent winners; humans don't**

| Judge | Pairs where the 80-item call isn't backed by humans | Simulated 80-item evals making such a call | PPI, k = 10 / 25 / 50 human labels | PPI++, k = 10 / 25 / 50 | Correction reused from another pair (k = 50) |
|---|---|---|---|---|---|
| Qwen3 8B | 7 of 15 | 46.4% | 4.8% / 5.4% / 6.2% | 6.6% / 8.0% / 11.0% | 33.7% |
| Gemma 4 31B | 5 of 15 | 41.5% | 4.9% / 5.5% / 6.0% | 6.9% / 8.2% / 10.9% | 32.0% |
| Nemotron 3 Ultra 550B | 6 of 15 | 43.3% | 4.7% / 5.2% / 5.9% | 6.6% / 8.0% / 10.9% | 38.1% |
| Mistral Large 4 (reasoning) | 5 of 15 | 38.3% | 4.7% / 5.4% / 5.9% | 6.8% / 8.2% / 11.0% | 23.9% |
| Humans only (the same 80 votes) | 0 of 15 | 14.2% | | The k labels alone: 4.6% / 7.8% / 10.3% | |

(Every column after the second is from simulated evals of 80 items.)

- **No judge ever picked the opposite winner** to a pair humans can separate. Every unsupported
  call is a winner invented on a pair people see as a tie, as in Phase 1.
- **Judges amplify:** on close pairs the judges' win rates sit 10-13 points from a tie on average.
  Bradley-Terry puts humans 1.5 points from a tie, and the same 80 votes 4.2 points.
- **The judges agree with each other, not with people.** All four call o3 over ChatGPT-4o,
  Grok 3 over ChatGPT-4o, and DeepSeek R1 over Gemini 2.5 Flash; people see ties. A panel of
  these judges would make the same call. Item-level agreement with the single human vote is
  close to chance (Cohen's kappa 0.05-0.13).
- **Plain PPI brings unsupported calls down to about 5%, but mostly by saying "can't tell"**
  (about 93-95% of evals make no call, which is the right answer on a tie). Because the judges barely
  agree with the voters item by item, the correction adds noise. On clear pairs, plain PPI with k
  human labels finds the winner *less* often than those k labels on their own (k = 50: 54-61%
  against 75%; simulated 80-item evals resampled from 20 items).
- **Power-tuned PPI (PPI++) fixes that, and shows how little the judge adds.** PPI++ weights the
  judge by how well it tracks the humans (Angelopoulos, Duchi and Zrnic, 2023). With these judges
  it does as well as the human labels alone, but hardly better:
  - clear pairs, finding the winner at k = 10 / 25 / 50: 37-40% / 63-65% / 76-77%, against
    35% / 62% / 75% for the labels alone;
  - close pairs, unsupported calls: 6.6-6.9% / 8.0-8.2% / 10.9-11.0%, against 4.6% / 7.8% / 10.3%.
    At k = 10, PPI++ is slightly less cautious than the labels alone, a known cost of estimating
    the weight from only 10 items.

  The best weight on the judge, given many more judge than human scores, averages **0.06-0.29**
  (1 = trust the judge fully, 0 = ignore it). On these pairs, the human labels do the work;
  the judge adds little.
- **Unsupported calls rise with k for every method that targets the human mean,** including the
  labels alone (4.6% to 10.3%). That's a property of the simulation, not of PPI: it treats each
  pair's 80 votes as the population, and for a few close pairs those votes lean further from a
  tie than the Bradley-Terry truth (claude-sonnet-4 vs command-a: 41% of the 80 votes).
- **A correction measured on another pair doesn't transfer** (24-38% unsupported), as in Phase 1.

**Clear pairs (10, control):** every judge's 20-item win rate is on the humans' side for all
10 pairs. Calls on all 20 items: 8-10 correct, none wrong.

**RQ8 judge size:** no clear trend from 8B to about 1T (7, 5, 6 and 5 of 15 close pairs). The
reasoning judge is lowest in the simulated evals (38%), but that is one judge.

**RQ9 position bias** (100 close items shown both ways): judges change their verdict when the
answers swap on **26-51%** of items (GPT-4 on MT-Bench in 2023: 16%, though that was over all
pairs, not close ones). Mostly toward the first answer: Nemotron chose the first answer both
times on 41 of 100 items. Over all calls, the first answer wins 58.5-68.5% of decisive verdicts.

| Judge | Verdict changes on swap | First answer both times | Second answer both times | One tie | First answer wins (decisive calls) |
|---|---|---|---|---|---|
| Qwen3 8B | 51 of 100 | 25 | 4 | 22 | 66.5% |
| Gemma 4 31B | 26 of 100 | 13 | 1 | 12 | 58.5% |
| Nemotron 3 Ultra | 45 of 100 | 41 | 0 | 4 | 68.5% |
| Mistral Large 4 | 36 of 100 | 32 | 1 | 3 | 61.8% |

**RQ7 same-family preference (exploratory):** no judge favours its own family's models more than
the other judges do. Points toward the family model, compared with the other judges on the same
items (95% interval): Qwen +1.6 [-2.3, +5.5] over 8 pairs, Gemma +2.3 [-1.2, +5.7] over 9 pairs,
Mistral -2.9 [-8.5, +2.7] over 4 pairs. Qwen3 8B does rate Qwen models 8.4 points above the
voters, but the other judges do much the same.

**Still to do:**
- figures (needs matplotlib, not installed yet);
- leave-one-model-out check: qwen3-235b is in 8 of the 25 judged pairs;
- the Phase 2 findings write-up;
- optional Gemini judge.

If a Claude data point is ever wanted, the lean design makes it cheap: about $0.15 for Claude
Haiku 5.5 and about $3 for Sonnet 5.5 through the Batch API. Optional.

## New research questions (on top of RQ1–RQ6)

- **RQ7 – Same-family bias:** does a judge exaggerate gaps more when a model from its own
  family is in the pair? Pair counts per family are in the table above; they're small, so this
  is exploratory.
- **RQ8 – Judge size:** across the ladder from 8B to about 1T, does a bigger judge overclaim
  less on the close pairs? Mistral Large 4 also reasons, so its point mixes size and reasoning.
- **RQ9 – Position bias now:** on the 100 items shown both ways, how often does each judge
  (except Gemini, which sees one order) change its verdict when the answers swap (GPT-4 in
  2023: 16%)? In the pilot the 8B judge kept the same verdict for only 3 of 10 items.

Headline metric: **share of close pairs where a judge-only 200-item eval confidently picks a winner the human verdict doesn't support**, per judge, and how far 10/25/50 human labels bring it down.

## Code changes needed

1. ✅ `arena.py`: Arena loader that reads only the needed columns and filters English, single-turn votes (step 1).
2. ✅ `truth.py`: Bradley–Terry fit (standard library only, iterative updates) with bootstrap intervals (step 1).
   ✅ `select_pairs.py`: pair selection and token counts (step 1); also writes `items.csv`.
3. ✅ `judge.py`: the lean plan (`--make-plan`) and one runner for every judge (Ollama on this
   Mac, NVIDIA's and Google's free tiers), standard library only for HTTP. It runs calls in
   parallel workers with shared pacing, saves each verdict as it arrives, retries unreadable
   ones on the next run, logs tokens, and stops cleanly at a daily limit. `arena.fetch_texts`
   gets the prompts and answers for the chosen votes (18 Nov). NVIDIA-only `--max-tokens` and
   `--stream` options for long reasoning replies (20 Nov).
4. ✅ `stats.py`: t-based interval for small k (Phase 1 showed the normal interval is slightly too
   narrow at k = 10). Standard library only; `mean_ci(..., t=True)` and `ppi_mean(..., t=True)`.
   Phase 1 keeps the normal interval, and its results are unchanged (checked byte for byte, 20 Nov).
   ✅ Power-tuned PPI (PPI++): `ppi_lambda` picks the weight on the judge and `ppi_mean(..., lam=)`
   uses it; the simulation's `ppi_tuned` method runs it. Phase 1 still unchanged (20 Nov).
5. ✅ `run_phase2.py`: full-data verdicts, simulation grid, RQ7 and RQ9, and a summary, written to
   `results/phase2/analysis/` (20 Nov). `simulate.py` now separates running the evals
   (`simulate_decisions`) from scoring them (`tally`), so each run is scored against both truths.
6. ✅ Tests with fake judges and canned responses, so tests make no network calls
   (94 tests in all on 20 Nov).

## Timeline

| Dates | Work |
|---|---|
| 17 Nov ✅ | Exact pair counts, Bradley–Terry truth, pair selection |
| 18 Nov ✅ | Free judges, lean plan, runner and tests; prompts and answers fetched (692 MB read); local pilot; local and NVIDIA runs started |
| 18–20 Nov ✅ | Qwen, Gemma and Nemotron finished on 18 Nov; Mistral on 20 Nov after streaming was added |
| 20 Nov ✅ | Analysis code and first results (see [First results](#first-results-20-nov)) |
| 19–21 Nov | Gemini about 500 calls a day, once its key is set (optional) |
| 21–24 Nov | Simulation grid, analysis, RQ7–RQ9, figures |
| 25–27 Nov | Blog post + cheat sheet |
| 28–30 Nov | README, reproduction check, launch |

## What you need to do

1. ✅ **NVIDIA key** (18 Nov): saved as `export NVIDIA_API_KEY=...` in `~/.zshrc`, which the
   runs load. Never paste a key into chat.
2. **Gemini key (optional):** create one in Google AI Studio and add
   `export GEMINI_API_KEY=...` to `~/.zshrc` the same way. AI Studio's rate-limit page shows your
   daily limit for `gemini-3.5-flash-lite`.
3. **Keep the Mac awake and plugged in** while the local judge runs (the runner also asks macOS
   not to sleep). Closing other apps speeds it up.

## Risks

- **Arena voters are noisy and anonymous**, and judge prompts differ from what a crowd values
  (style, length). That's partly the point: the study measures agreement with real users.
- **Bradley–Terry truth** assumes transitivity; the per-pair check covers this (see step 1 results).
- **80 items per close pair** gives about ±10 points per pair. Conclusions are pooled per group;
  borderline pairs can get more items later.
- **Leaving out answers over 4,000 tokens** drops some long code and reasoning answers. The same
  items are used for every judge and for the human side.
- **Free tiers can change without notice:** NVIDIA's free model list and both providers' limits
  aren't officially published. Each judge starts with a 5-call test, the runner stops cleanly
  and resumes, and any judge that becomes unavailable is reported as missing, not replaced quietly
  (as with DeepSeek on 18 Nov).
- **Mistral Large 4 reasons and the others don't,** so it can't cleanly answer "does size help?"
  on its own; it's reported separately as the reasoning judge.
- **Gemini's Interactions API is new:** the request fields come from Google's docs as of 18 Nov.
  The first calls check them.
- **A small 4-bit judge** is weaker than Phase 1's GPT-4. That's part of the question (RQ8),
  and it matches what teams on a budget actually use.
- **Responses are from 2025 models** judged by 2026 judges; state this in the write-up.
