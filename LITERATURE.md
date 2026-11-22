# Literature check (21 Nov 2026)

What earlier work already shows about each Phase 1 and Phase 2 finding, and how to phrase
the findings so the write-up neither overclaims nor undersells. Checked by web search plus the
paper's own text or abstract. Each reference says how far it was checked.

## Bottom line

- **The phenomenon isn't new.** Judges agreeing with people on big gaps but not small ones was
  shown for GPT-4 in 2023, position bias is known to be worst when answers are close in quality,
  correlated errors between LLMs are well documented, and PPI for LLM evals is published
  (including on Chatbot Arena data).
- **What this study adds is the decision view at practitioner scale.** On real close LMArena
  pairs, with four current open judges, it counts how often a 25-80 item eval names a winner
  people don't back, shows that more items make that worse, that different judges invent the
  *same* winner, and what 10/25/50 human labels buy. I found no paper that puts these together or
  states the "more items make it worse" and "shared invented winners" points in this form.
- **Safe framing for the post:** "Known in principle; here is what it looks like in numbers a team
  would actually face, and what to do about it." Avoid "new finding" or "first to show".

## Finding by finding

| Our finding | Closest prior work | Known? | How to say it |
|---|---|---|---|
| Judges match people on clear pairs and fail on close ones | Zheng et al. 2023, §4.2: GPT-4's agreement with humans rises from about 70% to nearly 100% as the win-rate gap between models grows | **Known** | "Confirms, with current judges and decisions rather than agreement" |
| Judges exaggerate gaps (Phase 1: median 1.31×; Phase 2: 10-13 points from a tie vs 1.5 for people) | Kargi & Salinas 2026 build judge-based Elo and correct its miscalibration; a search excerpt says judge-side rating gaps exceed human ones (not checked in the full text) | Partly known | "Consistent with reports that judge-based ratings are overconfident" |
| More items make a judge-only eval more confidently wrong (18-26% at 25 items, 38-46% at 80) | Textbook bias-variance: a biased estimator's interval shrinks around the wrong value. It is the reason PPI exists (Angelopoulos et al. 2023 show intervals that treat predictions as truth lose coverage). Lee et al. 2025 note the naive judge estimate is biased | Known in theory; not found shown this way for LLM judges on close pairs | "Known in theory; here is how fast it happens" |
| Different judges invent the same winner (4 of 15 close pairs, 3+ judges) | Kim et al. 2025 (ICML): errors are correlated across 350+ LLMs, more so for larger models and the same provider. Kohli 2026: 9 judges give about 2 independent votes. Han & Kim 2026: judges' disagreements with the human vote on Arena battles co-occur more than chance | **Shared errors known**; shared invented winners on human ties not found stated | "In line with correlated-error findings; on close pairs it means a panel agrees on a winner people don't see." Qualifies Verga et al. 2024 (a panel of judges reduces bias) for close calls. Our evidence rests on 4 pairs, so say so |
| A few human labels fix the false confidence; PPI++ ≈ the labels alone (best judge weight 0.06-0.29) | PPI: Angelopoulos et al. 2023; PPI++: Angelopoulos, Duchi & Zrnic 2023. Boyeau et al. 2024 (ICML 2025): up to 50% more effective human labels on Chatbot Arena with GPT-4. Dorner, Nastl & Hardt (ICLR 2025): when the judge is no better than the models it judges, debiasing can at most halve the labels needed; real savings were smaller | **Known**; our result fits Dorner et al. | "As theory predicts, when the judge barely agrees with people item by item (kappa 0.05-0.13), it adds little on top of the labels" |
| A correction measured on another pair doesn't transfer | Fiedler 2026: sharing calibration across the models being compared can make a comparison point the wrong way with high apparent confidence. Li 2026 (in STUDY.md): judge calibration doesn't transfer across agent versions | Known in related settings | "Matches recent warnings about reusing calibration" |
| Position bias: 26-51% of verdicts change on swap, mostly toward the first answer | Zheng et al. 2023, Table 2: on two GPT-3.5 answers to the same question (near-identical quality), GPT-4 kept its verdict 65% of the time (35% changed; 30% toward the first answer). Shi et al. (AACL-IJCNLP 2025): position bias depends strongly on the quality gap, worst for close answers | **Known** | "In line with GPT-4's 35% on near-identical answers", not "worse than 2023" |
| No extra same-family preference (RQ7, exploratory) | Panickssery, Bowman & Feng (NeurIPS 2024): LLM judges recognise and favour their own outputs | Self-preference known; ours is a null on family (not self) and small samples | "No sign of family preference beyond other judges, on 4-9 pairs" |
| Agreement looks fine while decisions are wrong (RQ6) | Han & Kim 2026: median kappa 0.22 with Arena votes and unstable rankings across judges | Partly known | Cite alongside |

## Correction made

- **GPT-4 position bias.** PHASE2.md compared our 26-51% with "GPT-4 on MT-Bench in 2023: 16%".
  The 16% is our own Phase 1 count of "inconsistent" verdicts in the MT-Bench dataset's bundled
  GPT-4 judgments (380 of 2,400, over all pairs; see FINDINGS.md). The paper itself (Table 2)
  reports 35% changed for GPT-4 on near-identical answers, the fairer comparison for close pairs.
  PHASE2.md now gives both.

## Closest recent work to watch

- **Han & Kim, "Swapping the Judge Changes the Ranking" (arXiv 2611.04090, 3 Nov 2026,
  preregistered).** Five judges re-judge 4,000 Chatbot Arena battles with human votes: low
  agreement with people, rankings that change with the judge, and correlated disagreements with
  the human vote. Same data family as ours; it looks at rankings and agreement, not at confident
  winner calls on close pairs or human-label budgets. Cite it and say how ours differs.
- **Kohli, "Nine Judges, Two Effective Votes" (arXiv 2605.29800, May 2026).** Panels lose most
  of their nominal independence to shared mistakes; the best single judge matched the panel.

## References

Checked against the paper's text (T), its abstract page (A), or only a search result (S).

- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic (2023). Prediction-powered inference. *Science*. (S; already cited in STUDY.md)
- Angelopoulos, Duchi & Zrnic (2023). PPI++: Efficient prediction-powered inference. arXiv [2311.01453](https://arxiv.org/abs/2311.01453). (S)
- Boyeau, Angelopoulos, Yosef, Malik & Jordan (2024). AutoEval done right: using synthetic data for model evaluation. ICML 2025. arXiv [2403.07008](https://arxiv.org/abs/2403.07008). (A)
- Chatzi, Straitouri, Thejaswi & Gomez-Rodriguez (2024). Prediction-powered ranking of large language models. NeurIPS 2024. arXiv [2402.17826](https://arxiv.org/abs/2402.17826). (S)
- Chiang et al. (2024). Chatbot Arena: an open platform for evaluating LLMs by human preference. ICML 2024. arXiv [2403.04132](https://arxiv.org/abs/2403.04132). (S; the data source)
- Dorner, Nastl & Hardt (2025). Limits to scalable evaluation at the frontier: LLM as judge won't beat twice the data. ICLR 2025. arXiv [2410.13341](https://arxiv.org/abs/2410.13341). (S)
- Fiedler (2026). Bias and uncertainty in LLM-as-a-judge estimation. arXiv [2605.06939](https://arxiv.org/abs/2605.06939). (S)
- Han & Kim (2026). Swapping the judge changes the ranking: a preregistered study of cross-provider replication in LLM-judged system comparisons. arXiv [2611.04090](https://arxiv.org/abs/2611.04090). (A)
- Kargi & Salinas (2026). From uncertain judgments to calibrated rankings: conformal Elo estimation for LLM evaluation. NeurIPS 2026. arXiv [2606.13221](https://arxiv.org/abs/2606.13221). (A)
- Kim, Garg, Peng & Garg (2025). Correlated errors in large language models. ICML 2025. arXiv [2506.07962](https://arxiv.org/abs/2506.07962). (A)
- Kohli (2026). Nine judges, two effective votes: correlated errors undermine LLM evaluation panels. arXiv [2605.29800](https://arxiv.org/abs/2605.29800). (A)
- Lee et al. (2025). How to correctly report LLM-as-a-judge evaluations. arXiv [2511.21140](https://arxiv.org/abs/2511.21140). (S)
- Panickssery, Bowman & Feng (2024). LLM evaluators recognize and favor their own generations. NeurIPS 2024. arXiv [2404.13076](https://arxiv.org/abs/2404.13076). (S)
- Shi, Ma, Liang, Diao, Ma & Vosoughi (2024). Judging the judges: a systematic study of position bias in LLM-as-a-judge. AACL-IJCNLP 2025. arXiv [2406.07791](https://arxiv.org/abs/2406.07791). (A, partly T)
- Verga et al. (2024). Replacing judges with juries: evaluating LLM generations with a panel of diverse models. arXiv [2404.18796](https://arxiv.org/abs/2404.18796). (S)
- Zheng et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. arXiv [2306.05685](https://arxiv.org/abs/2306.05685). (T: §3.3 Table 2, §4.2)

The five 2024-2026 papers in STUDY.md's "gap in existing work" table (Miller; Signal and Noise;
Anand; Li; Reliability without Validity) still apply and should be cited too.

## Not checked yet

- The full texts of Dorner et al., Boyeau et al. and Kargi & Salinas (only abstracts or search
  excerpts were read). Read the relevant sections before quoting their numbers in the post.
- Whether anyone has published "more items make a judge-only eval more confidently wrong" with
  numbers for LLM judges. Searches found the theory and related warnings, not this result.
