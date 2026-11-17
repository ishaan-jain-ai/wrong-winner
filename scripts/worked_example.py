"""Numbers behind the step-by-step example in the plain-language write-up.

Loads MT-Bench, then prints the tied pair's scores, example questions, and one
simulated 100-question test of each kind (judge only, people only, PPI, borrowed
correction) for seeds 0-4.

Run from the repo root:
    python scripts/worked_example.py
"""
import random
import sys
from collections import Counter
from statistics import fmean

sys.path.insert(0, "src")

from datasets import load_dataset  # noqa: E402

from wrong_winner.data import MTBENCH_REPO, build_pairs  # noqa: E402
from wrong_winner.simulate import (  # noqa: E402
    _one_eval,
    ground_truth,
    judge_full_data_verdict,
)
from wrong_winner.stats import decide, mean_ci  # noqa: E402

ds = load_dataset(MTBENCH_REPO)
keep = ["question_id", "model_a", "model_b", "winner", "turn"]
human_raw = list(ds["human"])
judge_raw = list(ds["gpt4_pair"])
pairs = build_pairs(
    [{c: r[c] for c in keep} for r in human_raw],
    [{c: r[c] for c in keep} for r in judge_raw],
)
by_name = {p.name: p for p in pairs}

def fmt(e):
    return f"{e.value:.3f} [{e.low:.3f}, {e.high:.3f}]"

TARGET = "claude-v1 vs gpt-3.5-turbo"
SOURCE = "claude-v1 vs gpt-4"
CLEAR = "gpt-3.5-turbo vs gpt-4"

print("columns human:", ds["human"].column_names)
print("columns judge:", ds["gpt4_pair"].column_names)

for name in (TARGET, SOURCE, CLEAR):
    p = by_name[name]
    h, j = ground_truth(p), judge_full_data_verdict(p)
    shared = p.shared_items()
    drift = fmean(fmean(p.judge[i]) - fmean(p.human[i]) for i in shared)
    print(f"\n== {name}")
    print("  human items", len(p.human), "human votes", sum(map(len, p.human.values())),
          "judge items", len(p.judge), "shared", len(shared))
    print("  people:", fmt(h.estimate), "->", h.winner or "too_close")
    print("  judge :", fmt(j.estimate), "->", j.winner or "too_close")
    print("  mean(judge - people) on shared items:", round(drift, 3))
    print("  judge item scores:", Counter(s for v in p.judge.values() for s in v))
    print("  people vote scores:", Counter(s for v in p.human.values() for s in v))

# Judge raw winners on the tied pair (to count 'tie (inconsistent)')
def in_pair(r, a, b):
    return {r["model_a"], r["model_b"]} == {a, b}

jr = [r for r in judge_raw if in_pair(r, "claude-v1", "gpt-3.5-turbo")]
print("\njudge raw winners on tied pair:", Counter(r["winner"] for r in jr))
hr = [r for r in human_raw if in_pair(r, "claude-v1", "gpt-3.5-turbo")]
print("people raw winners on tied pair:", Counter(r["winner"] for r in hr))

# Example items: people votes vs judge verdict, from claude-v1's point of view.
p = by_name[TARGET]
def label(s):
    return "claude-v1" if s == 1 else "gpt-3.5-turbo" if s == 0 else "tie"

def question_text(qid, turn):
    for r in human_raw:
        if r["question_id"] == qid:
            users = [m["content"] for m in r["conversation_a"] if m["role"] == "user"]
            return users[turn - 1]
    return "?"

print("\nExample items (score from claude-v1's view):")
cats = {"judge A, people B": [], "judge A, people split/tie": [], "agree": [], "judge B, people A": []}
for item in p.shared_items():
    hm, jm = fmean(p.human[item]), fmean(p.judge[item])
    if jm == 1 and hm == 0:
        cats["judge A, people B"].append(item)
    elif jm == 1 and hm == 0.5:
        cats["judge A, people split/tie"].append(item)
    elif jm == hm and jm in (0, 1):
        cats["agree"].append(item)
    elif jm == 0 and hm == 1:
        cats["judge B, people A"].append(item)
for k, items in cats.items():
    print(f"  {k}: {len(items)} items")
    for item in items[:3]:
        q = question_text(*item).replace("\n", " ")
        print(f"    {item} people={[label(s) for s in p.human[item]]} judge={[label(s) for s in p.judge[item]]}")
        print(f"      Q: {q[:160]}")

# Raw judge winners for the first examples (to show 'tie (inconsistent)')
print("\nRaw judge rows for a few items:")
for item in (cats["judge A, people split/tie"][:2] + cats["judge A, people B"][:2]):
    for r in jr:
        if (r["question_id"], int(r["turn"])) == item:
            print("  ", item, r["model_a"], "vs", r["model_b"], "->", r["winner"])

# Coin-style error bars example: 60 wins out of 100.
e = mean_ci([1.0] * 60 + [0.0] * 40)
print("\n60/100 wins:", fmt(e), decide(e))
e = mean_ci([1.0] * 55 + [0.0] * 45)
print("55/100 wins:", fmt(e), decide(e))

# One simulated team test of each kind (first draw of seed 0..), n=100.
pool = p.shared_items()
for seed in range(5):
    rng = random.Random(seed)
    sample = rng.choices(pool, k=100)
    rng = random.Random(seed)
    est = _one_eval(p, pool, 100, 0, "judge", rng, 0.95, None)
    print(f"\nseed {seed} judge-only n=100:", fmt(est), decide(est))
    rng = random.Random(seed)
    est = _one_eval(p, pool, 100, 100, "human_only", rng, 0.95, None)
    print(f"seed {seed} people-only n=100:", fmt(est), decide(est))

    # PPI n=100, k=10, broken into its parts (same draw order as _one_eval).
    rng = random.Random(seed)
    sample = rng.choices(pool, k=100)
    judge_scores = [rng.choice(p.judge[i]) for i in sample]
    human_scores = [rng.choice(p.human[i]) for i in sample[:10]]
    base = fmean(judge_scores[10:])
    jl, hl = fmean(judge_scores[:10]), fmean(human_scores)
    rng = random.Random(seed)
    est = _one_eval(p, pool, 100, 10, "ppi", rng, 0.95, None)
    print(f"seed {seed} PPI n=100 k=10: judge on 90 unlabelled={base:.3f}; on the 10 labelled judge={jl:.3f} people={hl:.3f}"
          f" rectifier={hl - jl:+.3f} -> {fmt(est)} {decide(est)}")
    print("   labelled judge:", judge_scores[:10])
    print("   labelled people:", human_scores)

    rng = random.Random(seed)
    est = _one_eval(p, pool, 100, 50, "transport", rng, 0.95, by_name[SOURCE])
    print(f"seed {seed} transport n=100 k=50 from {SOURCE}:", fmt(est), decide(est))
