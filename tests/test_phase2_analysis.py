"""Tests for the Phase 2 analysis: t intervals, decision tallies and the run_phase2 helpers."""

from collections import Counter

import pytest

from wrong_winner.judge import Item, orders_for
from wrong_winner.run_phase2 import (
    HUMANS,
    JUDGES,
    JudgeInfo,
    PairInfo,
    best_judge_weight,
    build_pairs,
    family,
    family_rows,
    leave_one_out_rows,
    outcome,
    position_row,
    primary_order,
    read_pairs,
    transport_sources,
    verdict_rows,
)
from wrong_winner.simulate import PairData, simulate_decisions, simulate_pair, tally
from wrong_winner.stats import Estimate, mean_ci, ppi_lambda, ppi_mean, t_value


# ---------------------------------------------------------------------------
# Student's t
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df, expected",
    [(1, 12.7062), (2, 4.3027), (9, 2.2622), (24, 2.0639), (49, 2.0096), (199, 1.9720)],
)
def test_t_value_matches_tables(df, expected):
    assert t_value(0.95, df) == pytest.approx(expected, abs=1e-4)


def test_t_value_tends_to_z_and_checks_input():
    assert t_value(0.95, 100_000) == pytest.approx(1.95996, abs=1e-4)
    assert t_value(0.99, 9) == pytest.approx(3.2498, abs=1e-4)
    with pytest.raises(ValueError):
        t_value(0.95, 0)
    with pytest.raises(ValueError):
        t_value(1.5, 5)


def test_t_intervals_are_wider_for_small_samples():
    scores = [1, 0, 1, 1, 0, 1, 0.5, 1, 1, 0]
    z, t = mean_ci(scores), mean_ci(scores, t=True)
    assert t.value == z.value
    assert t.high - t.low == pytest.approx((z.high - z.low) * 2.2622 / 1.95996, rel=1e-3)
    p_z = ppi_mean([1, 1, 0, 1], [1, 0, 1], [1, 1, 0])
    p_t = ppi_mean([1, 1, 0, 1], [1, 0, 1], [1, 1, 0], t=True)
    assert p_t.value == p_z.value
    assert (p_t.high - p_t.low) > (p_z.high - p_z.low)  # t with k - 1 = 2 degrees of freedom


# ---------------------------------------------------------------------------
# Decisions and tallies
# ---------------------------------------------------------------------------


def test_tally_against_each_truth():
    decisions = Counter({"A": 6, "B": 1, None: 3})
    assert tally(decisions, "A") == {
        "correct_rate": 0.6, "wrong_rate": 0.1, "overclaim_rate": 0.0, "no_decision_rate": 0.3,
    }
    assert tally(decisions, "B")["wrong_rate"] == 0.6
    assert tally(decisions, None) == {
        "correct_rate": 0.0, "wrong_rate": 0.0, "overclaim_rate": 0.7, "no_decision_rate": 0.3,
    }
    with pytest.raises(ValueError):
        tally(Counter(), None)


def _pair(judge_scores, human_scores):
    items = range(len(judge_scores))
    return PairData(
        "a", "b",
        human={i: [h] for i, h in zip(items, human_scores)},
        judge={i: [j] for i, j in zip(items, judge_scores)},
    )


def test_simulate_decisions_matches_simulate_pair():
    pair = _pair([1, 1, 1, 0.5, 1, 0, 1, 1] * 5, [1, 0, 0.5, 0, 1, 0, 1, 0] * 5)
    decisions = simulate_decisions(pair, n=30, reps=200, seed=3, method="judge")
    assert sum(decisions.values()) == 200
    row = simulate_pair(pair, n=30, reps=200, seed=3, method="judge")
    rates = tally(decisions, row["truth"] if row["truth"] != "too_close" else None)
    assert row["correct_rate"] == rates["correct_rate"]
    assert row["overclaim_rate"] == rates["overclaim_rate"]


def test_simulate_decisions_with_t_is_more_cautious():
    pair = _pair([1, 1, 0, 1, 0.5, 1, 0, 1, 1, 0.5] * 4, [0.5] * 40)
    z = simulate_decisions(pair, n=10, reps=500, seed=1)
    t = simulate_decisions(pair, n=10, reps=500, seed=1, t=True)
    assert t[None] >= z[None]


# ---------------------------------------------------------------------------
# run_phase2 helpers
# ---------------------------------------------------------------------------


def test_family_prefixes():
    assert family("qwen3-235b-a22b-no-thinking") == "qwen"
    assert family("gemini-2.5-flash") == "google"
    assert family("mistral-medium-2505") == "mistral"
    assert family("o3-2025-04-16") == "openai"
    assert family("chatgpt-4o-latest-20250326") == "openai"
    assert family("amazon.nova-pro-v1:0") == "amazon"
    assert family("something-new") == "other"


def test_same_family_pair_counts_match_the_plan():
    pairs = read_pairs()
    assert len(pairs) == 25
    counts = Counter()
    for p in pairs.values():
        for fam in {family(p.model_a), family(p.model_b)}:
            counts[fam] += 1
    assert (counts["qwen"], counts["google"], counts["mistral"]) == (8, 9, 4)  # as in PHASE2.md


def test_judge_files_are_distinct():
    assert len({j.file for j in JUDGES}) == len(JUDGES)
    assert len({j.key for j in JUDGES}) == len(JUDGES)


def test_outcome():
    assert outcome(None, None) == "no_call"
    assert outcome("A", None) == "overclaim"
    assert outcome("A", "A") == "correct"
    assert outcome("B", "A") == "wrong"


def test_primary_order_is_the_coin_flip_for_two_way_items():
    assert primary_order("x", ("ba",)) == "ba"
    assert primary_order("x", ("ab", "ba")) == orders_for("x", "one", 0)[0]


PAIR = PairInfo("m1", "m2", "close", Estimate(0.5, 0.48, 0.52), Estimate(0.5, 0.4, 0.6))


def _item(id_, model_a, model_b, human):
    return Item(id_, "close", model_a, model_b, human, 100)


def test_build_pairs_orients_scores_to_the_pair():
    plan = [
        (_item("i1", "m1", "m2", 1.0), ("ab",)),  # same orientation as the pair
        (_item("i2", "m2", "m1", 1.0), ("ba",)),  # flipped: m2 won the vote
        (_item("i3", "m1", "m2", 0.5), ("ab",)),  # no verdict
    ]
    verdicts = {("i1", "ab"): "A", ("i2", "ba"): "A"}  # i2: first shown is m1, so m1 wins
    data = build_pairs(plan, {PAIR.key: PAIR}, verdicts)[PAIR.key]
    assert data.human == {"i1": [1.0], "i2": [0.0], "i3": [0.5]}
    assert data.judge == {"i1": [1.0], "i2": [1.0]}
    humans = build_pairs(plan, {PAIR.key: PAIR})[PAIR.key]
    assert humans.judge == humans.human


def test_verdict_rows_flag_an_overclaim():
    plan = [(_item(f"i{n}", "m1", "m2", 0.5), ("ab",)) for n in range(40)]
    verdicts = {(f"i{n}", "ab"): "A" if n % 4 else "tie" for n in range(40)}
    data = build_pairs(plan, {PAIR.key: PAIR}, verdicts)
    (row,) = verdict_rows("j", {PAIR.key: PAIR}, data)
    assert row["truth_bt"] == "too_close"
    assert row["judge_call"] == "A"
    assert row["outcome_bt"] == "overclaim"
    assert row["items"] == 40


def test_position_row_counts_changes():
    plan = [
        (_item("same", "m1", "m2", 0.5), ("ab", "ba")),   # A then B: same model both times
        (_item("first", "m1", "m2", 0.5), ("ab", "ba")),  # A both times: first answer wins
        (_item("tie", "m1", "m2", 0.5), ("ab", "ba")),    # tie then B
        (_item("gap", "m1", "m2", 0.5), ("ab", "ba")),    # one order missing
        (_item("one", "m1", "m2", 0.5), ("ab",)),
    ]
    verdicts = {
        ("same", "ab"): "A", ("same", "ba"): "B",
        ("first", "ab"): "A", ("first", "ba"): "A",
        ("tie", "ab"): "tie", ("tie", "ba"): "B",
        ("gap", "ab"): "A",
        ("one", "ab"): "B",
    }
    row = position_row("j", plan, verdicts)
    assert row["items_both_ways"] == 3
    assert row["items_missing_an_order"] == 1
    assert row["changed"] == 2
    assert (row["first_both_times"], row["second_both_times"], row["one_tie"]) == (1, 0, 1)


def test_family_rows_measure_preference_against_other_judges():
    pair = PairInfo("qwen-x", "other-y", "close", Estimate(0.5, 0.48, 0.52), Estimate(0.5, 0.4, 0.6))
    pairs = {pair.key: pair}
    items = [f"i{n}" for n in range(10)]

    def data(judge_scores):
        return {pair.key: PairData(
            pair.model_a, pair.model_b,
            human={i: [0.5] for i in items},
            judge={i: [s] for i, s in zip(items, judge_scores)},
        )}

    infos = [
        JudgeInfo("q", "Q", "q.jsonl", "8B", 8, "qwen", False),
        JudgeInfo("o", "O", "o.jsonl", "8B", 8, "nvidia", False),
    ]
    judged = {"q": data([1.0] * 8 + [0.5] * 2), "o": data([0.5] * 10)}
    rows = family_rows(infos, pairs, judged)
    by_pair = {r["pair"]: r for r in rows}
    assert set(by_pair) == {pair.name, "all"}
    assert by_pair[pair.name]["family_model"] == "qwen-x"
    assert by_pair[pair.name]["self_minus_others_points"] == pytest.approx(40.0)
    assert by_pair[pair.name]["judge_minus_humans_points"] == pytest.approx(40.0)
    assert by_pair["all"]["pairs"] == 1


def test_transport_sources_stay_in_group():
    pairs = read_pairs()
    sources = transport_sources(pairs)
    assert set(sources) == set(pairs)
    for target, source in sources.items():
        assert source != target
        assert pairs[source].group == pairs[target].group


def test_humans_label_is_reserved():
    assert HUMANS not in {j.key for j in JUDGES}


# ---------------------------------------------------------------------------
# PPI++ (power-tuned PPI)
# ---------------------------------------------------------------------------


def test_ppi_with_weight_zero_is_the_human_mean():
    human = [1, 0, 0.5, 1, 1, 0, 1, 0.5]
    judge = [0, 1, 1, 0, 1, 1, 0.5, 0]
    for t in (False, True):
        assert ppi_mean([1, 1, 0], judge, human, t=t, lam=0.0) == mean_ci(human, t=t)


def test_ppi_weight_one_is_plain_ppi():
    args = ([1, 0.5, 0, 1], [1, 0, 1], [1, 1, 0])
    assert ppi_mean(*args, lam=1.0) == ppi_mean(*args)


def test_ppi_lambda_trusts_a_judge_that_tracks_humans():
    human = [1, 0, 1, 0, 1, 0.5, 0, 1]
    unlabelled = [1, 0, 0.5, 1] * 20
    lam = ppi_lambda(unlabelled, human, human)  # judge == humans on the labelled items
    assert 0.8 < lam <= 1.0  # about 1 / (1 + 8 / 80), from the k / N term


def test_ppi_lambda_ignores_a_useless_judge():
    human = [1, 0, 1, 0, 1, 0, 1, 0]
    assert ppi_lambda([1, 0] * 20, [1, 1, 0, 0, 1, 1, 0, 0], human) == 0.0  # uncorrelated
    assert ppi_lambda([0] * 20, [1, 0, 0, 1, 0, 1, 1, 0], human) == 0.0  # anti-correlated: clipped
    assert ppi_lambda([0.5] * 20, [0.5] * 8, human) == 0.0  # constant judge
    assert ppi_lambda([], [1, 0] * 4, human) == 0.0  # no unlabelled items
    with pytest.raises(ValueError):
        ppi_lambda([1], [1, 0], [1])


def test_ppi_tuned_simulation_never_much_worse_than_humans_alone():
    # A judge with no relation to the humans: PPI++ should fall back to the labels.
    human = [1, 0, 0.5, 1, 0, 1, 0, 0.5, 1, 1] * 8
    judge = [0.5, 1, 1, 0, 1, 0, 0.5, 1, 0, 1] * 8
    pair = _pair(judge, human)
    plain = simulate_decisions(pair, n=80, k=25, method="ppi", reps=400, seed=2, t=True)
    tuned = simulate_decisions(pair, n=80, k=25, method="ppi_tuned", reps=400, seed=2, t=True)
    alone = simulate_decisions(pair, n=25, k=25, method="human_only", reps=400, seed=2, t=True)
    assert sum(tuned.values()) == 400
    calls = lambda d: d["A"] + d["B"]  # noqa: E731
    assert calls(tuned) >= calls(plain)
    assert abs(calls(tuned) - calls(alone)) <= 40


def test_ppi_tuned_needs_labels():
    pair = _pair([1, 0] * 10, [1, 0] * 10)
    with pytest.raises(ValueError):
        simulate_decisions(pair, n=10, k=0, method="ppi_tuned")
    with pytest.raises(ValueError):
        simulate_decisions(pair, n=10, k=20, method="ppi_tuned")


def test_leave_one_out_drops_each_models_pairs():
    def v(judge, pair, call, outcome_bt):
        return {"judge": judge, "group": "close", "pair": pair, "judge_call": call, "outcome_bt": outcome_bt}

    def g(judge, pair, overclaim):
        method, k = ("judge", "0") if judge != HUMANS else ("human_only", "80")
        return {"judge": judge, "group": "close", "pair": pair, "rule": "significant", "n": "80",
                "k": k, "method": method, "overclaim_bt": str(overclaim), "wrong_bt": "0.0"}

    pairs = ["x vs y", "x vs z", "y vs z"]
    verdicts, grid = [], []
    for j in ("j1", "j2", "j3"):
        verdicts += [v(j, "x vs y", "A", "overclaim"), v(j, "x vs z", "none", "no_call"),
                     v(j, "y vs z", "B" if j == "j1" else "A", "overclaim")]
        grid += [g(j, p, 0.6 if p != "x vs z" else 0.0) for p in pairs]
    verdicts += [v(HUMANS, p, "none", "no_call") for p in pairs]
    grid += [g(HUMANS, p, 0.1) for p in pairs]
    verdicts.append({**v("j1", "a vs b", "A", "overclaim"), "group": "clear"})  # ignored: not close

    rows = leave_one_out_rows(verdicts, grid)
    by = {(r["left_out"], r["judge"]): r for r in rows}
    assert {r["left_out"] for r in rows} == {"(none)", "x", "y", "z"}
    full = by[("(none)", "j1")]
    assert (full["close_pairs"], full["unsupported_all_items"]) == (3, 2)
    assert full["sim_unsupported_n80"] == pytest.approx(0.4)
    assert full["pairs_shared_by_3_judges"] == 1  # x vs y: all three call A; y vs z splits 2 to 1
    no_x = by[("x", "j1")]
    assert (no_x["close_pairs"], no_x["unsupported_all_items"], no_x["pairs_shared_by_3_judges"]) == (1, 1, 0)
    assert by[("y", "j2")]["unsupported_share"] == 0.0
    assert by[("(none)", HUMANS)]["unsupported_all_items"] == 0
    assert by[("(none)", HUMANS)]["sim_unsupported_n80"] == pytest.approx(0.1)


def test_best_judge_weight():
    assert best_judge_weight([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert best_judge_weight([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    assert best_judge_weight([1, 0, 1, 0], [0.5] * 4) == 0.0
    assert best_judge_weight([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
