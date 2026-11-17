import random
import unittest

from wrong_winner.data import build_pairs, check_schema, winner_to_score
from wrong_winner.simulate import PairData, ground_truth, run_grid, simulate_pair


def synthetic_pair(human_p, judge_p, items=200, annotators=3, seed=0):
    """Binary win/lose scores: humans favour A with prob human_p, the judge with judge_p."""
    rng = random.Random(seed)
    pair = PairData("A", "B")
    for i in range(items):
        pair.human[i] = [float(rng.random() < human_p) for _ in range(annotators)]
        pair.judge[i] = [float(rng.random() < judge_p)]
    return pair


def too_close_pair(items=200, seed=0):
    rng = random.Random(seed)
    pair = PairData("A", "B")
    for i in range(items):
        pair.human[i] = [1.0 if i % 2 else 0.0]
        pair.judge[i] = [float(rng.random() < 0.5)]
    return pair


REPS = 400


class TestSimulate(unittest.TestCase):
    def test_honest_judge_rarely_wrong(self):
        pair = synthetic_pair(0.75, 0.75)
        self.assertEqual(ground_truth(pair).winner, "A")
        row = simulate_pair(pair, n=200, method="judge", reps=REPS)
        self.assertLess(row["wrong_rate"], 0.02)
        self.assertGreater(row["correct_rate"], 0.9)

    def test_small_n_has_more_no_decisions(self):
        pair = synthetic_pair(0.62, 0.62)
        small = simulate_pair(pair, n=25, method="judge", reps=REPS)
        large = simulate_pair(pair, n=200, method="judge", reps=REPS)
        self.assertGreater(small["no_decision_rate"], large["no_decision_rate"])

    def test_biased_judge_picks_wrong_winner_and_ppi_fixes_it(self):
        pair = synthetic_pair(0.65, 0.35)
        self.assertEqual(ground_truth(pair).winner, "A")
        judge = simulate_pair(pair, n=200, method="judge", reps=REPS)
        ppi = simulate_pair(pair, n=200, method="ppi", k=100, reps=REPS)
        self.assertGreater(judge["wrong_rate"], 0.9)
        self.assertLess(ppi["wrong_rate"], 0.05)

    def test_too_close_pair_counts_confident_calls_as_overclaims(self):
        pair = too_close_pair()
        self.assertIsNone(ground_truth(pair).winner)
        row = simulate_pair(pair, n=50, method="judge", rule="naive", reps=REPS)
        self.assertEqual(row["truth"], "too_close")
        self.assertEqual(row["correct_rate"], 0.0)
        self.assertEqual(row["wrong_rate"], 0.0)
        self.assertGreater(row["overclaim_rate"], 0.8)

    def test_clear_pair_never_overclaims(self):
        pair = synthetic_pair(0.65, 0.35)
        row = simulate_pair(pair, n=200, method="judge", reps=REPS)
        self.assertEqual(row["overclaim_rate"], 0.0)

    def test_transport_uses_source_correction(self):
        # Target judge is biased; source pair has an unbiased judge, so the
        # reused correction is ~0 and never recovers the right winner, while
        # the same labels taken on the target pair itself do.
        target = synthetic_pair(0.65, 0.35, seed=1)
        source = synthetic_pair(0.6, 0.6, seed=2)
        transport = simulate_pair(
            target, n=200, k=100, method="transport", transport_source=source, reps=REPS
        )
        local = simulate_pair(target, n=200, k=100, method="ppi", reps=REPS)
        self.assertLess(transport["correct_rate"], 0.1)
        self.assertGreater(local["correct_rate"], transport["correct_rate"] + 0.3)

    def test_deterministic_with_seed(self):
        pair = synthetic_pair(0.6, 0.55)
        a = simulate_pair(pair, n=50, method="judge", reps=200, seed=7)
        b = simulate_pair(pair, n=50, method="judge", reps=200, seed=7)
        self.assertEqual(a, b)

    def test_rates_sum_to_one(self):
        pair = synthetic_pair(0.6, 0.55)
        row = simulate_pair(pair, n=50, method="ppi", k=20, reps=200)
        total = (
            row["correct_rate"]
            + row["wrong_rate"]
            + row["overclaim_rate"]
            + row["no_decision_rate"]
        )
        self.assertAlmostEqual(total, 1.0)

    def test_input_validation(self):
        pair = synthetic_pair(0.6, 0.6)
        with self.assertRaises(ValueError):
            simulate_pair(pair, n=50, method="ppi", k=1)
        with self.assertRaises(ValueError):
            simulate_pair(pair, n=10, method="ppi", k=20)
        with self.assertRaises(ValueError):
            simulate_pair(pair, n=50, method="transport", k=10)
        with self.assertRaises(ValueError):
            simulate_pair(pair, n=50, method="magic")

    def test_run_grid(self):
        pairs = [synthetic_pair(0.7, 0.7, seed=s) for s in range(2)]
        rows = run_grid(pairs, [{"n": 25, "method": "judge"}], reps=50)
        self.assertEqual(len(rows), 2)


class TestData(unittest.TestCase):
    def test_winner_to_score(self):
        self.assertEqual(winner_to_score("model_a"), 1.0)
        self.assertEqual(winner_to_score("model_b"), 0.0)
        self.assertEqual(winner_to_score("tie"), 0.5)
        self.assertEqual(winner_to_score("tie (bothbad)"), 0.5)
        with self.assertRaises(ValueError):
            winner_to_score("draw-ish")

    def test_build_pairs_canonicalises_order(self):
        human = [
            {"question_id": 81, "turn": 1, "model_a": "zeta", "model_b": "alpha", "winner": "model_a"},
            {"question_id": 81, "turn": 1, "model_a": "alpha", "model_b": "zeta", "winner": "model_a"},
            {"question_id": 82, "turn": 2, "model_a": "alpha", "model_b": "zeta", "winner": "tie"},
        ]
        judge = [
            {"question_id": 81, "turn": 1, "model_a": "zeta", "model_b": "alpha", "winner": "model_b"},
        ]
        (pair,) = build_pairs(human, judge)
        self.assertEqual((pair.model_a, pair.model_b), ("alpha", "zeta"))
        self.assertEqual(pair.human[(81, 1)], [0.0, 1.0])
        self.assertEqual(pair.human[(82, 2)], [0.5])
        self.assertEqual(pair.judge[(81, 1)], [1.0])
        self.assertEqual(pair.shared_items(), [(81, 1)])

    def test_check_schema(self):
        with self.assertRaises(ValueError):
            check_schema([{"question_id": 1}])
        check_schema(
            [{"question_id": 1, "model_a": "a", "model_b": "b", "winner": "tie", "turn": 1}]
        )


if __name__ == "__main__":
    unittest.main()
