import math
import unittest

from wrong_winner.simulate import PairData, agreement, score_label
from wrong_winner.stats import cohen_kappa, pearson_r


class TestKappa(unittest.TestCase):
    def test_known_value(self):
        # observed 0.75, expected 0.5 -> kappa 0.5
        self.assertAlmostEqual(cohen_kappa("AABB", "ABBB"), 0.5)

    def test_perfect_agreement(self):
        self.assertAlmostEqual(cohen_kappa(["A", "B", "tie"], ["A", "B", "tie"]), 1.0)

    def test_chance_level_is_about_zero(self):
        self.assertAlmostEqual(cohen_kappa("ABAB", "AABB"), 0.0)

    def test_single_label_edge_cases(self):
        self.assertEqual(cohen_kappa("AAA", "AAA"), 1.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            cohen_kappa("AB", "A")


class TestPearson(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(pearson_r([0, 0.5, 1], [0.1, 0.6, 1.1]), 1.0)

    def test_constant_is_nan(self):
        self.assertTrue(math.isnan(pearson_r([1, 1, 1], [0, 0.5, 1])))


class TestPairAgreement(unittest.TestCase):
    def test_score_label(self):
        self.assertEqual(score_label(0.8), "A")
        self.assertEqual(score_label(0.2), "B")
        self.assertEqual(score_label(0.5), "tie")

    def test_high_agreement_can_still_flip_a_close_winner(self):
        # Humans: A barely ahead (6 A wins, 4 B wins). The judge agrees on 8/10
        # items but its two disagreements both go to B, flipping the winner.
        human = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        judge = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        pair = PairData("A", "B")
        for i, (h, j) in enumerate(zip(human, judge)):
            pair.human[i] = [float(h)]
            pair.judge[i] = [float(j)]
        a = agreement(pair)
        self.assertAlmostEqual(a["raw_agreement"], 0.8)
        self.assertGreater(a["pearson_r"], 0.6)
        self.assertGreater(a["human_win_rate_a"], 0.5)
        self.assertLess(a["judge_win_rate_a"], 0.5)

    def test_needs_shared_items(self):
        pair = PairData("A", "B", human={1: [1.0]}, judge={2: [0.0]})
        with self.assertRaises(ValueError):
            agreement(pair)


if __name__ == "__main__":
    unittest.main()
