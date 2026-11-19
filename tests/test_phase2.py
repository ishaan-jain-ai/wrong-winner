import unittest

from wrong_winner.arena import Vote, filter_votes, pair_counts
from wrong_winner.data import winner_to_score
from wrong_winner.select_pairs import Candidate, group_of, select, spread_pick
from wrong_winner.truth import bootstrap_pairs, fit_bradley_terry, win_prob


def vote(a, b, score, language="en", turns=1, i=0):
    return Vote(str(i), a, b, score, language, turns, 10, 100, 100)


def exact_outcomes(strengths, games_per_pair=1000):
    """Outcomes whose win counts match the Bradley–Terry probabilities exactly."""
    models = sorted(strengths)
    out = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            wins = round(games_per_pair * win_prob(strengths, a, b))
            out += [(a, b, 1.0)] * wins + [(a, b, 0.0)] * (games_per_pair - wins)
    return out


def candidate(a, b, gap):
    return Candidate(a, b, 100, 0.5 + gap / 100, 0, 1, gap, 0.5, 0, 1, False, False, True, False)


class TestArena(unittest.TestCase):
    def test_both_bad_counts_as_tie(self):
        self.assertEqual(winner_to_score("both_bad"), 0.5)

    def test_canonical_flips_score(self):
        self.assertEqual(vote("zeta", "alpha", 1.0).canonical(), (("alpha", "zeta"), 0.0))
        self.assertEqual(vote("alpha", "zeta", 0.5).canonical(), (("alpha", "zeta"), 0.5))

    def test_filter_votes(self):
        votes = [
            vote("a", "b", 1.0),
            vote("a", "b", 1.0, language="pl"),
            vote("a", "b", 1.0, turns=2),
            vote("a", "a", 1.0),
        ]
        self.assertEqual(len(filter_votes(votes)), 1)
        self.assertEqual(len(filter_votes(votes, single_turn=False)), 2)

    def test_pair_counts_merge_orders(self):
        votes = [vote("a", "b", 1.0), vote("b", "a", 0.0), vote("a", "c", 0.5)]
        self.assertEqual(pair_counts(votes), {("a", "b"): 2, ("a", "c"): 1})


class TestBradleyTerry(unittest.TestCase):
    def test_recovers_known_strengths(self):
        true = {"a": 1.0, "b": 2.0, "c": 4.0, "d": 0.5}
        fitted = fit_bradley_terry(exact_outcomes(true), prior=0.0)
        for x in true:
            for y in true:
                if x < y:
                    self.assertAlmostEqual(win_prob(fitted, x, y), win_prob(true, x, y), places=3)

    def test_ties_count_half(self):
        fitted = fit_bradley_terry([("a", "b", 0.5)] * 10, prior=0.0)
        self.assertAlmostEqual(win_prob(fitted, "a", "b"), 0.5)

    def test_prior_keeps_unbeaten_model_finite(self):
        outcomes = [("a", "b", 1.0)] * 20 + [("b", "c", 0.5)] * 20
        fitted = fit_bradley_terry(outcomes)
        self.assertGreater(win_prob(fitted, "a", "b"), 0.9)
        self.assertLess(win_prob(fitted, "a", "b"), 1.0)

    def test_transitive_evidence(self):
        # a and c never met; a beats b and b beats c, so a should beat c.
        outcomes = [("a", "b", 1.0)] * 70 + [("a", "b", 0.0)] * 30
        outcomes += [("b", "c", 1.0)] * 70 + [("b", "c", 0.0)] * 30
        fitted = fit_bradley_terry(outcomes, prior=0.0)
        self.assertGreater(win_prob(fitted, "a", "c"), win_prob(fitted, "a", "b"))

    def test_bootstrap_interval(self):
        outcomes = exact_outcomes({"a": 1.0, "b": 1.5, "c": 1.0}, games_per_pair=200)
        (t,) = bootstrap_pairs(outcomes, [("a", "b")], reps=50, seed=1)
        self.assertLess(t.low, t.p)
        self.assertGreater(t.high, t.p)
        self.assertAlmostEqual(t.p, 0.4, places=2)
        self.assertTrue(t.separable)
        self.assertAlmostEqual(t.gap_points, 10, places=0)


class TestSelection(unittest.TestCase):
    def test_group_boundaries(self):
        self.assertEqual(group_of(0.0), "close")
        self.assertEqual(group_of(2.99), "close")
        self.assertEqual(group_of(3.0), "small")
        self.assertEqual(group_of(7.0), "clear")
        self.assertEqual(group_of(40.0), "clear")

    def test_spread_pick_covers_range(self):
        rows = [candidate("m", f"x{i}", 3 + i * 0.1) for i in range(30)]
        picked = spread_pick(rows, 5)
        self.assertEqual(len(picked), 5)
        self.assertEqual(picked[0].bt_gap_points, 3.0)
        self.assertAlmostEqual(picked[-1].bt_gap_points, 5.9)
        self.assertEqual(spread_pick(rows[:3], 5), rows[:3])

    def test_select_labels_groups(self):
        rows = [candidate("m", f"c{i}", 1.0) for i in range(20)]
        rows += [candidate("m", f"s{i}", 5.0) for i in range(20)]
        rows += [candidate("m", f"l{i}", 12.0) for i in range(20)]
        chosen = select(rows)
        groups = [c.group for c in chosen]
        self.assertEqual(groups.count("close"), 15)
        self.assertEqual(groups.count("small"), 15)
        self.assertEqual(groups.count("clear"), 10)


if __name__ == "__main__":
    unittest.main()
