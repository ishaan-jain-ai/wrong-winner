import unittest

from wrong_winner.stats import (
    CorrectionRefused,
    Estimate,
    JudgeError,
    decide,
    estimate_judge_error,
    mean_ci,
    ppi_mean,
    rogan_gladen,
    wilson_interval,
    z_value,
)


class TestIntervals(unittest.TestCase):
    def test_z_value(self):
        self.assertAlmostEqual(z_value(0.95), 1.959964, places=5)

    def test_wilson_known_value(self):
        est = wilson_interval(8, 10)
        self.assertAlmostEqual(est.value, 0.8)
        self.assertAlmostEqual(est.low, 0.4902, places=4)
        self.assertAlmostEqual(est.high, 0.9433, places=4)

    def test_wilson_zero_successes_stays_in_range(self):
        est = wilson_interval(0, 10)
        self.assertEqual(est.low, 0.0)
        self.assertGreater(est.high, 0.0)

    def test_wilson_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_mean_ci_symmetric(self):
        est = mean_ci([1.0, 0.0] * 50)
        self.assertAlmostEqual(est.value, 0.5)
        self.assertAlmostEqual(est.value - est.low, est.high - est.value)
        self.assertTrue(est.low < 0.5 < est.high)

    def test_mean_ci_handles_ties(self):
        est = mean_ci([1.0, 0.5, 0.5, 0.0, 1.0])
        self.assertAlmostEqual(est.value, 0.6)


class TestJudgeCorrection(unittest.TestCase):
    def test_rogan_gladen_recovers_true_rate(self):
        # true 0.3, sens 0.9, spec 0.8 -> observed 0.3*0.9 + 0.7*0.2 = 0.41
        err = JudgeError(sensitivity=0.9, specificity=0.8)
        self.assertAlmostEqual(rogan_gladen(0.41, err), 0.3)

    def test_rogan_gladen_perfect_judge_is_identity(self):
        err = JudgeError(1.0, 1.0)
        self.assertAlmostEqual(rogan_gladen(0.37, err), 0.37)

    def test_rogan_gladen_refuses_near_chance(self):
        with self.assertRaises(CorrectionRefused):
            rogan_gladen(0.5, JudgeError(0.55, 0.5))

    def test_rogan_gladen_clips(self):
        self.assertEqual(rogan_gladen(0.0, JudgeError(0.9, 0.8)), 0.0)

    def test_estimate_judge_error(self):
        human = [True, True, True, True, False, False, False, False, False]
        judge = [True, True, True, False, False, False, False, False, True]
        err = estimate_judge_error(judge, human)
        self.assertAlmostEqual(err.sensitivity, 0.75)
        self.assertAlmostEqual(err.specificity, 0.8)

    def test_estimate_judge_error_needs_both_classes(self):
        with self.assertRaises(ValueError):
            estimate_judge_error([True, False], [True, True])

    def test_ppi_removes_constant_bias(self):
        human_unlab = [0.0, 0.5, 1.0, 1.0, 0.0, 0.5]
        human_lab = [1.0, 0.0, 0.5, 0.5]
        bias = 0.2
        est = ppi_mean(
            judge_unlabelled=[h + bias for h in human_unlab],
            judge_labelled=[h + bias for h in human_lab],
            human_labelled=human_lab,
        )
        self.assertAlmostEqual(est.value, sum(human_unlab) / len(human_unlab))

    def test_ppi_with_perfect_judge_matches_judge_mean(self):
        judge = [1.0, 0.0, 1.0, 1.0]
        est = ppi_mean(judge, [1.0, 0.0], [1.0, 0.0])
        self.assertAlmostEqual(est.value, 0.75)

    def test_ppi_all_labelled(self):
        est = ppi_mean([], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(est.value, 1 / 3)

    def test_ppi_needs_labels(self):
        with self.assertRaises(ValueError):
            ppi_mean([1.0], [], [])


class TestDecide(unittest.TestCase):
    def test_naive(self):
        self.assertEqual(decide(Estimate(0.51, 0.3, 0.7), "naive"), "A")
        self.assertEqual(decide(Estimate(0.49, 0.3, 0.7), "naive"), "B")
        self.assertIsNone(decide(Estimate(0.5, 0.3, 0.7), "naive"))

    def test_significant(self):
        self.assertIsNone(decide(Estimate(0.6, 0.45, 0.75), "significant"))
        self.assertEqual(decide(Estimate(0.6, 0.52, 0.68), "significant"), "A")
        self.assertEqual(decide(Estimate(0.4, 0.32, 0.48), "significant"), "B")

    def test_unknown_rule(self):
        with self.assertRaises(ValueError):
            decide(Estimate(0.5, 0.4, 0.6), "vibes")


if __name__ == "__main__":
    unittest.main()
