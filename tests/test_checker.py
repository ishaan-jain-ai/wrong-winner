import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from wrong_winner.checker import (
    A_FIRST,
    B_FIRST,
    HUMAN,
    InputError,
    Item,
    add_labels,
    check,
    exit_code,
    labels_needed,
    parse_verdict,
    pick,
    read_items,
    read_labels,
    render,
)
from wrong_winner.cli import main
from wrong_winner.stats import Estimate


def judged(b, a, tie=0):
    """Items judged in one order, alternating which answer is shown first.

    Scores are from B's view: b items where B wins, then a where A wins, then ties.
    """
    scores = [1.0] * b + [0.0] * a + [0.5] * tie
    return [
        Item(str(i), a_first=s) if i % 2 == 0 else Item(str(i), b_first=s)
        for i, s in enumerate(scores)
    ]


def with_humans(items, humans):
    """The items with human scores on the first len(humans) of them."""
    return [
        Item(it.id, it.a_first, it.b_first, humans[i] if i < len(humans) else None)
        for i, it in enumerate(items)
    ]


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestParse(unittest.TestCase):
    def test_systems_and_ties(self):
        self.assertEqual(parse_verdict(" A ", A_FIRST), 0.0)
        self.assertEqual(parse_verdict("b", B_FIRST), 1.0)
        self.assertEqual(parse_verdict("TIE", HUMAN), 0.5)
        self.assertEqual(parse_verdict("draw", HUMAN), 0.5)
        self.assertIsNone(parse_verdict("  ", A_FIRST))

    def test_names(self):
        self.assertEqual(parse_verdict("Current", A_FIRST, "current", "candidate"), 0.0)
        self.assertEqual(parse_verdict("candidate", HUMAN, "current", "candidate"), 1.0)

    def test_positions_depend_on_the_column(self):
        self.assertEqual(parse_verdict("first", A_FIRST), 0.0)  # A shown first and won
        self.assertEqual(parse_verdict("second", A_FIRST), 1.0)
        self.assertEqual(parse_verdict("first", B_FIRST), 1.0)
        self.assertEqual(parse_verdict("second", B_FIRST), 0.0)

    def test_humans_have_no_positions_and_junk_is_refused(self):
        with self.assertRaises(InputError):
            parse_verdict("first", HUMAN)
        with self.assertRaisesRegex(InputError, "judge_a_first"):
            parse_verdict("maybe", A_FIRST)


class TestItem(unittest.TestCase):
    def test_a_flip_counts_as_a_tie(self):
        self.assertEqual(Item("1", a_first=0.0, b_first=1.0).judge, 0.5)
        self.assertEqual(Item("1", a_first=1.0, b_first=0.5).judge, 0.75)
        self.assertEqual(Item("1", b_first=1.0).judge, 1.0)
        self.assertIsNone(Item("1").judge)


class TestCheck(unittest.TestCase):
    def test_flips_and_first_answer(self):
        # 6 items where the first answer wins both times, 4 where B wins both times.
        items = [Item(str(i), a_first=0.0, b_first=1.0) for i in range(6)]
        items += [Item(str(i), a_first=1.0, b_first=1.0) for i in range(6, 10)]
        r = check(items)
        self.assertEqual((r.both_orders, r.flips), (10, 6))
        self.assertAlmostEqual(r.flip_rate.value, 0.6)
        self.assertAlmostEqual(r.first_answer_rate.value, 16 / 20)
        self.assertAlmostEqual(r.judge.value, 0.7)

    def test_a_significant_but_modest_judge_lead_is_not_proven(self):
        r = check(judged(52, 28))  # 65%: interval excludes a tie, but only 15 points out
        self.assertGreater(r.judge.low, 0.5)
        self.assertEqual((r.result, r.basis), ("not_proven", "judge"))
        self.assertEqual(exit_code(r, "better"), 1)
        self.assertEqual(exit_code(r, "not-worse"), 0)
        self.assertEqual(exit_code(r, "none"), 0)
        self.assertIn("NOT PROVEN", render(r))

    def test_a_wide_judge_margin_counts(self):
        r = check(judged(68, 12))  # 85%
        self.assertEqual(r.result, "b_better")
        self.assertEqual(exit_code(r), 0)
        self.assertIn("B IS BETTER (judge alone, wide margin)", render(r))

    def test_clear_gap_is_adjustable(self):
        r = check(judged(52, 28), clear_gap=0.1)
        self.assertEqual(r.result, "b_better")
        text = " ".join(render(r).split())
        self.assertIn("B IS BETTER (judge alone)", text)
        self.assertIn("could still be a tie", text)

    def test_a_clearly_worse_candidate_fails_not_worse(self):
        r = check(judged(10, 70))
        self.assertEqual(r.result, "a_better")
        self.assertEqual(exit_code(r, "not-worse"), 1)

    def test_no_difference_and_too_few_items(self):
        self.assertEqual(check(judged(40, 40)).result, "no_difference")
        self.assertEqual(check(judged(5, 0)).result, "too_few_items")

    def test_labels_overrule_the_judge(self):
        items = judged(60, 20)  # the judge alone: 75%, not proven
        self.assertEqual(check(items).result, "not_proven")
        r = check(with_humans(items, [1.0, 0.0] * 15))  # people split 15-15
        self.assertEqual((r.result, r.basis, r.labels), ("no_difference", "labels", 30))
        self.assertAlmostEqual(r.humans.value, 0.5)
        self.assertEqual(r.judge_weight, 0.0)  # the labelled items all have the same judge score
        self.assertIsNone(r.labels_needed)
        self.assertIn("cost, latency or safety", render(r))

    def test_labels_can_back_the_candidate(self):
        r = check(with_humans(judged(60, 20), [1.0] * 27 + [0.0] * 3))
        self.assertEqual((r.result, r.basis), ("b_better", "labels"))
        self.assertEqual(exit_code(r), 0)
        self.assertIn("Backed by your 30 labels", render(r))

    def test_few_labels_do_not_decide(self):
        r = check(with_humans(judged(60, 20), [0.0] * 5))
        self.assertEqual((r.basis, r.labels), ("judge", 5))
        self.assertTrue(any("Only 5 label" in n for n in r.notes))

    def test_a_judge_that_tracks_people_gets_weight(self):
        pattern = [1.0, 0.0, 1.0, 0.5]
        items = [Item(str(i), a_first=pattern[i % 4]) for i in range(80)]
        r = check(with_humans(items, [pattern[i % 4] for i in range(20)]))
        self.assertEqual(r.agreed, 20)
        self.assertAlmostEqual(r.kappa, 1.0)
        # Cov / ((1 + k/N) Var): sample covariance over 20 items, sample variance over all 80.
        self.assertAlmostEqual(r.judge_weight, (20 / 19) / (80 / 79) / (1 + 20 / 60))

    def test_notes(self):
        r = check(with_humans(judged(60, 20), [0.0] * 25 + [1.0] * 5))
        self.assertTrue(any("opposite directions" in n for n in r.notes))
        one_way = check([Item(str(i), a_first=1.0 if i % 3 else 0.0) for i in range(30)])
        self.assertTrue(any("one order only" in n for n in one_way.notes))
        self.assertTrue(any("showed A first" in n for n in one_way.notes))
        missing = check(judged(20, 10) + [Item("x", human=1.0)])
        self.assertTrue(any("no judge verdict" in n and "1 of them labelled" in n for n in missing.notes))

    def test_unanimous_scores_get_a_wilson_interval(self):
        r = check(judged(20, 0))
        self.assertEqual(r.judge.value, 1.0)
        self.assertAlmostEqual(r.judge.low, 0.8389, places=3)  # not a zero-width interval
        self.assertEqual(r.result, "b_better")
        labelled = check(with_humans(judged(20, 0), [1.0] * 12))
        self.assertLess(labelled.combined.low, 1.0)
        self.assertEqual(labelled.combined, labelled.humans)

    def test_labels_needed(self):
        self.assertEqual(labels_needed(Estimate(0.55, 0.45, 0.65), 25), 100)
        self.assertIsNone(labels_needed(Estimate(0.5, 0.4, 0.6), 25))

    def test_add_labels_refuses_unknown_and_conflicting_ids(self):
        items = [Item("1", a_first=1.0, human=1.0), Item("2", a_first=0.0)]
        self.assertEqual(add_labels(items, {"2": 0.5})[1].human, 0.5)
        with self.assertRaisesRegex(InputError, "not in the verdicts file"):
            add_labels(items, {"3": 1.0})
        with self.assertRaisesRegex(InputError, "different human label"):
            add_labels(items, {"1": 0.0})

    def test_report_to_dict(self):
        d = check(judged(52, 28)).to_dict()
        self.assertEqual(d["result"], "not_proven")
        self.assertEqual(set(d["judge"]), {"value", "low", "high"})
        json.dumps(d, allow_nan=False)


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_headers_are_case_insensitive_and_ids_optional(self):
        path = self.tmp / "v.csv"
        path.write_text("﻿Judge_A_First,JUDGE_B_FIRST\na,b\nfirst,\n", encoding="utf-8")
        items = read_items(path)
        self.assertEqual([it.id for it in items], ["1", "2"])
        self.assertEqual(items[0].judge, 0.5)
        self.assertEqual(items[1].judge, 0.0)

    def test_bad_files_are_explained(self):
        cases = {
            "needs a 'judge_a_first'": ("id,verdict\n1,a\n"),
            "line 3: id '1' appears more than once": "id,judge_a_first\n1,a\n1,b\n",
            "line 2: empty id": "id,judge_a_first\n,a\n",
            "line 2: can't read 'yes'": "id,judge_a_first\n1,yes\n",
        }
        for message, text in cases.items():
            path = self.tmp / "bad.csv"
            path.write_text(text)
            with self.subTest(message=message), self.assertRaisesRegex(InputError, message):
                read_items(path)
        with self.assertRaisesRegex(InputError, "no such file"):
            read_items(self.tmp / "missing.csv")

    def test_read_labels(self):
        path = write_csv(self.tmp / "l.csv", ["id", "prompt", "human"], [["1", "x", "b"], ["2", "y", ""]])
        self.assertEqual(read_labels(path), {"1": 1.0})
        bad = write_csv(self.tmp / "bad.csv", ["id", "label"], [["1", "b"]])
        with self.assertRaisesRegex(InputError, "missing: human"):
            read_labels(bad)

    def test_pick_hides_the_judge_and_keeps_the_rest(self):
        rows = [[str(i), f"q{i}", "a" if i % 2 else "b", "", "b"] for i in range(30)]
        rows[3][2] = ""  # no judge verdict: never picked
        path = write_csv(self.tmp / "v.csv", ["id", "prompt", "judge_a_first", "judge_b_first", "human"], rows)
        header, picked = pick(path, 10, seed=1)
        self.assertEqual(header, ["id", "prompt", "human"])
        self.assertEqual(len(picked), 10)
        self.assertNotIn("3", [r["id"] for r in picked])
        self.assertTrue(all(r["human"] == "" for r in picked))
        self.assertEqual([r["id"] for r in picked], sorted((r["id"] for r in picked), key=int))
        self.assertEqual(picked, pick(path, 10, seed=1)[1])
        self.assertNotEqual(picked, pick(path, 10, seed=2)[1])
        self.assertEqual(len(pick(path, 100)[1]), 29)


class TestCli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)
        rows = [[str(i), "current" if i < 28 else "candidate", ""] for i in range(80)]
        self.verdicts = write_csv(self.tmp / "v.csv", ["id", "judge_a_first", "judge_b_first"], rows)

    def tearDown(self):
        self.dir.cleanup()

    def test_check_text_and_exit_codes(self):
        code, out, _ = run(["check", str(self.verdicts), "--a", "current", "--b", "candidate"])
        self.assertEqual(code, 1)
        self.assertIn("current vs candidate, 80 items", out)
        self.assertIn("NOT PROVEN", out)
        self.assertIn(f"    wrong-winner pick {self.verdicts} 25 -o labels.csv --a current --b candidate\n", out)
        code, _, _ = run(["check", str(self.verdicts), "--a", "current", "--b", "candidate", "--gate", "none"])
        self.assertEqual(code, 0)
        code, _, _ = run(["check", str(self.verdicts), "--a", "current", "--b", "candidate", "--clear-gap", "10"])
        self.assertEqual(code, 0)

    def test_check_json(self):
        code, out, _ = run(["check", str(self.verdicts), "--a", "current", "--b", "candidate", "--json"])
        self.assertEqual(json.loads(out)["result"], "not_proven")

    def test_bad_input_exits_2(self):
        code, _, err = run(["check", str(self.verdicts)])  # "current" isn't a verdict without --a
        self.assertEqual(code, 2)
        self.assertIn("can't read 'current'", err)

    def test_pick_then_check_with_labels(self):
        labels = self.tmp / "labels.csv"
        code, _, err = run(["pick", str(self.verdicts), "30", "-o", str(labels), "--a", "current", "--b", "candidate"])
        self.assertEqual(code, 0)
        self.assertIn("Wrote 30 items", err)
        with labels.open(newline="") as f:
            picked = list(csv.DictReader(f))
        self.assertEqual(list(picked[0]), ["id", "human"])
        for row in picked:
            row["human"] = "tie"
        write_csv(labels, ["id", "human"], [[r["id"], r["human"]] for r in picked])
        before = labels.read_text()
        code, _, err = run(["pick", str(self.verdicts), "-o", str(labels)])
        self.assertEqual(code, 2)
        self.assertIn("not overwriting", err)
        self.assertEqual(labels.read_text(), before)
        code, out, _ = run(
            ["check", str(self.verdicts), "--labels", str(labels), "--a", "current", "--b", "candidate"]
        )
        self.assertEqual(code, 1)
        self.assertIn("Your labels (30 items)", out)
        self.assertIn("NO MEASURABLE DIFFERENCE", out)


if __name__ == "__main__":
    unittest.main()
