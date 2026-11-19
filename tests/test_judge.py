import http.client
import json
import os
import tempfile
import unittest
import urllib.error
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest import mock

from wrong_winner.arena import conversation_texts, message_text
from wrong_winner.judge import (
    CallFailed,
    DailyLimitReached,
    GeminiJudge,
    Item,
    NvidiaJudge,
    OllamaJudge,
    Reply,
    _post,
    build_messages,
    load_done,
    make_judge,
    make_plan,
    orders_for,
    parse_gemini,
    parse_openai,
    parse_verdict,
    plan_tasks,
    read_plan,
    reparse,
    run,
    usable,
    verdict_score,
    with_text,
    write_plan,
)


def item(i="x1", group="close", tokens=100, a="model-p", b="model-q"):
    return Item(i, group, a, b, 1.0, tokens, "What is 2+2?", "It is 4.", "It is 5.")


def text(prompt="Q", a="A", b="B", non_text=False):
    return {"prompt": prompt, "answer_a": a, "answer_b": b, "has_non_text": non_text}


def text_msg(role, content):
    return {"role": role, "content": [{"type": "text", "text": content, "image": None, "mimeType": None}]}


class FakeJudge:
    """Always prefers the answer that says 4, wherever it is shown."""

    name = "fake"

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def __call__(self, system, user):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise DailyLimitReached("quota_exceeded")
        self.calls += 1
        first = user.split("<answer_a>")[1].split("</answer_a>")[0]
        return Reply(json.dumps({"verdict": "A" if "4" in first else "B"}), 10, 3)


class TestArenaTexts(unittest.TestCase):
    def test_message_text(self):
        self.assertEqual(message_text(text_msg("user", "hi")), ("hi", False))
        with_image = {"content": [{"type": "text", "text": "look"}, {"type": "image", "image": "x.png"}]}
        self.assertEqual(message_text(with_image), ("look", True))

    def test_conversation_texts(self):
        conv_a = [text_msg("user", "Q"), text_msg("assistant", "A1")]
        conv_b = [text_msg("user", "Q"), text_msg("assistant", "B1")]
        got = conversation_texts(conv_a, conv_b)
        self.assertEqual((got["prompt"], got["answer_a"], got["answer_b"]), ("Q", "A1", "B1"))
        self.assertFalse(got["has_non_text"])
        with self.assertRaises(ValueError):
            conversation_texts(conv_a + conv_a, conv_b)


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.items = []
        for p in range(3):  # three close pairs with 50 items each, one clear pair with 30
            self.items += [item(f"c{p}-{n:02d}", "close", 100, f"m{p}", "z") for n in range(50)]
        self.items += [item(f"l-{n:02d}", "clear", 100, "big", "small") for n in range(30)]
        self.items += [item("long", "close", 9_000, "m0", "z"), item("small-gap", "small", 100)]
        self.texts = {it.id: text() for it in self.items}
        self.texts["c0-00"] = text(a="")  # empty answer: never planned

    def plan(self, **kw):
        return make_plan(self.items, self.texts, per_pair={"close": 40, "clear": 10}, both_orders=20, **kw)

    def test_counts_caps_and_groups(self):
        plan = self.plan()
        per_pair = Counter(it.pair for it, _ in plan)
        self.assertEqual(sorted(per_pair.values()), [10, 40, 40, 40])
        ids = {it.id for it, _ in plan}
        self.assertFalse(ids & {"long", "small-gap", "c0-00"})
        both = [it for it, o in plan if o == ("ab", "ba")]
        self.assertEqual(len(both), 20)
        self.assertTrue(all(it.group == "close" for it in both))
        self.assertTrue(all(len(o) == 1 for it, o in plan if it not in both))

    def test_deterministic_and_round_trips(self):
        plan = self.plan()
        self.assertEqual(plan, self.plan())
        self.assertNotEqual([it.id for it, _ in plan], [it.id for it, _ in self.plan(seed=1)])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plan.csv"
            write_plan(path, plan)
            no_text = [(replace(it, prompt="", answer_a="", answer_b=""), o) for it, o in plan]
            self.assertEqual(read_plan(path), no_text)  # the plan file holds ids, not texts

    def test_plan_tasks_shuffled_and_complete(self):
        plan = self.plan()
        tasks = plan_tasks(plan)
        self.assertEqual(len(tasks), sum(len(o) for _, o in plan))
        self.assertEqual(tasks, plan_tasks(plan))
        self.assertNotEqual([t[0].pair for t in tasks[:20]], [plan[i][0].pair for i in range(20)])

    def test_usable_and_with_text(self):
        self.assertTrue(usable(text()))
        self.assertFalse(usable(text(non_text=True)))
        self.assertFalse(usable(text(b="")))
        bare = Item("x", "close", "p", "q", 0.5, 10)
        self.assertEqual(with_text(bare, {"x": text("P", "a1", "b1")}).answer_b, "b1")


class TestPromptAndVerdicts(unittest.TestCase):
    def test_build_messages_swaps_answers(self):
        _, ab = build_messages(item(), "ab")
        _, ba = build_messages(item(), "ba")
        self.assertLess(ab.index("It is 4."), ab.index("It is 5."))
        self.assertLess(ba.index("It is 5."), ba.index("It is 4."))
        self.assertNotIn("model-p", ab)  # model names stay hidden

    def test_parse_verdict(self):
        self.assertEqual(parse_verdict('{"verdict": "A"}'), "A")
        self.assertEqual(parse_verdict(' {"verdict":"Tie"} '), "tie")
        self.assertEqual(parse_verdict('"B"'), "B")
        self.assertEqual(parse_verdict("B."), "B")
        self.assertEqual(parse_verdict('```json\n{"verdict": "B"}\n```'), "B")
        self.assertEqual(parse_verdict('<think>A looks wrong, "verdict": "A"?</think>\n{"verdict": "tie"}'), "tie")
        self.assertEqual(parse_verdict('```json\n{"verdict": "a"}\n```'), "A")
        self.assertEqual(parse_verdict('<json>\n{"verdict": "b"}\n</json>'), "B")
        self.assertIsNone(parse_verdict('{"verdict": "C"}'))
        self.assertIsNone(parse_verdict("I think A"))
        self.assertIsNone(parse_verdict(""))

    def test_verdict_score_undoes_order(self):
        self.assertEqual(verdict_score("A", "ab"), 1.0)
        self.assertEqual(verdict_score("A", "ba"), 0.0)
        self.assertEqual(verdict_score("B", "ba"), 1.0)
        self.assertEqual(verdict_score("tie", "ba"), 0.5)
        self.assertIsNone(verdict_score(None, "ab"))

    def test_orders_for(self):
        self.assertEqual(orders_for("x", "both"), ("ab", "ba"))
        picks = [orders_for(f"id{i}", "one")[0] for i in range(400)]
        self.assertEqual(picks, [orders_for(f"id{i}", "one")[0] for i in range(400)])  # stable
        self.assertTrue(150 < picks.count("ab") < 250)  # roughly half each way


def no_sleep(seconds):
    return None


class TestBackends(unittest.TestCase):
    def test_ollama_request_and_reply(self):
        seen = {}

        def post(url, body, headers, timeout):
            seen.update(url=url, body=body)
            return 200, {"message": {"content": '{"verdict": "tie"}'}, "prompt_eval_count": 900, "eval_count": 6}

        reply = OllamaJudge("qwen3:8b", num_ctx=1000, post=post)("sys", "user")
        self.assertTrue(seen["url"].endswith("/api/chat"))
        self.assertFalse(seen["body"]["think"])
        self.assertEqual(seen["body"]["options"]["num_ctx"], 1000)
        self.assertEqual((reply.text, reply.in_tokens, reply.out_tokens), ('{"verdict": "tie"}', 900, 6))
        self.assertEqual(reply.error, "")

    def test_ollama_flags_full_context(self):
        post = lambda *a: (200, {"message": {"content": "{}"}, "prompt_eval_count": 990, "eval_count": 1})
        self.assertIn("context", OllamaJudge("m", num_ctx=1000, post=post)("s", "u").error)

    def test_parse_gemini(self):
        data = {
            "status": "completed",
            "steps": [
                {"type": "thought", "content": []},
                {"type": "model_output", "content": [{"type": "text", "text": '{"verdict": "B"}'}]},
            ],
            "usage": {"total_input_tokens": 1200, "total_output_tokens": 5, "total_thought_tokens": 0},
        }
        reply = parse_gemini(data)
        self.assertEqual(parse_verdict(reply.text), "B")
        self.assertEqual((reply.in_tokens, reply.out_tokens), (1200, 5))
        self.assertIn("incomplete", parse_gemini({"status": "incomplete"}).error)

    def test_nvidia_request_and_reply(self):
        seen = {}

        def post(url, body, headers, timeout):
            seen.update(url=url, body=body, headers=headers)
            return 200, {"choices": [{"message": {"content": '{"verdict": "A"}'}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1500, "completion_tokens": 7}}

        judge = NvidiaJudge("google/gemma-4-31b-it", api_key="k", rpm=0, post=post, sleep=no_sleep)
        reply = judge("sys", "user")
        self.assertEqual(judge.name, "nvidia_google_gemma-4-31b-it")
        self.assertTrue(seen["url"].endswith("/v1/chat/completions"))
        self.assertEqual(seen["headers"]["Authorization"], "Bearer k")
        self.assertEqual(seen["body"]["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(seen["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((parse_verdict(reply.text), reply.in_tokens, reply.out_tokens), ("A", 1500, 7))
        self.assertEqual(reply.thought_tokens, 0)

    def test_parse_openai_reasoning_and_length(self):
        data = {"choices": [{"message": {"content": "<think>hmm"}, "finish_reason": "length"}]}
        self.assertIn("limit", parse_openai(data).error)
        self.assertEqual(parse_openai({}).text, "")
        reasoned = {"choices": [{"message": {"content": '{"verdict": "B"}', "reasoning_content": "Let me see..."},
                                 "finish_reason": "stop"}], "usage": {"prompt_tokens": 900, "completion_tokens": 500}}
        reply = parse_openai(reasoned)
        self.assertEqual((parse_verdict(reply.text), reply.thought_tokens), ("B", 500))

    def test_pacing_spaces_requests_across_threads(self):
        judge = NvidiaJudge("m", api_key="k", rpm=60, monotonic=lambda: 100.0)
        self.assertEqual([judge._reserve() for _ in range(3)], [0.0, 1.0, 2.0])

    def test_hosted_retries_then_succeeds(self):
        responses = [(503, {"error": {"message": "overloaded"}}), (429, {"error": {"message": "per minute"}}),
                     (599, {"error": {"message": "network error"}}),
                     (200, {"steps": [{"type": "model_output", "content": [{"type": "text", "text": '"A"'}]}]})]
        judge = GeminiJudge("g", api_key="k", rpm=0, post=lambda *a: responses.pop(0), sleep=no_sleep)
        self.assertEqual(parse_verdict(judge("s", "u").text), "A")

    def test_hosted_daily_limit_stops(self):
        judge = NvidiaJudge("m", api_key="k", rpm=0, sleep=no_sleep,
                            post=lambda *a: (429, {"error": {"code": "quota_exceeded", "message": "daily quota"}}))
        with self.assertRaises(DailyLimitReached):
            judge("s", "u")

    def test_hosted_persistent_rate_limit_stops(self):
        judge = NvidiaJudge("m", api_key="k", rpm=0, max_retries=2, sleep=no_sleep,
                            post=lambda *a: (429, {"error": {"message": "too many requests"}}))
        with self.assertRaises(DailyLimitReached):
            judge("s", "u")

    def test_post_turns_connection_errors_into_599(self):
        for exc in (http.client.RemoteDisconnected("closed"), TimeoutError("read timed out"),
                    ConnectionResetError("reset"), urllib.error.URLError("no route")):
            with mock.patch("urllib.request.urlopen", side_effect=exc):
                status, data = _post("https://example.invalid", {}, {}, 1)
            self.assertEqual(status, 599)
            self.assertIn("network error", data["error"]["message"])

    def test_hosted_server_errors_raise_call_failed(self):
        judge = NvidiaJudge("m", api_key="k", rpm=0, max_retries=2, sleep=no_sleep,
                            post=lambda *a: (599, {"error": {"message": "network error: timed out"}}))
        with self.assertRaises(CallFailed):
            judge("s", "u")

    def test_rate_limit_backoff_grows_to_the_cap(self):
        waits = []
        judge = NvidiaJudge("m", api_key="k", rpm=0, max_retries=9, max_backoff=300, sleep=waits.append,
                            post=lambda *a: (429, {"error": {"message": "too many requests"}}))
        with self.assertRaises(DailyLimitReached):
            judge("s", "u")
        self.assertEqual([int(w) for w in waits], [2, 4, 8, 16, 32, 64, 128, 256, 300])

    def test_hosted_bad_request_raises(self):
        judge = GeminiJudge("g", api_key="k", rpm=0, sleep=no_sleep,
                            post=lambda *a: (400, {"error": {"message": "bad field"}}))
        with self.assertRaises(RuntimeError):
            judge("s", "u")

    def test_hosted_needs_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "NVIDIA_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                GeminiJudge("g")
            with self.assertRaises(RuntimeError):
                make_judge("nvidia:some/model")

    def test_make_judge(self):
        self.assertEqual(make_judge("ollama:qwen3:8b-q4_K_M").name, "ollama_qwen3-8b-q4_K_M")
        with self.assertRaises(ValueError):
            make_judge("other:model")
        with self.assertRaises(ValueError):
            make_judge("ollama")


class TestRun(unittest.TestCase):
    def tasks(self, n):
        return [(item(f"i{k}"), o) for k in range(n) for o in ("ab", "ba")]

    def test_run_saves_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fake.jsonl"
            stats = run(FakeJudge(), self.tasks(3), out, limit=4, progress_every=100)
            self.assertEqual(stats["done"], 4)
            stats = run(FakeJudge(), self.tasks(3), out, progress_every=100)
            self.assertEqual(stats["done"], 2)  # only the rest
            done = load_done(out)
            self.assertEqual(len(done), 6)
            # The fake judge always picks the right answer, so model_a wins in both orders.
            self.assertTrue(all(r["score"] == 1.0 for r in done.values()))

    def test_run_stops_at_daily_limit(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fake.jsonl"
            stats = run(FakeJudge(fail_after=3), self.tasks(5), out, progress_every=100)
            self.assertEqual((stats["done"], stats.get("stopped")), (3, 1))
            self.assertEqual(len(load_done(out)), 3)

    def test_load_done_ignores_half_line(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.jsonl"
            out.write_text(json.dumps({"id": "a", "order": "ab", "verdict": "A"}) + "\n" + '{"id": "b", "ord')
            self.assertEqual(list(load_done(out)), [("a", "ab")])

    def test_unreadable_verdicts_are_retried(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.jsonl"
            out.write_text(json.dumps({"id": "i0", "order": "ab", "verdict": None, "raw": "hmm"}) + "\n")
            self.assertEqual(load_done(out), {})
            stats = run(FakeJudge(), self.tasks(1)[:1], out, progress_every=100)
            self.assertEqual(stats["done"], 1)
            self.assertEqual(load_done(out)[("i0", "ab")]["verdict"], "A")
            self.assertEqual(len(out.read_text().splitlines()), 2)  # the old line stays

    def test_reparse_fixes_saved_replies_without_calls(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.jsonl"
            lines = [
                {"id": "a", "order": "ba", "verdict": None, "score": None, "raw": '{"verdict": "a"}'},
                {"id": "b", "order": "ab", "verdict": None, "score": None, "raw": "no idea"},
                {"id": "c", "order": "ab", "verdict": "B", "score": 0.0},
            ]
            out.write_text("".join(json.dumps(r) + "\n" for r in lines))
            self.assertEqual(reparse(out), 1)
            done = load_done(out)
            self.assertEqual(done[("a", "ba")]["verdict"], "A")
            self.assertEqual(done[("a", "ba")]["score"], 0.0)  # A shown first in "ba" is model_b
            self.assertTrue(done[("a", "ba")]["reparsed"])
            self.assertNotIn(("b", "ab"), done)
            self.assertEqual(reparse(out), 0)  # running it again changes nothing

    def test_run_parallel_saves_everything_once(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fake.jsonl"
            stats = run(FakeJudge(), self.tasks(20), out, workers=4, progress_every=100)
            self.assertEqual(stats["done"], 40)
            lines = out.read_text().splitlines()
            self.assertEqual(len(lines), 40)
            self.assertEqual(len({(json.loads(l)["id"], json.loads(l)["order"]) for l in lines}), 40)

    def test_run_parallel_stops_at_daily_limit(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fake.jsonl"
            stats = run(FakeJudge(fail_after=5), self.tasks(20), out, workers=3, progress_every=100)
            self.assertEqual(stats.get("stopped"), 1)
            self.assertEqual(len(load_done(out)), stats["done"])
            self.assertLess(stats["done"], 40)

    def test_run_skips_failing_calls_and_stops_if_many_fail(self):
        class Flaky(FakeJudge):
            def __call__(self, system, user):
                if "i1" in user or "i3" in user:
                    raise CallFailed("timed out")
                return super().__call__(system, user)

        tasks = [(replace(item(f"i{k}"), prompt=f"question i{k}"), o) for k in range(5) for o in ("ab", "ba")]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fake.jsonl"
            stats = run(Flaky(), tasks, out, progress_every=100)
            self.assertEqual((stats["done"], stats["skipped"]), (6, 4))
            self.assertNotIn("stopped", stats)
            stats = run(Flaky(), tasks, out, progress_every=100, max_skips=2)
            # Stops after 2 failures; calls already under way may still finish (and fail).
            self.assertEqual((stats["done"], stats["stopped"]), (0, 1))
            self.assertGreaterEqual(stats["skipped"], 2)

    def test_run_reraises_refused_request(self):
        class Refusing(FakeJudge):
            def __call__(self, system, user):
                if self.calls >= 2:
                    raise RuntimeError("400 bad request")
                return super().__call__(system, user)

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fake.jsonl"
            with self.assertRaises(RuntimeError):
                run(Refusing(), self.tasks(5), out, progress_every=100)
            self.assertEqual(len(load_done(out)), 2)  # what finished before the refusal is kept


if __name__ == "__main__":
    unittest.main()
