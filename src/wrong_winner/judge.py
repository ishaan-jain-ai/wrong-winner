"""Phase 2 judges: a local model through Ollama, and large models on free hosted tiers.

Each item is one arena vote: a user prompt and two answers. The judge sees the
prompt and both answers in one order and returns a verdict: "A", "B" or "tie".

Every judge follows the same plan (results/phase2/plan.csv, the lean design of
18 Nov): 80 items from each of the 15 close pairs and 20 from each of the 10
clear pairs, all at most 4,000 tokens, each shown in one answer order picked by
a fixed coin flip, plus 100 close items shown in both orders to measure
position bias. About 1,500 calls per judge.

Every verdict is appended to a JSONL file as soon as it arrives, so a run can
stop at any point (or hit a daily limit) and resume where it left off; calls
already made are never sent again. Calls run in a fixed shuffled order, so a
partial run still covers every pair.

Usage:
    python -m wrong_winner.judge --fetch-texts      # one-off: prompts and answers (~690 MB read)
    python -m wrong_winner.judge --make-plan        # writes results/phase2/plan.csv
    python -m wrong_winner.judge --judge ollama:qwen3:8b-q4_K_M
    python -m wrong_winner.judge --judge nvidia:google/gemma-4-31b-it --limit 5
    python -m wrong_winner.judge --judge gemini:gemini-3.5-flash-lite

Uses only the standard library for HTTP. Hosted judges read their key from the
environment (NVIDIA_API_KEY, GEMINI_API_KEY); keys are never printed or saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ITEMS_CSV = Path("results/phase2/items.csv")
PLAN_CSV = Path("results/phase2/plan.csv")
JUDGMENTS_DIR = Path("results/phase2/judgments")

PLAN_PER_PAIR = {"close": 80, "clear": 20}  # items per pair; groups not listed are left out
PLAN_MAX_TOKENS = 4_000  # dataset count plus judge instructions
PLAN_BOTH_ORDERS = 100  # close items shown both ways, for position bias

SYSTEM_PROMPT = """You are an impartial judge comparing two AI assistants' answers to the same user request.

Decide which answer the user would prefer. Weigh helpfulness, correctness, relevance, depth and clarity, and respect any instructions the user gave about format or length. Do not let the order of the answers, their length on its own, or their style sway you.

If the two answers are about equally good, or both are poor, the verdict is "tie".

Reply with JSON only: {"verdict": "A"}, {"verdict": "B"} or {"verdict": "tie"}."""

USER_TEMPLATE = """<user_request>
{prompt}
</user_request>

<answer_a>
{answer_a}
</answer_a>

<answer_b>
{answer_b}
</answer_b>"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["A", "B", "tie"]}},
    "required": ["verdict"],
}

ORDERS = ("ab", "ba")  # "ab": model_a's answer shown first, as the voter saw it

Task = Tuple["Item", str]  # (item, order)


# ---------------------------------------------------------------------------
# Items and the plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    id: str
    group: str
    model_a: str
    model_b: str
    human_score: float
    tokens: int
    prompt: str = ""
    answer_a: str = ""
    answer_b: str = ""

    @property
    def pair(self) -> Tuple[str, str]:
        return tuple(sorted((self.model_a, self.model_b)))  # type: ignore[return-value]


def read_items(path: Path = ITEMS_CSV) -> List[Item]:
    with path.open(newline="") as f:
        return [
            Item(r["id"], r["group"], r["model_a"], r["model_b"], float(r["human_score"]), int(r["tokens"]))
            for r in csv.DictReader(f)
        ]


def usable(text: Dict[str, object]) -> bool:
    """False for items holding images or files, and for the few rows with an empty
    prompt or answer (the voter may have seen something the row lacks)."""
    return not text["has_non_text"] and bool(text["prompt"] and text["answer_a"] and text["answer_b"])


def with_text(item: Item, texts: Dict[str, Dict[str, object]]) -> Item:
    t = texts[item.id]
    return replace(item, prompt=t["prompt"], answer_a=t["answer_a"], answer_b=t["answer_b"])


def orders_for(item_id: str, mode: str, seed: int = 0) -> Tuple[str, ...]:
    """Both orders, or one order picked by a stable coin flip per item."""
    if mode == "both":
        return ORDERS
    if mode == "one":
        digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
        return (ORDERS[digest[0] % 2],)
    raise ValueError("mode must be 'both' or 'one'")


def make_plan(
    items: Sequence[Item],
    texts: Dict[str, Dict[str, object]],
    per_pair: Dict[str, int] = PLAN_PER_PAIR,
    max_tokens: int = PLAN_MAX_TOKENS,
    both_orders: int = PLAN_BOTH_ORDERS,
    seed: int = 0,
) -> List[Tuple[Item, Tuple[str, ...]]]:
    """Which items each judge sees, and in which answer orders.

    Per pair, a seeded random sample of up to per_pair[group] usable items within
    max_tokens. Each gets one order by coin flip, except `both_orders` close items
    (a seeded sample) that get both.
    """
    by_pair: Dict[Tuple[str, str], List[Item]] = defaultdict(list)
    for it in items:
        if it.group in per_pair and it.tokens <= max_tokens and usable(texts[it.id]):
            by_pair[it.pair].append(it)
    rng = random.Random(seed)
    chosen: List[Item] = []
    for pair in sorted(by_pair):
        pool = sorted(by_pair[pair], key=lambda it: it.id)
        k = per_pair[pool[0].group]
        chosen += pool if len(pool) <= k else rng.sample(pool, k)
    close = sorted((it for it in chosen if it.group == "close"), key=lambda it: it.id)
    both = {it.id for it in rng.sample(close, min(both_orders, len(close)))}
    return [(it, ORDERS if it.id in both else orders_for(it.id, "one", seed)) for it in chosen]


PLAN_FIELDS = ("id", "group", "model_a", "model_b", "human_score", "tokens", "orders")


def write_plan(path: Path, plan: Sequence[Tuple[Item, Tuple[str, ...]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PLAN_FIELDS)
        for it, orders in plan:
            writer.writerow([it.id, it.group, it.model_a, it.model_b, it.human_score, it.tokens, " ".join(orders)])


def read_plan(path: Path = PLAN_CSV) -> List[Tuple[Item, Tuple[str, ...]]]:
    with path.open(newline="") as f:
        return [
            (
                Item(r["id"], r["group"], r["model_a"], r["model_b"], float(r["human_score"]), int(r["tokens"])),
                tuple(r["orders"].split()),
            )
            for r in csv.DictReader(f)
        ]


def plan_tasks(plan: Sequence[Tuple[Item, Tuple[str, ...]]], seed: int = 0) -> List[Task]:
    """Every (item, order) in the plan, in a fixed shuffled order."""
    tasks = [(it, o) for it, orders in plan for o in orders]
    return sorted(tasks, key=lambda t: hashlib.sha256(f"{seed}:{t[0].id}:{t[1]}".encode()).hexdigest())


# ---------------------------------------------------------------------------
# Prompts and verdicts
# ---------------------------------------------------------------------------


def build_messages(item: Item, order: str) -> Tuple[str, str]:
    """(system prompt, user message) with the answers in the given order."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    first, second = (item.answer_a, item.answer_b) if order == "ab" else (item.answer_b, item.answer_a)
    return SYSTEM_PROMPT, USER_TEMPLATE.format(prompt=item.prompt, answer_a=first, answer_b=second)


_THINKING = re.compile(r"<think>.*?</think>", re.S | re.I)
_VERDICT_FIELD = re.compile(r'"verdict"\s*:\s*"([^"]*)"')


def parse_verdict(text: str) -> Optional[str]:
    """"A", "B" or "tie" from the judge's reply, or None if it can't be read.

    Accepts the JSON asked for, also inside a code fence or after a <think> block,
    or a bare A / B / tie.
    """
    text = _THINKING.sub("", text or "").strip()
    match = _VERDICT_FIELD.search(text)
    if match:
        value = match.group(1)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
    if not isinstance(value, str):
        return None
    value = value.strip().strip("\"'.` ").strip().lower()
    if value in ("a", "b"):  # some judges write "a" / "b"
        return value.upper()
    if value == "tie":
        return "tie"
    return None


def verdict_score(verdict: Optional[str], order: str) -> Optional[float]:
    """Score from the item's model_a view (1 = model_a better), undoing the display order."""
    if verdict is None:
        return None
    if verdict == "tie":
        return 0.5
    first_wins = verdict == "A"
    model_a_shown_first = order == "ab"
    return 1.0 if first_wins == model_a_shown_first else 0.0


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    thought_tokens: int = 0
    error: str = ""


class DailyLimitReached(RuntimeError):
    """The provider's quota is used up; rerun after it resets."""


class CallFailed(RuntimeError):
    """One call kept failing (timeouts, server errors). The run skips it; the next run retries it."""


def _post(url: str, body: dict, headers: Dict[str, str], timeout: float) -> Tuple[int, dict]:
    return _request(url, body, headers, timeout, lambda resp: json.loads(resp.read() or b"{}"))


def _post_stream(url: str, body: dict, headers: Dict[str, str], timeout: float) -> Tuple[int, dict]:
    """Like _post for a streamed reply, joined into one completion. Here `timeout` is
    the longest silence allowed between pieces, not a limit on the whole reply."""
    return _request(url, body, headers, timeout, collect_stream)


def collect_stream(lines: Iterable[bytes]) -> dict:
    """Join a streamed OpenAI-style chat completion (server-sent events) into the dict a
    normal reply would have, so parse_openai reads both the same way.

    Raises ConnectionError if the stream ends before the reply is complete."""
    content: List[str] = []
    reasoning: List[str] = []
    finish, usage, pieces, complete = None, None, 0, False
    for raw in lines:
        line = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw).strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            complete = True
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if event.get("error"):
            raise ConnectionError(f"error in stream: {json.dumps(event['error'])[:300]}")
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
                pieces += 1
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                reasoning.append(thought)
                pieces += 1
            if choice.get("finish_reason"):
                finish, complete = choice["finish_reason"], True
    if not complete:
        raise ConnectionError(f"stream ended early after {pieces} pieces")
    message = {"content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    # Without a usage report, each piece is about one token.
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": usage or {"completion_tokens": pieces}}


def _request(url: str, body: dict, headers: Dict[str, str], timeout: float,
             read: Callable) -> Tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, read(resp)
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": {"message": raw.decode(errors="replace")[:500]}}
    except (OSError, http.client.HTTPException) as e:  # timeouts, refused or dropped connections
        return 599, {"error": {"message": f"network error: {type(e).__name__}: {e}"}}


class OllamaJudge:
    """A model served by Ollama on this machine. Thinking off, temperature 0."""

    def __init__(self, model: str, num_ctx: int = 16_384, host: str = "http://localhost:11434",
                 timeout: float = 600, post: Callable = _post):
        self.model, self.num_ctx, self.host, self.timeout, self._post = model, num_ctx, host, timeout, post
        self.name = "ollama_" + model.replace(":", "-").replace("/", "-")

    def __call__(self, system: str, user: str) -> Reply:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "format": VERDICT_SCHEMA,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 32},
            "keep_alive": "30m",
        }
        status, data = self._post(f"{self.host}/api/chat", body, {}, self.timeout)
        if status != 200:
            raise RuntimeError(f"Ollama returned {status}: {data.get('error', data)}")
        reply = Reply(
            text=(data.get("message") or {}).get("content", ""),
            in_tokens=int(data.get("prompt_eval_count") or 0),
            out_tokens=int(data.get("eval_count") or 0),
        )
        if reply.in_tokens + 32 >= self.num_ctx:
            reply.error = "prompt may have been cut to fit the context window"
        return reply


class HostedJudge:
    """Shared pacing and retries for judges behind a free hosted API.

    Paces requests to `rpm` per minute, retries transient errors (408, 429, 5xx,
    network) with backoff, and raises DailyLimitReached when the quota is used up
    or rate limiting doesn't clear.
    """

    URL = ""
    KEY_ENV = ""

    def __init__(self, model: str, api_key: Optional[str] = None, rpm: float = 10, timeout: float = 300,
                 max_retries: int = 8, max_backoff: float = 300, post: Callable = _post,
                 sleep: Callable = time.sleep, monotonic: Callable[[], float] = time.monotonic):
        key = api_key if api_key is not None else os.environ.get(self.KEY_ENV, "")
        if not key:
            raise RuntimeError(f"set {self.KEY_ENV} in your terminal first")
        self.model, self._key, self.timeout, self.max_retries = model, key, timeout, max_retries
        self.max_backoff = max_backoff
        self._post, self._sleep, self._monotonic = post, sleep, monotonic
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def _reserve(self) -> float:
        """Seconds to wait before this request's start slot. Safe across worker threads."""
        with self._lock:
            now = self._monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        return slot - now

    def _pace(self) -> None:
        wait = self._reserve()
        if wait > 0:
            self._sleep(wait)

    def _send(self, body: dict, headers: Dict[str, str]) -> dict:
        """POST with retries. Rate limits (429) back off for up to `max_backoff` seconds a
        time; timeouts and server errors up to a minute. Raises DailyLimitReached when the
        quota is used up or rate limiting never clears, CallFailed when one call keeps
        failing, and RuntimeError for a refused request (400, 401, 403, 404...)."""
        for attempt in range(self.max_retries + 1):
            self._pace()
            status, data = self._post(self.URL, body, headers, self.timeout)
            if status == 200:
                return data
            message = json.dumps(data.get("error", data) if isinstance(data, dict) else data)[:500]
            if status == 429 and _is_daily_limit(message):
                raise DailyLimitReached(message)
            if status in (408, 429) or status >= 500:
                if attempt < self.max_retries:
                    cap = self.max_backoff if status == 429 else 60
                    self._sleep(min(2 ** (attempt + 1), cap) + random.random())
                    continue
                if status == 429:
                    raise DailyLimitReached(f"still rate-limited after {self.max_retries} retries: {message}")
                raise CallFailed(f"{status} after {self.max_retries} retries: {message}")
            raise RuntimeError(f"{type(self).__name__} got {status}: {message}")
        raise AssertionError("unreachable")


def _is_daily_limit(message: str) -> bool:
    m = message.lower()
    return any(s in m for s in ("quota_exceeded", "daily", "perday", "per day"))


class GeminiJudge(HostedJudge):
    """Gemini through the Interactions API (free tier). Thinking at its minimum."""

    URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
    KEY_ENV = "GEMINI_API_KEY"
    API_REVISION = "2026-05-20"

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.name = "gemini_" + model

    def __call__(self, system: str, user: str) -> Reply:
        body = {
            "model": self.model,
            "system_instruction": system,
            "input": user,
            "generation_config": {"thinking_level": "minimal", "max_output_tokens": 256},
            "response_format": {"type": "text", "mime_type": "application/json", "schema": VERDICT_SCHEMA},
        }
        return parse_gemini(self._send(body, {"x-goog-api-key": self._key, "Api-Revision": self.API_REVISION}))


def parse_gemini(data: dict) -> Reply:
    """Text and token counts from an Interactions API response."""
    texts = [
        c.get("text", "")
        for step in data.get("steps") or []
        if step.get("type") == "model_output"
        for c in step.get("content") or []
        if c.get("type") == "text"
    ]
    text = "".join(texts) or data.get("output_text", "")
    usage = data.get("usage") or {}
    reply = Reply(
        text=text,
        in_tokens=int(usage.get("total_input_tokens") or 0),
        out_tokens=int(usage.get("total_output_tokens") or 0),
        thought_tokens=int(usage.get("total_thought_tokens") or 0),
    )
    if data.get("status") == "incomplete":
        reply.error = "incomplete response"
    return reply


class NvidiaJudge(HostedJudge):
    """A model on NVIDIA's free endpoints (build.nvidia.com), OpenAI-style chat API.

    Asks for thinking off (`enable_thinking: false`). Models that honour it
    (Nemotron) answer in a few tokens; models that always reason (Mistral Large 4
    on this endpoint) put the reasoning in a separate field and still answer in
    `content`, so the output limit leaves room for it. No JSON schema (the docs
    don't say it's supported): the prompt asks for JSON and parse_verdict reads it.
    """

    URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    KEY_ENV = "NVIDIA_API_KEY"

    def __init__(self, model: str, max_tokens: int = 4096, timeout: float = 900, stream: bool = False,
                 **kwargs):
        # Calls can wait in NVIDIA's queue for several minutes (Mistral Large 4: up to ~8).
        # Streaming sends the reply in pieces as it is written, which keeps long replies
        # from being dropped; the timeout then limits the silence between pieces.
        if stream:
            kwargs.setdefault("post", _post_stream)
        super().__init__(model, timeout=timeout, **kwargs)
        self.max_tokens, self.stream = max_tokens, stream
        self.name = "nvidia_" + model.replace("/", "_")

    def __call__(self, system: str, user: str) -> Reply:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return parse_openai(self._send(body, {"Authorization": f"Bearer {self._key}"}))


def parse_openai(data: dict) -> Reply:
    """Text and token counts from an OpenAI-style chat completion.

    When the model reasoned in a separate field, all its output tokens are
    counted as thought tokens (the API doesn't split them).
    """
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    out = int(usage.get("completion_tokens") or 0)
    reply = Reply(
        text=message.get("content") or "",
        in_tokens=int(usage.get("prompt_tokens") or 0),
        out_tokens=out,
        thought_tokens=out if message.get("reasoning_content") else 0,
    )
    if choice.get("finish_reason") == "length":
        reply.error = "hit the output limit"
    return reply


# NVIDIA's free tier allows about 40 requests a minute per account, and its calls
# wait in a queue (up to a couple of minutes), so several run at once.
DEFAULT_RPM = {"gemini": 10, "nvidia": 35}
DEFAULT_WORKERS = {"ollama": 1, "gemini": 2, "nvidia": 16}


def backend_of(spec: str) -> str:
    return spec.partition(":")[0]


STREAM_TIMEOUT = 1200  # longest silence while streaming: covers NVIDIA's queue (up to ~8 min)


def make_judge(spec: str, num_ctx: int = 16_384, rpm: Optional[float] = None,
               max_tokens: Optional[int] = None, stream: bool = False):
    """`max_tokens` (NVIDIA only) raises the reply limit for models that reason at length;
    the call timeout grows with it (15 minutes per 4,096 tokens). `stream` (NVIDIA only)
    streams replies instead, with the timeout limiting the silence between pieces."""
    backend, _, model = spec.partition(":")
    if not model:
        raise ValueError("judge must look like ollama:<model>, nvidia:<model> or gemini:<model>")
    if (max_tokens or stream) and backend != "nvidia":
        raise ValueError("max_tokens and stream are only supported for nvidia judges")
    if backend == "ollama":
        return OllamaJudge(model, num_ctx=num_ctx)
    if backend == "gemini":
        return GeminiJudge(model, rpm=rpm or DEFAULT_RPM["gemini"])
    if backend == "nvidia":
        limits: Dict[str, object] = {}
        if max_tokens:
            limits.update(max_tokens=max_tokens, timeout=900 * max(1.0, max_tokens / 4096))
        if stream:
            limits.update(stream=True, timeout=STREAM_TIMEOUT)
        return NvidiaJudge(model, rpm=rpm or DEFAULT_RPM["nvidia"], **limits)
    raise ValueError(f"unknown judge backend {backend!r}")


# ---------------------------------------------------------------------------
# Running with a cache
# ---------------------------------------------------------------------------


def load_done(path: Path) -> Dict[Tuple[str, str], dict]:
    """(item id, order) -> latest saved record with a readable verdict.

    Unreadable verdicts are left out, so the next run tries them again (their old
    lines stay in the file). Ignores a half-written last line.
    """
    done: Dict[Tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("verdict") is not None:
                done[(rec["id"], rec["order"])] = rec
    return done


def reparse(path: Path) -> int:
    """Read unreadable replies again with the current parser, without calling the judge.

    For each unreadable record whose saved reply now parses, appends a corrected
    record marked "reparsed" (the original line stays). Returns how many were fixed.
    """
    if not path.exists():
        return 0
    done = load_done(path)
    fixed = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec["id"], rec["order"])
        if rec.get("verdict") is not None or key in done or "raw" not in rec:
            continue
        verdict = parse_verdict(rec["raw"])
        if verdict is None:
            continue
        new = {k: v for k, v in rec.items() if k != "raw"}
        new.update(verdict=verdict, score=verdict_score(verdict, rec["order"]), reparsed=True)
        done[key] = new
        fixed.append(new)
    with path.open("a") as f:
        for rec in fixed:
            f.write(json.dumps(rec) + "\n")
    return len(fixed)


def _judge_one(judge, item: Item, order: str, stop: threading.Event, clock: Callable[[], float]):
    """One call, in a worker thread. Returns a record, or None if the run is stopping."""
    if stop.is_set():
        return None
    system, user = build_messages(item, order)
    start = clock()
    reply = judge(system, user)
    took = clock() - start
    verdict = parse_verdict(reply.text)
    rec = {
        "id": item.id,
        "order": order,
        "verdict": verdict,
        "score": verdict_score(verdict, order),
        "in_tokens": reply.in_tokens,
        "out_tokens": reply.out_tokens,
        "thought_tokens": reply.thought_tokens,
        "seconds": round(took, 2),
    }
    if verdict is None:
        rec["raw"] = reply.text[:300]
    if reply.error:
        rec["error"] = reply.error
    return rec


def run(judge, tasks: Sequence[Task], out: Path, limit: Optional[int] = None, workers: int = 1,
        progress_every: int = 25, clock: Callable[[], float] = time.monotonic,
        max_skips: int = 25) -> Dict[str, int]:
    """Judge every (item, order) without a readable verdict in `out`. Returns counts.

    With workers > 1, calls run in parallel threads (hosted judges pace themselves);
    records are written from this thread only, in the order they finish. A call that
    keeps failing is skipped (and retried by the next run); after `max_skips` of them
    the run stops, since the service is probably down.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    remaining = [(it, o) for it, o in tasks if (it.id, o) not in done]
    todo = remaining if limit is None else remaining[:limit]
    print(f"{judge.name}: {len(tasks) - len(remaining)} of {len(tasks)} calls already saved, "
          f"{len(todo)} to run with {workers} worker(s)", flush=True)
    stats = {"done": 0, "unreadable": 0, "flagged": 0, "skipped": 0}
    stop = threading.Event()
    failure: Optional[BaseException] = None
    started = clock()
    with out.open("a") as f, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_judge_one, judge, it, o, stop, clock) for it, o in todo]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except DailyLimitReached as e:
                if not stop.is_set():
                    print(f"Stopped: limit reached ({e}). Rerun the same command after it resets.", flush=True)
                stop.set()
                stats["stopped"] = 1
                continue
            except CallFailed as e:  # skip this call; the next run tries it again
                stats["skipped"] += 1
                print(f"  skipped a call: {e}", flush=True)
                if stats["skipped"] >= max_skips and not stop.is_set():
                    print(f"Stopped: {max_skips} calls failed; the service looks down. Rerun later.", flush=True)
                    stop.set()
                    stats["stopped"] = 1
                continue
            except Exception as e:  # a request the API refused: stop the run and report it
                stop.set()
                failure = failure or e
                continue
            if rec is None:
                continue
            stats["unreadable"] += rec["verdict"] is None
            stats["flagged"] += "error" in rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            stats["done"] += 1
            if stats["done"] % progress_every == 0 or stats["done"] == len(todo):
                per_call = (clock() - started) / stats["done"]  # wall time, so parallel runs count right
                left = (len(todo) - stats["done"]) * per_call
                print(f"  {stats['done']}/{len(todo)}  {per_call:.1f}s per call  about {left / 3600:.1f} h left",
                      flush=True)
    if failure is not None:
        raise failure
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch-texts", action="store_true", help="download prompts and answers, then exit")
    parser.add_argument("--make-plan", action="store_true", help="write results/phase2/plan.csv, then exit")
    parser.add_argument("--reparse", action="store_true",
                        help="re-read unreadable saved replies with the current parser (no calls), then exit")
    parser.add_argument("--judge", help="ollama:<model>, nvidia:<model> or gemini:<model>")
    parser.add_argument("--groups", nargs="+", default=["close", "clear"])
    parser.add_argument("--limit", type=int, help="run at most this many calls")
    parser.add_argument("--num-ctx", type=int, default=16_384, help="Ollama context window")
    parser.add_argument("--rpm", type=float, help="requests per minute for hosted judges")
    parser.add_argument("--workers", type=int, help="calls in parallel (default: 1 local, 16 NVIDIA, 2 Gemini)")
    parser.add_argument("--max-tokens", type=int,
                        help="reply token limit for NVIDIA judges (default 4096); use for replies cut off mid-reasoning")
    parser.add_argument("--stream", action="store_true",
                        help="stream NVIDIA replies in pieces, which keeps long replies from being dropped")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from .arena import DEFAULT_TEXTS, fetch_texts, load_texts

    if args.reparse:
        for path in sorted(JUDGMENTS_DIR.glob("*.jsonl")):
            print(f"{path.name}: {reparse(path)} unreadable replies now read")
        return
    if args.fetch_texts:
        items = read_items()
        print(f"Fetching prompts and answers for {len(items)} items...")
        fetch_texts([it.id for it in items])
        print(f"Saved {DEFAULT_TEXTS}")
        return
    if not DEFAULT_TEXTS.exists():
        parser.error(f"{DEFAULT_TEXTS} is missing: run with --fetch-texts first")
    texts = load_texts()
    if args.make_plan:
        plan = make_plan(read_items(), texts, seed=args.seed)
        write_plan(PLAN_CSV, plan)
        calls = sum(len(o) for _, o in plan)
        for g in PLAN_PER_PAIR:
            rows = [(it, o) for it, o in plan if it.group == g]
            print(f"  {g}: {len({it.pair for it, _ in rows})} pairs, {len(rows)} items, "
                  f"{sum(len(o) for _, o in rows)} calls")
        print(f"Wrote {PLAN_CSV}: {len(plan)} items, {calls} calls per judge")
        return
    if not args.judge:
        parser.error("--judge is required (or use --fetch-texts / --make-plan)")
    if not PLAN_CSV.exists():
        parser.error(f"{PLAN_CSV} is missing: run with --make-plan first")

    plan = [(with_text(it, texts), orders) for it, orders in read_plan() if it.group in args.groups]
    tasks = plan_tasks(plan, args.seed)
    print(f"Plan: {len(plan)} items in groups {', '.join(args.groups)}, {len(tasks)} calls")
    judge = make_judge(args.judge, num_ctx=args.num_ctx, rpm=args.rpm, max_tokens=args.max_tokens,
                       stream=args.stream)
    workers = args.workers or DEFAULT_WORKERS.get(backend_of(args.judge), 1)
    stats = run(judge, tasks, JUDGMENTS_DIR / f"{judge.name}.jsonl", args.limit, workers)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
