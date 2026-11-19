"""LMArena human preference votes (lmarena-ai/arena-human-preference-140k, CC-BY-4.0).

Each row is one person comparing two anonymous models on one prompt. Phase 2 uses
these votes as the human verdicts.

`fetch_votes` reads only a few small columns from the dataset's parquet files
(about 7 MB of the 1.6 GB, because parquet stores columns separately) and saves
them to a local file, so the network is used once. The response texts, which the
judges need, are fetched separately for the chosen pairs only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .data import winner_to_score

ARENA_REPO = "lmarena-ai/arena-human-preference-140k"
N_FILES = 7
DEFAULT_CACHE = Path("data/arena_votes.parquet")

# Parquet column path -> name in the local file.
COLUMNS = {
    "id": "id",
    "model_a": "model_a",
    "model_b": "model_b",
    "winner": "winner",
    "language": "language",
    "conv_metadata.turns": "turns",
    "conv_metadata.sum_user_tokens": "user_tokens",
    "conv_metadata.sum_assistant_a_tokens": "a_tokens",
    "conv_metadata.sum_assistant_b_tokens": "b_tokens",
}


@dataclass(frozen=True)
class Vote:
    id: str
    model_a: str
    model_b: str
    score: float  # from model_a's view: 1 = A wins, 0.5 = tie or both bad, 0 = B wins
    language: str
    turns: int
    user_tokens: int
    a_tokens: int
    b_tokens: int

    def canonical(self) -> Tuple[Tuple[str, str], float]:
        """(alphabetical pair, score from the alphabetically-first model's view)."""
        if self.model_a <= self.model_b:
            return (self.model_a, self.model_b), self.score
        return (self.model_b, self.model_a), 1.0 - self.score


def _parquet_paths() -> List[str]:
    return [
        f"datasets/{ARENA_REPO}/data/train-{i:05d}-of-{N_FILES:05d}.parquet"
        for i in range(N_FILES)
    ]


def selected_bytes(columns: Iterable[str] = tuple(COLUMNS)) -> Tuple[int, int]:
    """(bytes of `columns`, bytes of the whole dataset), from file footers.

    A column is matched by its full path ("conv_metadata.turns") or by its
    top-level name ("conversation_a" covers all its nested parts).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    wanted = set(columns)
    fs = HfFileSystem()
    selected = total = 0
    for path in _parquet_paths():
        with fs.open(path, "rb", block_size=1 << 16) as f:
            md = pq.ParquetFile(f).metadata
        for rg in range(md.num_row_groups):
            group = md.row_group(rg)
            for c in range(group.num_columns):
                col = group.column(c)
                total += col.total_compressed_size
                name = col.path_in_schema
                if name in wanted or name.split(".")[0] in wanted:
                    selected += col.total_compressed_size
    return selected, total


def fetch_votes(cache: Path = DEFAULT_CACHE, max_bytes: int = 50_000_000) -> Path:
    """Read the needed columns from Hugging Face and save them to `cache`.

    Refuses if the footers report more than `max_bytes` for those columns.
    Needs the optional dependency: pip install -e ".[data]"
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    selected, _ = selected_bytes()
    if selected > max_bytes:
        raise RuntimeError(
            f"needed columns are {selected / 1e6:.1f} MB, over the {max_bytes / 1e6:.0f} MB cap"
        )
    fs = HfFileSystem()
    tables = []
    for path in _parquet_paths():
        # Small blocks so only the needed column chunks travel over the network.
        with fs.open(path, "rb", block_size=1 << 16) as f:
            table = pq.read_table(f, columns=list(COLUMNS))
        # pyarrow returns a nested column under its last name ("turns", not
        # "conv_metadata.turns").
        read_names = [path.split(".")[-1] for path in COLUMNS]
        tables.append(table.select(read_names).rename_columns(list(COLUMNS.values())))
    out = pa.concat_tables(tables)
    if set(out.column("model_a").to_pylist()) & {None, ""}:
        raise ValueError("some votes have no model name")
    cache.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, cache)
    return cache


def load_votes(cache: Path = DEFAULT_CACHE) -> List[Vote]:
    """Load the saved votes (run `fetch_votes` first)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(cache).to_pylist()
    return [
        Vote(
            id=r["id"],
            model_a=r["model_a"],
            model_b=r["model_b"],
            score=winner_to_score(r["winner"]),
            language=r["language"] or "",
            turns=int(r["turns"] or 0),
            user_tokens=int(r["user_tokens"] or 0),
            a_tokens=int(r["a_tokens"] or 0),
            b_tokens=int(r["b_tokens"] or 0),
        )
        for r in rows
    ]


DEFAULT_TEXTS = Path("data/arena_texts.parquet")
TEXT_COLUMNS = ("id", "conversation_a", "conversation_b")


def message_text(message: dict) -> Tuple[str, bool]:
    """(text of one conversation message, whether it also held non-text parts)."""
    parts = message.get("content") or []
    texts = [p.get("text") or "" for p in parts if p.get("type") == "text"]
    other = any(p.get("type") != "text" or p.get("image") for p in parts)
    return "\n".join(t for t in texts if t), other


def conversation_texts(conv_a: List[dict], conv_b: List[dict]) -> Dict[str, object]:
    """Prompt and the two answers of a single-turn vote."""
    users = [m for m in conv_a if m.get("role") == "user"]
    answers_a = [m for m in conv_a if m.get("role") == "assistant"]
    answers_b = [m for m in conv_b if m.get("role") == "assistant"]
    if len(users) != 1 or len(answers_a) != 1 or len(answers_b) != 1:
        raise ValueError("expected one user message and one answer per side")
    prompt, p_other = message_text(users[0])
    answer_a, a_other = message_text(answers_a[0])
    answer_b, b_other = message_text(answers_b[0])
    return {
        "prompt": prompt,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "has_non_text": p_other or a_other or b_other,
    }


def fetch_texts(ids: Iterable[str], out: Path = DEFAULT_TEXTS, max_bytes: int = 800_000_000) -> Path:
    """Fetch the prompt and both answers for the given vote ids and save them to `out`.

    Reads only the id and conversation columns (about 690 MB over the network,
    because the chosen votes are spread over every row group), keeps the chosen
    rows and saves them (a few MB). Refuses if the footers report more than
    `max_bytes` for those columns.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    wanted = sorted(set(ids))
    selected, _ = selected_bytes(TEXT_COLUMNS)
    if selected > max_bytes:
        raise RuntimeError(
            f"conversation columns are {selected / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB cap"
        )
    fs = HfFileSystem()
    wanted_arr = pa.array(wanted)
    rows: List[Dict[str, object]] = []
    for i, path in enumerate(_parquet_paths(), 1):
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg, columns=list(TEXT_COLUMNS))
                table = table.filter(pc.is_in(table.column("id"), value_set=wanted_arr))
                for r in table.to_pylist():
                    rows.append({"id": r["id"], **conversation_texts(r["conversation_a"], r["conversation_b"])})
        print(f"  file {i}/{N_FILES}: {len(rows)} of {len(wanted)} items so far")
    missing = set(wanted) - {r["id"] for r in rows}
    if missing:
        raise RuntimeError(f"{len(missing)} ids not found in the dataset")
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    return out


def load_texts(path: Path = DEFAULT_TEXTS) -> Dict[str, Dict[str, object]]:
    """id -> {prompt, answer_a, answer_b, has_non_text} (run `fetch_texts` first)."""
    import pyarrow.parquet as pq

    return {r["id"]: r for r in pq.read_table(path).to_pylist()}


def filter_votes(
    votes: Iterable[Vote], language: str = "en", single_turn: bool = True
) -> List[Vote]:
    """Keep votes in one language and, optionally, single-turn conversations only."""
    return [
        v
        for v in votes
        if v.language == language
        and (not single_turn or v.turns == 1)
        and v.model_a != v.model_b
    ]


def pair_counts(votes: Iterable[Vote]) -> Dict[Tuple[str, str], int]:
    """Number of votes per alphabetical model pair."""
    return dict(Counter(v.canonical()[0] for v in votes))
