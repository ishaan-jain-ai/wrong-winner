import csv
from pathlib import Path

import pytest

from wrong_winner.figures import FIGURES, SHORT_NAMES, build_all, pair_label, spread_labels

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/phase2/analysis"


def test_spread_labels_keeps_order_and_gap():
    values = [0.43, 0.46, 0.415, 0.38, 0.14]
    out = spread_labels(values, 0.024)
    ranked = sorted(range(len(values)), key=values.__getitem__)
    assert sorted(range(len(out)), key=out.__getitem__) == ranked
    positions = sorted(out)
    assert all(b - a >= 0.024 - 1e-12 for a, b in zip(positions, positions[1:]))
    # Colliding labels are centred on their values, so the group doesn't drift.
    group = [i for i in range(4)]
    assert sum(out[i] for i in group) == pytest.approx(sum(values[i] for i in group))


def test_spread_labels_leaves_separated_values_alone():
    assert spread_labels([0.1, 0.5, 0.9], 0.05) == [0.1, 0.5, 0.9]
    assert spread_labels([0.3], 0.05) == [0.3]


def test_every_model_in_the_pairs_has_a_short_name():
    with (ROOT / "results/phase2/pairs.csv").open(newline="") as f:
        models = {m for row in csv.DictReader(f) for m in (row["model_a"], row["model_b"])}
    assert models <= set(SHORT_NAMES)
    assert pair_label("chatgpt-4o-latest-20250326 vs o3-2025-04-16") == "ChatGPT-4o vs o3"


def test_build_all_writes_every_figure(tmp_path):
    pytest.importorskip("matplotlib")
    if not (ANALYSIS / "summary.csv").exists():
        pytest.skip("no analysis results")
    written = build_all(ANALYSIS, tmp_path, formats=("png",), dpi=60)
    assert [p.name for p in written] == [f"fig{i}_{name}.png" for i, name in enumerate(FIGURES, 1)]
    assert all(p.stat().st_size > 10_000 for p in written)
