"""The spanning-children switch of the one nucleus and cell ownership (schema 968).

``spacr.measure._assign_nuclei_to_cells(..., spanning=)`` is the one routine
Measure (``spanning_merge``) and OPS (``ops_ph_spanning_merge``) give nuclei
to cells through. ``'merge'`` (default) is the behaviour before the switch;
``'best_parent'`` skips ``_merge_spanning_children``, so OPS pairing is the
pairing of the frozen A1 run (commit 26bc722c15) again, byte for byte.

The frozen references below are copies, not imports: ``_pair_26bc722c15`` is
the body of ``spacr.ops_sample.pair_nuclei_with_cells`` at 26bc722c15 and
``_pair_before_switch`` its body on the 1023 base (ws-offload-ws-genotype-
defaults-1023 @ 9fcb3cf896), with the kernel written out as it was then.
"""
import ast
import json
import pathlib

import numpy as np
import pytest

from spacr import measure as M
from spacr import ops_sample

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACR = ROOT / "spacr"
A1_RAW = pathlib.Path("/media/carruthers/mnt2/spacr/opsmeasure_scratch/raw_masks")
A1_OLD = pathlib.Path("/media/carruthers/mnt2/spacr/opsmeasure_scratch/masks")
A1_SITES = (357, 575, 713, 1078)


def _pair_26bc722c15(nuclei, cells):
    """spacr.ops_sample.pair_nuclei_with_cells at 26bc722c15, verbatim."""
    nuclei = np.asarray(nuclei)
    cells = np.asarray(cells)
    if nuclei.shape != cells.shape:
        raise ValueError(
            f"nucleus and cell masks differ in shape: {nuclei.shape} and "
            f"{cells.shape}")
    n_flat = nuclei.ravel().astype(np.int64)
    c_flat = cells.ravel().astype(np.int64)
    both = (n_flat > 0) & (c_flat > 0)
    n_total = int(np.unique(n_flat[n_flat > 0]).size)
    c_total = int(np.unique(c_flat[c_flat > 0]).size)
    if not both.any():
        return (np.zeros(nuclei.shape, np.int32),
                np.zeros(cells.shape, np.int32),
                {"nuclei": n_total, "cells": c_total, "paired": 0,
                 "nuclei_without_cell": n_total,
                 "cells_without_nucleus": c_total,
                 "extra_nuclei_in_cell": 0})
    width = int(c_flat.max()) + 1
    pair, overlap = np.unique(n_flat[both] * width + c_flat[both],
                              return_counts=True)
    pair_n, pair_c = pair // width, pair % width
    order = np.lexsort((-overlap, pair_n))
    pair_n, pair_c, overlap = pair_n[order], pair_c[order], overlap[order]
    first = np.r_[True, pair_n[1:] != pair_n[:-1]]
    best_n, best_c, best_o = pair_n[first], pair_c[first], overlap[first]
    order = np.lexsort((-best_o, best_c))
    best_n, best_c = best_n[order], best_c[order]
    first = np.r_[True, best_c[1:] != best_c[:-1]]
    keep_n, keep_c = best_n[first], best_c[first]
    labels = np.arange(1, keep_n.size + 1, dtype=np.int32)
    n_lut = np.zeros(int(n_flat.max()) + 1, np.int32)
    c_lut = np.zeros(width, np.int32)
    n_lut[keep_n] = labels
    c_lut[keep_c] = labels
    new_cells = c_lut[cells.astype(np.int64)]
    new_nuclei = n_lut[nuclei.astype(np.int64)]
    new_nuclei[new_nuclei != new_cells] = 0
    counts = {
        "nuclei": n_total, "cells": c_total, "paired": int(keep_n.size),
        "nuclei_without_cell": int(n_total - np.unique(pair_n).size),
        "cells_without_nucleus": int(c_total - np.unique(pair_c).size),
        "extra_nuclei_in_cell": int(best_n.size - keep_n.size),
    }
    return new_nuclei, new_cells, counts


def _kernel_before_switch(nucleus_mask, cell_mask):
    """spacr.measure._assign_nuclei_to_cells on the 1023 base, verbatim."""
    nucleus_mask, cell_mask, merged = M._merge_spanning_children(nucleus_mask, cell_mask)
    nucleus_mask, shared_nuclei, shared_cells = M._assign_shared_nuclei(nucleus_mask, cell_mask)
    return nucleus_mask, cell_mask, shared_nuclei, shared_cells, merged


def _pair_before_switch(nuclei, cells):
    """spacr.ops_sample.pair_nuclei_with_cells on the 1023 base, verbatim."""
    nuclei = np.asarray(nuclei)
    cells = np.asarray(cells)
    before = int(np.unique(cells[cells > 0]).size)
    nuclei, cells, _shared_n, _shared_c, merged = _kernel_before_switch(
        nuclei.copy(), cells.copy())
    n_total = int(np.unique(nuclei[nuclei > 0]).size)
    c_total = int(np.unique(cells[cells > 0]).size)
    pair_n, pair_c, overlap = M._label_overlaps(nuclei, cells)
    if not pair_n.size:
        return (np.zeros(nuclei.shape, np.int32),
                np.zeros(cells.shape, np.int32),
                {"nuclei": n_total, "cells": before, "paired": 0,
                 "nuclei_without_cell": n_total,
                 "cells_without_nucleus": c_total,
                 "extra_nuclei_in_cell": 0, "merged_cells": merged})
    best_n, best_c, best_o = M._best_parent(pair_n, pair_c, overlap)
    keep_c, keep_n, _ = M._best_parent(best_c, best_n, best_o)
    labels = np.arange(1, keep_n.size + 1, dtype=np.int32)
    n_lut = np.zeros(int(nuclei.max()) + 1, np.int32)
    c_lut = np.zeros(int(cells.max()) + 1, np.int32)
    n_lut[keep_n] = labels
    c_lut[keep_c] = labels
    new_cells = c_lut[cells.astype(np.int64)]
    new_nuclei = n_lut[nuclei.astype(np.int64)]
    new_nuclei[new_nuclei != new_cells] = 0
    counts = {
        "nuclei": n_total, "cells": before, "paired": int(keep_n.size),
        "nuclei_without_cell": int(n_total - np.unique(pair_n).size),
        "cells_without_nucleus": int(c_total - np.unique(pair_c).size),
        "extra_nuclei_in_cell": int(best_n.size - keep_n.size),
        "merged_cells": merged,
    }
    return new_nuclei, new_cells, counts


def _same(got, want):
    """Byte identity of two ``(nuclei, cells, counts)``; ``want`` may lack merged_cells."""
    for a, b in zip(got[:2], want[:2]):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    extra = set(got[2]) - set(want[2])
    assert extra <= {"merged_cells"}
    assert {k: got[2][k] for k in want[2]} == want[2]
    if extra:
        assert got[2]["merged_cells"] == 0


def _handmade():
    """Spanning cases: split nuclei, trims, disconnected labels, ties, empty fields."""
    cases = []
    cells = np.zeros((20, 40), np.int32)
    cells[:, :20], cells[:, 20:] = 1, 2
    n = np.zeros_like(cells)
    n[8:12, 16:24] = 1                                  # 50/50 over two cells
    cases.append((n.copy(), cells.copy()))
    n[:] = 0
    n[8:12, 0:19] = 1
    n[8:12, 19:21] = 1                                  # 94.7 % in cell 1
    n[2:5, 30:33] = 2
    cases.append((n.copy(), cells.copy()))
    n[:] = 0
    n[2:4, 2:4] = 3                                     # one label, two pieces
    n[15:17, 30:33] = 3
    n[10:12, 10:12] = 1                                 # two nuclei in cell 1
    n[10:13, 5:8] = 2
    cases.append((n.copy(), cells.copy()))
    c3 = cells.copy()
    c3[:, 30:] = 0                                      # nucleus partly on background
    n[:] = 0
    n[5:9, 26:34] = 4
    n[12:14, 18:22] = 7                                 # exact tie, 4 px each
    cases.append((n.copy(), c3))
    three = np.zeros((30, 30), np.int32)
    three[:, :10], three[:, 10:20], three[:, 20:] = 5, 9, 2
    n = np.zeros_like(three)
    n[10:20, 5:25] = 1                                  # over three cells
    n[0:3, 0:3] = 2
    cases.append((n, three))
    cases.append((np.zeros((6, 6), np.int32), np.zeros((6, 6), np.int32)))
    z = np.zeros((6, 6), np.int32)
    z[1:3, 1:3] = 1
    cases.append((z, np.zeros((6, 6), np.int32)))       # nuclei but no cells
    return cases


def _random_cases(count=120, seed=20261125):
    """Voronoi cells with background and disk nuclei placed across their borders."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        h, w = rng.integers(30, 80, size=2)
        k = int(rng.integers(2, 12))
        seeds = rng.uniform((0, 0), (h, w), size=(k, 2))
        yy, xx = np.mgrid[:h, :w]
        dist = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
        cells = (np.argmin(dist, axis=-1) + 1).astype(np.int32)
        cells[rng.random((h, w)) < 0.02] = 0
        cells[np.min(dist, axis=-1) > rng.uniform(80, 400)] = 0
        perm = np.r_[0, rng.permutation(k) + 1 + int(rng.integers(0, 5))]
        cells = perm[cells].astype(np.int32)
        nuclei = np.zeros((h, w), np.int32)
        for label in range(1, int(rng.integers(1, 2 * k + 2)) + 1):
            cy, cx = rng.uniform((0, 0), (h, w))
            r = rng.uniform(1.5, 6)
            nuclei[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
            if rng.random() < 0.1:
                cy, cx = rng.uniform((0, 0), (h, w))
                nuclei[(yy - cy) ** 2 + (xx - cx) ** 2 <= 4] = label
        out.append((nuclei, cells))
    return out


CASES = _handmade() + _random_cases()


@pytest.mark.parametrize("index", range(len(CASES)))
def test_merge_default_is_byte_identical_to_the_output_before_the_switch(index):
    nuclei, cells = CASES[index]
    want = _pair_before_switch(nuclei, cells)
    for got in (ops_sample.pair_nuclei_with_cells(nuclei, cells),
                ops_sample._pair_nuclei_with_cells(nuclei, cells),
                ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="merge"),
                ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning=None)):
        _same(got, want)
        assert got[2] == want[2]
    k_want = _kernel_before_switch(nuclei.copy(), cells.copy())
    k_got = M._assign_nuclei_to_cells(nuclei.copy(), cells.copy())
    assert [a.tobytes() for a in k_got[:2]] == [a.tobytes() for a in k_want[:2]]
    assert k_got[2:] == k_want[2:]


@pytest.mark.parametrize("index", range(len(CASES)))
def test_best_parent_is_byte_identical_to_the_frozen_a1_pairing(index):
    nuclei, cells = CASES[index]
    want = _pair_26bc722c15(nuclei, cells)
    got = ops_sample._pair_nuclei_with_cells(nuclei.copy(), cells.copy(), spanning="best_parent")
    _same(got, want)


def test_the_cases_really_span_and_the_two_arms_differ():
    """The fixtures exercise the merge: the arms differ on some of them."""
    differ = merged = 0
    for nuclei, cells in CASES:
        a = ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="merge")
        b = ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="best_parent")
        merged += a[2]["merged_cells"] > 0
        differ += not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        assert b[2]["merged_cells"] == 0
    assert merged >= 10 and differ >= 10


def test_best_parent_in_the_kernel_merges_nothing_and_gives_each_nucleus_its_best_cell():
    cells = np.zeros((20, 40), np.int32)
    cells[:, :20], cells[:, 20:] = 1, 2
    nuclei = np.zeros_like(cells)
    nuclei[8:12, 14:24] = 1                             # 6 px wide in 1, 4 in 2
    n, c, shared_n, shared_c, merged = M._assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), spanning="best_parent")
    assert merged == 0 and np.array_equal(c, cells)
    assert shared_n == {1} and shared_c == {1, 2}
    assert (n[:, 20:] == 0).all() and (n[8:12, 14:20] == 1).all()
    n, c, *_rest, merged = M._assign_nuclei_to_cells(nuclei.copy(), cells.copy())
    assert merged == 1 and np.unique(c[c > 0]).tolist() == [1]


def test_an_unknown_mode_is_refused():
    z = np.zeros((4, 4), np.int32)
    with pytest.raises(ValueError, match="spanning_merge"):
        M._assign_nuclei_to_cells(z, z.copy(), spanning="union")
    with pytest.raises(ValueError, match="spanning_merge"):
        ops_sample._pair_nuclei_with_cells(z, z.copy(), spanning="union")


def _a1_available():
    return all((A1_RAW / f"site-{s}.npz").is_file() for s in A1_SITES)


@pytest.mark.skipif(not _a1_available(), reason="the 4 A1 check fields are WS-local")
@pytest.mark.parametrize("site", A1_SITES)
def test_a1_check_fields_best_parent_equals_the_frozen_run(site):
    """LaC024A A1 check fields 357, 575, 713, 1078 (the raw nucleus and cell
    masks _ph_segment_field passes to the pairing): 'best_parent' gives the
    masks and counts of 26bc722c15's pairing, which are the nightly masks
    stored before 1612c53561; 'merge' gives the pairing before the switch."""
    raw = np.load(A1_RAW / f"site-{site}.npz")
    nuclei, cells = raw["nuclei"], raw["cells"]
    want = _pair_26bc722c15(nuclei, cells)
    got = ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="best_parent")
    _same(got, want)
    if (A1_OLD / f"site-{site}.npz").is_file():
        stored = np.load(A1_OLD / f"site-{site}.npz")
        assert np.array_equal(stored["nuclei"], got[0])
        assert np.array_equal(stored["cells"], got[1])
    merged = ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="merge")
    _same(merged, _pair_before_switch(nuclei, cells))


def _calls_with_keyword(path, function, callee, keyword):
    """The ``keyword=`` source of every call of ``callee`` in one top-level function."""
    tree = ast.parse((SPACR / path).read_text(encoding="utf-8"))
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == callee:
                found.extend(ast.unparse(k.value) for k in sub.keywords if k.arg == keyword)
    return found


def test_measure_and_ops_reach_the_same_kernel_with_their_own_setting():
    assert _calls_with_keyword("measure.py", "_measure_crop_core", "_assign_nuclei_to_cells",
                               "spanning") == ["settings.get('spanning_merge')"]
    assert _calls_with_keyword("ops_sample.py", "_pair_nuclei_with_cells",
                               "_assign_nuclei_to_cells", "spanning") == ["spanning"]
    assert _calls_with_keyword("ops_engine.py", "_ph_pair_raw", "_pair_nuclei_with_cells",
                               "spanning") == ["settings.get('ops_ph_spanning_merge')"]


def test_ops_pairing_calls_the_kernel_with_the_mode(monkeypatch):
    seen = []
    real = M._assign_nuclei_to_cells

    def spy(n, c, spanning="merge"):
        seen.append(spanning)
        return real(n, c, spanning=spanning)

    monkeypatch.setattr(M, "_assign_nuclei_to_cells", spy)
    nuclei, cells = CASES[0]
    ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="best_parent")
    ops_sample.pair_nuclei_with_cells(nuclei, cells)
    assert seen == ["best_parent", "merge"]


def test_settings_are_registered_with_merge_as_default():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS, OPS_TYPES
    from spacr.settings import (ALPHA_FEATURES, categories, expected_types,
                                get_measure_crop_settings, tooltips)

    assert OPS_DEFAULTS["ops_ph_spanning_merge"] == "merge"
    assert OPS_TYPES["ops_ph_spanning_merge"] is str
    assert "ops_ph_spanning_merge" in OPS_CATEGORIES["Phenotype segmentation"]
    assert all(len(keys) <= 20 for name, keys in OPS_CATEGORIES.items()
               if "ops_ph_spanning_merge" in keys)
    assert "ops_ph_spanning_merge" in ALPHA_FEATURES[752]["settings"]
    tip = OPS_TOOLTIPS["ops_ph_spanning_merge"]
    assert len(tip) <= 600 and tip.endswith("Default 'merge'. Alpha.")
    assert get_measure_crop_settings({})["spanning_merge"] == "merge"
    assert expected_types["spanning_merge"] is str
    assert "spanning_merge" in categories["Measurements"]
    assert len(tooltips["spanning_merge"]) <= 600
    assert tooltips["spanning_merge"].endswith("Default 'merge'.")


@pytest.mark.parametrize("version, expected", [
    (0, "best_parent"), (931, "best_parent"), (958, "best_parent"),
    (959, None), (965, None), (966, None), (967, None), (968, None),
])
def test_files_from_the_frozen_run_era_replay_best_parent(version, expected):
    """Schema 968: only files stamped below 959, where OPS pairing did not
    merge yet, get 'best_parent'; newer files without the key keep 'merge'."""
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                _fold_older_defaults)

    assert SETTINGS_SCHEMA_VERSION >= 968
    assert DEFAULT_CHANGES[968] == {"ops_ph_spanning_merge": "best_parent"}
    settings = {}
    applied = _fold_older_defaults(settings, version, known=set(OPS_DEFAULTS))
    assert settings.get("ops_ph_spanning_merge") == expected
    assert ("ops_ph_spanning_merge" in applied) == (expected is not None)
    if expected:
        assert applied["ops_ph_spanning_merge"] == ("best_parent", 968)
    explicit = {"ops_ph_spanning_merge": "merge"}
    _fold_older_defaults(explicit, version, known=set(OPS_DEFAULTS))
    assert explicit["ops_ph_spanning_merge"] == "merge"
    measure = {}
    _fold_older_defaults(measure, version, known={"spanning_merge"})
    assert "spanning_merge" not in measure


def test_an_old_ops_settings_file_resolves_to_best_parent(tmp_path):
    from spacr.cli import resolve_module, resolve_settings
    from spacr.settings import SETTINGS_SCHEMA_VERSION, SETTINGS_VERSION_KEY

    module = resolve_module("ops")
    for stamp, expected in ((958, "best_parent"), (959, "merge"),
                            (SETTINGS_SCHEMA_VERSION, "merge")):
        path = tmp_path / f"ops_{stamp}.json"
        path.write_text(json.dumps({SETTINGS_VERSION_KEY: stamp}), encoding="utf-8")
        assert resolve_settings(module, str(path))["ops_ph_spanning_merge"] == expected
