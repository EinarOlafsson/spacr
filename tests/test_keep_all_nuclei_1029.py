"""Items 1028/1029 (alpha, L7): keep every nucleus of a cell, and ONE small-fragment merge.

The paper compared full-resolution cpsam_v2 with the old 2x masks on plate
A A1 and found second nuclei dropped (+459), fragments under 300 px (+356)
and nuclei without a cell (+155). Two opt-in pairing options recover them,
both inside the one shared pairing ``spacr.measure._assign_nuclei_to_cells``
that Measure, OPS (``spacr.ops_sample._pair_nuclei_with_cells``) and the
large-site preset (``spacr.ops_large_sites._site_pairing``) call:

- ``extra_nuclei`` / ``ops_ph_extra_nuclei`` ``'keep_all'``: every nucleus
  of a cell is kept as its own object linked to the cell; the OPS
  phenotype table stays one row per physical cell, with ``nucleus_count``,
  ``nucleus_ids`` and ``nucleus_area_total``.
- ``merge_small_nuclei_px`` / ``ops_ph_merge_small_nuclei_px``: a nucleus
  under the threshold merges into the touching nucleus of its cell, by the
  same ``spacr.measure._merge_small_nucleus_pieces`` the large-site
  small-piece merge uses.

Both are off by default and replay from the stored raw masks of item N6.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine, ops_sample
from spacr import ops_large_sites as ols
from spacr.measure import (_assign_nuclei_to_cells, _merge_small_nucleus_pieces,
                           _nucleus_pairing_active, _nucleus_pairing_options)

#: sha256 of :func:`_default_digest_parts` computed on origin/nightly
#: 4a54ba5be2 (before these options existed), so the default is pinned to
#: the bytes nightly produced, not to itself.
NIGHTLY_DEFAULT_DIGEST = "14383020a059e35689276856e20289b18b70800583e0b9c365cba87944b874d6"


def _disc(shape, centre, radius):
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2


def _field():
    """Four cells.

    cell 1: two separate round nuclei, 1 (larger) and 2: binucleate;
    cell 2: nucleus 3 with a small fragment 4 (about 28 px) touching it;
    cell 3: nucleus 5, and fragment 6 inside cell 3 that touches only
            nucleus 7 of cell 4 (a piece must not cross cells);
    cell 4: nucleus 7;
    nucleus 8: outside every cell.
    """
    shape = (70, 260)
    cells = np.zeros(shape, np.int32)
    nuclei = np.zeros(shape, np.int32)
    cells[5:60, 5:60] = 1
    nuclei[_disc(shape, (32, 20), 9)] = 1
    nuclei[_disc(shape, (32, 44), 7)] = 2
    cells[5:60, 65:120] = 2
    nuclei[_disc(shape, (32, 90), 10)] = 3
    nuclei[_disc(shape, (32, 103), 3) & (nuclei == 0)] = 4
    cells[5:60, 125:170] = 3
    nuclei[_disc(shape, (32, 140), 8)] = 5
    cells[5:60, 170:225] = 4
    nuclei[_disc(shape, (32, 179), 9)] = 7
    nuclei[29:35, 166:170] = 6
    nuclei[_disc(shape, (32, 245), 6)] = 8
    dna = (nuclei > 0).astype(np.float32) * 100.0
    return nuclei, cells, dna


def _on(**changes):
    options = _nucleus_pairing_options()
    options.update(changes)
    return options


def _default_digest_parts():
    """The default outputs of every pairing path on :func:`_field`, as bytes."""
    from spacr.ops_settings import ops_defaults

    nuclei, cells, dna = _field()
    parts = []
    public = ops_sample.pair_nuclei_with_cells(nuclei, cells)
    parts += [public[0].tobytes(), public[1].tobytes(), repr(sorted(public[2].items()))]
    live = ops_engine._ph_pair_raw(nuclei, cells, ops_defaults(), dna=dna)
    parts += [live[0].tobytes(), live[1].tobytes(), repr(sorted(live[2].items()))]
    measured = _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                                       options=_nucleus_pairing_options(), dna=dna)
    parts += [measured[0].tobytes(), measured[1].tobytes(), repr(measured[2:])]
    site = ols._site_pairing(nuclei, cells,
                             ols._site_pairing_options(ols.large_site_settings({})), None)
    parts += [site[0].tobytes(), site[1].tobytes(), repr(sorted(site[2].items()))]
    return [p if isinstance(p, bytes) else p.encode() for p in parts]


def _digest():
    return hashlib.sha256(b"|".join(_default_digest_parts())).hexdigest()


def _same_partition(a, b, where):
    """Whether two label images group the pixels of ``where`` identically."""
    pairs = np.unique(np.stack([a[where], b[where]], 1), axis=0)
    return (len(np.unique(pairs[:, 0])) == len(pairs)
            == len(np.unique(pairs[:, 1])))


# -- defaults ---------------------------------------------------------------

def test_both_options_are_off_by_default_and_take_no_alpha_path():
    from spacr.ops_settings import OPS_DEFAULTS, ops_defaults
    from spacr.settings import ALPHA_FEATURES, get_measure_crop_settings

    options = _nucleus_pairing_options()
    assert options["extra_nuclei"] == "drop"
    assert options["merge_small_nuclei_px"] == 0.0
    assert not _nucleus_pairing_active(options)
    assert ops_engine._ph_pairing_options(ops_defaults()) is None
    assert ols._site_pairing_options(ols.large_site_settings({})) is None
    assert OPS_DEFAULTS["ops_ph_extra_nuclei"] == "drop"
    assert OPS_DEFAULTS["ops_ph_merge_small_nuclei_px"] == 0.0
    measure = get_measure_crop_settings({})
    assert measure["extra_nuclei"] == "drop"
    assert measure["merge_small_nuclei_px"] == 0.0
    assert set(ALPHA_FEATURES[1029]["settings"]) == {
        "extra_nuclei", "merge_small_nuclei_px",
        "ops_ph_extra_nuclei", "ops_ph_merge_small_nuclei_px"}
    with pytest.raises(ValueError, match="extra_nuclei"):
        _nucleus_pairing_options({"ops_ph_extra_nuclei": "both"}, prefix="ops_ph_")


def test_the_default_is_byte_identical_to_nightly():
    assert _digest() == NIGHTLY_DEFAULT_DIGEST


def test_the_frac_merge_of_the_large_site_preset_is_unchanged_without_the_new_threshold():
    nuclei, cells, _ = _field()
    for frac in (0.0, 0.25):
        old, info = _merge_small_nucleus_pieces(nuclei, frac)
        new, info_new = _merge_small_nucleus_pieces(nuclei, frac, min_area_px=0)
        assert old.tobytes() == new.tobytes() and info == info_new
        assert "min_area_px" not in info


# -- keep every nucleus -----------------------------------------------------

def test_a_binucleate_cell_keeps_both_nuclei_only_under_keep_all():
    nuclei, cells, _ = _field()
    first, second = (nuclei == 1), (nuclei == 2)
    n_def, c_def, counts_def, record_def = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _nucleus_pairing_options())
    assert record_def is None
    assert n_def[first].all() and not n_def[second].any()
    assert counts_def["extra_nuclei_in_cell"] >= 1

    n_keep, c_keep, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(extra_nuclei="keep_all"))
    home = int(c_keep[first][0])
    assert set(np.unique(n_keep[first])) == {home}
    key = int(np.unique(n_keep[second])[0])
    assert key > counts["paired"]
    assert (key, home) in record["second_nuclei"]
    assert (c_keep[second] == home).all()
    assert counts["binucleate_kept"] == len(record["second_nuclei"])
    assert c_keep.tobytes() == c_def.tobytes()
    assert counts["paired"] == counts_def["paired"]
    keep_paired = (n_keep > 0) & (n_keep <= counts["paired"])
    assert n_keep[keep_paired].tobytes() == n_def[keep_paired].tobytes()


def test_the_large_site_and_measure_paths_keep_the_same_nuclei():
    nuclei, cells, _ = _field()
    settings = ols.large_site_settings({"ops_ph_extra_nuclei": "keep_all"})
    site = ols._site_pairing(nuclei, cells, ols._site_pairing_options(settings), None)
    ops = ops_sample._pair_nuclei_with_cells(nuclei, cells, _on(extra_nuclei="keep_all"))
    assert site[0].tobytes() == ops[0].tobytes()
    assert site[1].tobytes() == ops[1].tobytes()

    record: dict = {}
    _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                            options=_on(extra_nuclei="keep_all"), record=record)
    assert record["nucleus_ids"][1] == [1, 2]
    assert record["nuclei_per_cell"][1] == 2


def test_the_ops_table_keeps_one_row_per_physical_cell(tmp_path):
    nuclei, cells, dna = _field()
    n_out, c_out, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(extra_nuclei="keep_all"))
    path = ops_engine._ph_fragments_path(str(tmp_path), 0)
    ops_engine._ph_write_fragments(path, "A1", 0, record)
    table = ops_sample.measure_phenotype_field(
        n_out, c_out, np.stack([dna, dna], axis=-1), ["dapi", "gfp"], texture=False)
    assert len(table) == counts["paired"] + len(record["second_nuclei"])

    flat = ops_engine._ph_pairing_columns(table.copy(), c_out, path)
    assert len(flat) == len(table)
    folded = ops_engine._ph_pairing_columns(table.copy(), c_out, path,
                                            one_row_per_cell=True)
    cell_keys = sorted(int(v) for v in np.unique(c_out[c_out > 0]))
    assert sorted(folded["label"].astype(int)) == cell_keys
    assert folded["label"].is_unique
    rows = folded.set_index("label")
    for key, home in record["second_nuclei"]:
        assert rows.loc[home, "nucleus_count"] == 2
        assert bool(rows.loc[home, "binucleate"])
        assert rows.loc[home, "nucleus_ids"] == f"{home};{key}"
        areas = table.set_index("label")["nucleus_area"]
        assert rows.loc[home, "nucleus_area_total"] == pytest.approx(
            areas[home] + areas[key])
        assert rows.loc[home, "nucleus_area"] == areas[home]
    single = [k for k in cell_keys if k not in dict(record["second_nuclei"]).values()]
    assert all(rows.loc[k, "nucleus_ids"] == str(k) for k in single)
    assert (rows.loc[single, "nucleus_count"] == 1).all()


def test_the_measure_cell_table_names_every_nucleus_of_a_cell():
    from spacr.measure import _flag_shared_nuclei

    nuclei, cells, _ = _field()
    record: dict = {}
    n_out, c_out, shared_n, shared_c, _m = _assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), options=_on(extra_nuclei="keep_all"),
        record=record)
    morphology = {"cell": pd.DataFrame({"label": [1, 2, 3, 4]}),
                  "nucleus": pd.DataFrame({"label": sorted(np.unique(n_out[n_out > 0]))})}
    _flag_shared_nuclei(morphology, {"nucleus": shared_n, "cell": shared_c,
                                     "pairing": record})
    cell = morphology["cell"].set_index("label")
    assert cell.loc[1, "nucleus_ids"] == "1;2"
    assert cell.loc[1, "nucleus_count"] == 2 and bool(cell.loc[1, "binucleate"])
    assert len(morphology["cell"]) == 4


# -- one small-fragment merge -------------------------------------------------

def test_a_fragment_below_the_threshold_merges_identically_in_ops_measure_and_large_site():
    nuclei, cells, dna = _field()
    fragment, partner = (nuclei == 4), (nuclei == 3)
    across = nuclei == 6
    assert 0 < fragment.sum() < 300 and across.sum() < 300

    measure_settings = {"merge_small_nuclei_px": 300}
    ops_settings = {"ops_ph_merge_small_nuclei_px": 300}
    record: dict = {}
    m_n, m_c, *_ = _assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), options=_nucleus_pairing_options(measure_settings),
        dna=dna, record=record)
    o_n, o_c, o_counts = ops_engine._ph_pair_raw(nuclei, cells, ops_settings, dna=dna)
    s_n, s_c, s_counts = ols._site_pairing(
        nuclei, cells, ols._site_pairing_options(ols.large_site_settings(ops_settings)), dna)

    for labels in (m_n, o_n, s_n):
        assert len(np.unique(labels[fragment | partner])) == 1
        assert labels[partner].all()
    assert len(np.unique(m_n[across | (nuclei == 7)])) == 2
    assert o_n.tobytes() == s_n.tobytes() and o_c.tobytes() == s_c.tobytes()
    assert o_counts == s_counts
    assert o_counts["small_fragments_merged"] == record["small_fragments"]["n_merged"] == 1
    assert _same_partition(m_n, o_n, o_n > 0)

    off_n, _c, off_counts = ops_engine._ph_pair_raw(nuclei, cells, {}, dna=dna)
    assert not off_n[fragment].any() or len(np.unique(off_n[fragment | partner])) == 2
    assert "small_fragments_merged" not in off_counts


def test_a_fragment_at_or_above_the_threshold_is_not_merged():
    nuclei, cells, _ = _field()
    size = int((nuclei == 4).sum())
    record: dict = {}
    out, *_ = _assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), options=_on(merge_small_nuclei_px=size),
        record=record)
    assert record["small_fragments"]["n_merged"] == 0
    assert len(np.unique(out[(nuclei == 3) | (nuclei == 4)])) == 2


def test_the_seeded_large_site_runs_the_same_merge_before_the_cells_exist():
    nuclei, _cells, _ = _field()
    labels, record = ols._pair_before_cells(nuclei, _on(merge_small_nuclei_px=300))
    direct, info = _merge_small_nucleus_pieces(nuclei, 0.0, min_area_px=300)
    assert labels.tobytes() == direct.tobytes()
    assert record["small_fragments"] == info
    assert len(np.unique(labels[(nuclei == 3) | (nuclei == 4)])) == 1


def test_one_implementation_of_the_small_merge():
    import inspect

    from spacr import measure

    source = inspect.getsource(measure._assign_nuclei_to_cells)
    assert "merge_small_nuclei_px" in source and "_merge_small_nucleus_pieces(" in source
    for module in (ops_sample, ops_engine, ols):
        assert "def _merge_small" not in inspect.getsource(module)
    assert "_pair_before_cells" in inspect.getsource(ols.merge_small_nuclei)


# -- replay from the stored raw masks (item N6) -----------------------------

def test_replay_from_stored_raw_masks_equals_live_pairing(tmp_path):
    nuclei, cells, dna = _field()
    path = str(tmp_path / "site-0.npz")
    np.savez_compressed(path, nuclei=np.zeros_like(nuclei, np.uint32),
                        cells=np.zeros_like(cells, np.uint32),
                        **ops_engine._ph_raw_arrays({"nuclei": nuclei, "cells": cells}))
    for settings in ({}, {"ops_ph_extra_nuclei": "keep_all"},
                     {"ops_ph_merge_small_nuclei_px": 300},
                     {"ops_ph_extra_nuclei": "keep_all",
                      "ops_ph_merge_small_nuclei_px": 300}):
        live = ops_engine._ph_pair_raw(nuclei, cells, settings,
                                       dna=dna if settings else None)
        again = ops_engine._ph_pair_stored(path, settings)
        assert live[0].tobytes() == again[0].tobytes()
        assert live[1].tobytes() == again[1].tobytes()
        assert live[2] == again[2]


def test_a_real_phenotype_run_replays_and_writes_one_row_per_cell(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import tests.test_ops_engine_phenotype_coverage as cover

    def fake(settings, gpu, key="cellpose_model"):
        return cover._ThresholdModel(grow=0 if "nucleus" in key else 5)

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)
    tiles = cover._write_fields(tmp_path / "pheno")
    index = ops_engine._index_tiles(str(tiles), "uncycled")
    root = tmp_path / "run"
    (root / "A1").mkdir(parents=True)
    db = str(root / "measurements.db")
    cover._placed(db)
    settings = dict(cover.SETTINGS, ops_store_raw_masks=True,
                    ops_ph_extra_nuclei="keep_all", ops_ph_merge_small_nuclei_px=300)
    ops_engine._ph_fields(db, cover.PLATE, "A1", index["A1"], settings,
                          False, str(root), segment=True, measure=True)
    masks = ops_engine._ph_mask_paths(str(root), cover.PLATE, "A1", settings)
    assert masks
    for _site, path in masks.items():
        stored = ops_engine._ph_read_masks(path)
        nuclei, cells, _counts = ops_engine._ph_pair_stored(path, settings)
        assert nuclei.astype(np.uint32).tobytes() == stored["nuclei"].tobytes()
        assert cells.astype(np.uint32).tobytes() == stored["cells"].tobytes()
    table = pd.read_parquet(os.path.join(str(root), "ops_phenotype_cells_A1.parquet"))
    assert "nucleus_ids" in table and "nucleus_count" in table
    assert not table.duplicated(["site", "label"]).any()
