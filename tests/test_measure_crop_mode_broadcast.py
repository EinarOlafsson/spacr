"""Per-crop-mode settings must broadcast the way ``png_size`` always did.

``crop_mode`` is a list and ``dialate_pngs`` / ``dialate_png_ratios`` /
``png_size`` are indexed by position in it. Only ``png_size`` ever had the
``* len(crop_ls)`` broadcast; the other two were hard-broadcast to length 3
when scalar and used as given when a list, so the shipped default
``dialate_png_ratios=[0.2]`` raised ``IndexError: list index out of range``
on the second crop mode of every field. ``_measure_crop_core`` catches that
per field, so the symptom was not a traceback but a run that wrote the first
mode's crops, wrote no others, and reported its fields as failed.

Everything here drives the real ``_measure_crop_core`` -- real
``_merge_and_save_to_database`` / ``filepaths_to_database`` writers, real
``cv2.imwrite`` -- on a synthetic merged stack. CPU only, no Cellpose.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from spacr.settings import get_measure_crop_settings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_merged_stack(masks, rng, n_channels=4, with_organelle=False):
    """Merged (H, W, C) stack: intensity channels then mask slices at the
    dims measure_crop expects (cell=4, nucleus=5, pathogen=6, organelle=7)."""
    cell = masks["cell"].astype(np.uint16)
    nucleus = masks["nucleus"].astype(np.uint16)
    pathogen = masks["pathogen"].astype(np.uint16)
    H, W = cell.shape
    chans = []
    for _ in range(n_channels):
        base = rng.integers(50, 200, size=(H, W)).astype(np.uint16)
        base[cell > 0] += 3000
        chans.append(base)
    layers = chans + [cell, nucleus, pathogen]
    if with_organelle:
        organelle = np.zeros_like(cell)
        organelle[nucleus > 0] = nucleus[nucleus > 0]
        layers.append(organelle.astype(np.uint16))
    return np.stack(layers, axis=-1).astype(np.uint16)


def _write_stack(tmp_path, data, name="plate1_A01_F001.npy"):
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    # The pipeline creates measurements/ upstream.
    (tmp_path / "measurements").mkdir(parents=True, exist_ok=True)
    np.save(merged / name, data)
    return merged, name


def _settings_for(merged_dir, **over):
    s = get_measure_crop_settings(settings={})
    s.update({
        "src": str(merged_dir),
        "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [64, 64],
        "save_measurements": True, "save_png": True, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": True,
    })
    s.update(over)
    return s


def _pngs_by_folder(root):
    """{crop-mode folder: sorted png paths} for every PNG written under ``root``.

    _generate_names buckets crops by nucleus/pathogen count, so the same
    ``<mode>_png`` folder name appears under several parents; collect them
    all rather than keeping the last one seen.
    """
    out = {}
    for dirpath, _dirs, files in os.walk(str(root)):
        pngs = sorted(f for f in files if f.endswith(".png"))
        if pngs:
            out.setdefault(os.path.basename(dirpath), []).extend(
                os.path.join(dirpath, f) for f in pngs)
    return {k: sorted(v) for k, v in out.items()}


def _run(tmp_path, masks, rng, data=None, **over):
    """Run one field and return (succeeded, {folder: png paths})."""
    from spacr.measure import _measure_crop_core
    if data is None:
        data = _build_merged_stack(
            masks, rng, with_organelle=over.pop("_organelle", False))
    else:
        over.pop("_organelle", None)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, **over)
    _idx, _avg, cells, _figs = _measure_crop_core(0, [], name, settings)
    # _measure_crop_core's cross-process failure sentinel is the plain int 0;
    # the success path always assigns np.unique(...), an ndarray.
    succeeded = isinstance(cells, np.ndarray)
    return succeeded, _pngs_by_folder(tmp_path)


# ---------------------------------------------------------------------------
# _per_crop_mode -- the broadcast itself
# ---------------------------------------------------------------------------

def test_per_crop_mode_broadcasts_a_scalar():
    from spacr.measure import _per_crop_mode
    assert _per_crop_mode(0.2, 3, "dialate_png_ratios") == [0.2, 0.2, 0.2]
    assert _per_crop_mode(True, 2, "dialate_pngs") == [True, True]
    # An int scalar used to match no branch at all (the check was
    # `isinstance(..., float)`), leaving the name unbound.
    assert _per_crop_mode(0, 2, "dialate_png_ratios") == [0, 0]


def test_per_crop_mode_broadcasts_a_one_element_list():
    from spacr.measure import _per_crop_mode
    # This is the shipped default: dialate_png_ratios == [0.2].
    assert _per_crop_mode([0.2], 4, "dialate_png_ratios") == [0.2] * 4
    assert _per_crop_mode((0.2,), 2, "dialate_png_ratios") == [0.2, 0.2]


def test_per_crop_mode_passes_a_matching_list_through():
    from spacr.measure import _per_crop_mode
    assert _per_crop_mode([0.1, 0.2, 0.3], 3, "x") == [0.1, 0.2, 0.3]


def test_per_crop_mode_pads_a_short_list_and_says_so(capsys):
    from spacr.measure import _per_crop_mode
    assert _per_crop_mode([0.1, 0.2], 4, "dialate_png_ratios") == [0.1, 0.2, 0.2, 0.2]
    out = capsys.readouterr().out
    assert "dialate_png_ratios" in out
    assert "2 entries" in out and "4" in out


def test_per_crop_mode_truncates_a_long_list_and_says_so(capsys):
    from spacr.measure import _per_crop_mode
    assert _per_crop_mode([0.1, 0.2, 0.3], 2, "dialate_png_ratios") == [0.1, 0.2]
    assert "ignoring the extra" in capsys.readouterr().out


def test_per_crop_mode_rejects_an_empty_list():
    from spacr.measure import _per_crop_mode
    with pytest.raises(ValueError, match="dialate_pngs"):
        _per_crop_mode([], 2, "dialate_pngs")


def test_per_crop_mode_returns_nothing_when_there_are_no_crop_modes():
    """crop_mode=[] crops nothing, so an empty per-mode list is fine."""
    from spacr.measure import _per_crop_mode
    assert _per_crop_mode([], 0, "dialate_pngs") == []
    assert _per_crop_mode(0.2, 0, "dialate_png_ratios") == []


def test_empty_crop_mode_writes_measurements_and_no_crops(tmp_path,
                                                          synth_masks_multi, rng):
    ok, folders = _run(tmp_path, synth_masks_multi, rng, crop_mode=[],
                       dialate_png_ratios=[], dialate_pngs=[])
    assert ok
    assert folders == {}
    assert (tmp_path / "measurements" / "measurements.db").is_file()


# ---------------------------------------------------------------------------
# the reported defect, end to end
# ---------------------------------------------------------------------------

def test_two_crop_modes_with_the_shipped_default_ratio(tmp_path, synth_masks_multi, rng):
    """crop_mode=['cell','nucleus'] + the shipped dialate_png_ratios=[0.2].

    Before: IndexError on crop_idx 1, the field filed as FAILED, and only
    cell_png written.
    """
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       crop_mode=["cell", "nucleus"])
    assert ok, "the field must not be filed as failed"
    assert set(folders) == {"cell_png", "nucleus_png"}
    assert len(folders["cell_png"]) == len(folders["nucleus_png"]) > 0


def test_two_crop_modes_with_dilation_on_and_the_default_ratio(tmp_path,
                                                              synth_masks_multi, rng):
    """Same, with dialate_pngs=True so the ratio is actually read."""
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       crop_mode=["cell", "nucleus"], dialate_pngs=True)
    assert ok
    assert set(folders) == {"cell_png", "nucleus_png"}


def test_four_crop_modes_with_scalar_settings(tmp_path, synth_masks_multi, rng):
    """A scalar was hard-broadcast to LENGTH 3 -- a fourth mode raised.

    'cytoplasm' sits at index 3 here, so it is the entry the old
    ``[v, v, v]`` broadcast could not reach.
    """
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       crop_mode=["cell", "nucleus", "pathogen", "cytoplasm"],
                       dialate_pngs=True, dialate_png_ratios=0.2)
    assert ok
    assert set(folders) >= {"cell_png", "nucleus_png", "cytoplasm_png"}


def test_int_scalar_ratio_no_longer_leaves_the_name_unbound(tmp_path,
                                                            synth_masks_multi, rng):
    """dialate_png_ratios=0 matched neither `float` nor `list`.

    The name stayed unbound and the whole field died with UnboundLocalError
    before a single crop was written.
    """
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       dialate_pngs=True, dialate_png_ratios=0)
    assert ok
    assert folders["cell_png"]


def test_broadcast_ratio_equals_an_explicit_per_mode_list(tmp_path,
                                                          synth_masks_multi, rng):
    """A broadcast [0.5] must produce exactly what [0.5, 0.5] produces.

    Not merely "does not crash": the crops must be byte-identical, which is
    what makes it a broadcast rather than a silently different setting.
    """
    a = tmp_path / "broadcast"
    b = tmp_path / "explicit"
    # The SAME pixels both times -- rng is a live generator, so building the
    # stack twice would compare two different fields.
    data = _build_merged_stack(synth_masks_multi, rng)
    ok_a, fa = _run(a, synth_masks_multi, rng, data=data,
                    crop_mode=["cell", "nucleus"],
                    dialate_pngs=True, dialate_png_ratios=[0.5])
    ok_b, fb = _run(b, synth_masks_multi, rng, data=data,
                    crop_mode=["cell", "nucleus"],
                    dialate_pngs=[True, True], dialate_png_ratios=[0.5, 0.5])
    assert ok_a and ok_b
    assert sorted(fa) == sorted(fb)
    for folder in fa:
        assert len(fa[folder]) == len(fb[folder])
        for pa, pb in zip(fa[folder], fb[folder]):
            assert os.path.basename(pa) == os.path.basename(pb)
            with open(pa, "rb") as f1, open(pb, "rb") as f2:
                assert f1.read() == f2.read(), f"{folder}/{os.path.basename(pa)}"


def test_png_size_broadcasts_to_every_crop_mode(tmp_path, synth_masks_multi, rng):
    """A single [w, h] applies to every mode, and the PNGs really are that size."""
    import cv2
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       crop_mode=["cell", "nucleus", "pathogen"],
                       png_size=[48, 48])
    assert ok
    assert set(folders) >= {"cell_png", "nucleus_png"}
    for paths in folders.values():
        for p in paths:
            assert cv2.imread(p, cv2.IMREAD_UNCHANGED).shape[:2] == (48, 48)


def test_png_size_per_mode_is_still_honoured(tmp_path, synth_masks_multi, rng):
    import cv2
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       crop_mode=["cell", "nucleus"],
                       png_size=[[48, 48], [32, 32]])
    assert ok
    for p in folders["cell_png"]:
        assert cv2.imread(p, cv2.IMREAD_UNCHANGED).shape[:2] == (48, 48)
    for p in folders["nucleus_png"]:
        assert cv2.imread(p, cv2.IMREAD_UNCHANGED).shape[:2] == (32, 32)


# ---------------------------------------------------------------------------
# crop_mode as a bare string
# ---------------------------------------------------------------------------

def test_string_crop_mode_writes_crops(tmp_path, synth_masks_multi, rng):
    """crop_mode='cell' used to write measurements and zero PNGs, silently.

    The str branch built a local and the very next line re-tested
    settings['crop_mode'] for list-ness, so the whole crop block was skipped
    and nothing raised.
    """
    ok, folders = _run(tmp_path, synth_masks_multi, rng, crop_mode="cell")
    assert ok
    assert folders.get("cell_png"), "a string crop_mode must still crop"


def test_string_crop_mode_matches_the_one_element_list(tmp_path, synth_masks_multi, rng):
    a = tmp_path / "as_str"
    b = tmp_path / "as_list"
    data = _build_merged_stack(synth_masks_multi, rng)
    ok_a, fa = _run(a, synth_masks_multi, rng, data=data, crop_mode="nucleus")
    ok_b, fb = _run(b, synth_masks_multi, rng, data=data, crop_mode=["nucleus"])
    assert ok_a and ok_b
    assert sorted(fa) == sorted(fb) == ["nucleus_png"]
    assert len(fa["nucleus_png"]) == len(fb["nucleus_png"]) > 0


# ---------------------------------------------------------------------------
# an unrecognised crop mode
# ---------------------------------------------------------------------------

def test_unknown_crop_mode_is_skipped_not_carried_over(tmp_path, synth_masks_multi,
                                                       rng, capsys):
    """An unknown mode used to fall through with the PREVIOUS mode's mask.

    crop_mode=['cell','banana'] re-cropped the cell mask under the name
    'banana' -- and then died in cv2.imwrite, taking the field with it.
    """
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       crop_mode=["cell", "banana"])
    assert ok, "one bad entry must not fail the whole field"
    assert set(folders) == {"cell_png"}
    out = capsys.readouterr().out
    assert "banana" in out and "skipping" in out


def test_only_unknown_crop_modes_still_writes_measurements(tmp_path,
                                                           synth_masks_multi, rng):
    ok, folders = _run(tmp_path, synth_masks_multi, rng, crop_mode=["banana"])
    assert ok
    assert folders == {}
    db = tmp_path / "measurements" / "measurements.db"
    assert db.is_file()
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert any("cell" in t for t in tables)


# ---------------------------------------------------------------------------
# the database the crops are registered in
# ---------------------------------------------------------------------------

def test_png_list_records_every_crop_mode(tmp_path, synth_masks_multi, rng):
    """filepaths_to_database is the real writer; every mode must reach it."""
    ok, folders = _run(tmp_path, synth_masks_multi, rng,
                       crop_mode=["cell", "nucleus"])
    assert ok
    db = tmp_path / "measurements" / "measurements.db"
    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT png_path FROM png_list").fetchall()
    finally:
        con.close()
    paths = [r[0] for r in rows]
    assert any("cell_png" in p for p in paths)
    assert any("nucleus_png" in p for p in paths)
    assert len(paths) == sum(len(v) for v in folders.values())
