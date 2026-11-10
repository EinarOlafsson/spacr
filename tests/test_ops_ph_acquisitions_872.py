"""Phenotype acquisitions split across folders (item 872, B3).

A synthetic plate holds two phenotype acquisitions side by side, as Funk
plate B does: the first images DAPI, GFP and A594, the second A594 again
and AF750, offset by a few pixels. Indexed as one folder, the two A594
files of a site collide; that used to keep whichever file the walk reached
last and is now refused. ``ops_ph_acquisitions`` names each folder's
channels: the index then takes A594 from the first acquisition, keeps the
second's A594 only to register on, and ignores a third folder that is not
listed (Funk plate D's single-file GFP re-image).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr import ops_engine                                         # noqa: E402

SHIFT = (3, -2)


def _disc_field(seed=0, size=64):
    """A field of bright discs on noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    plane = rng.normal(10, 1, (size, size))
    for cy, cx in ((14, 18), (20, 46), (44, 22), (46, 44), (32, 32)):
        plane[np.hypot(yy - cy, xx - cx) < 5] += 300
    return plane.astype(np.uint16)


def _plate(root, sites=(0, 1)):
    """Two acquisitions of one well and an unlisted GFP re-image."""
    first, second, again = (root / "phenotype", root / "phenotype_af750",
                            root / "phenotype_regfp")
    for folder in (first, second, again):
        folder.mkdir(parents=True)
    for site in sites:
        base = _disc_field(site)
        moved = ops_engine._ph_shift_plane(base, *SHIFT)
        for name in ("DAPI", "GFP", "A594"):
            tifffile.imwrite(first / f"20X_DAPI-GFP-A594_A1_{name}_Site-{site}.tif",
                             base)
        for name in ("A594", "AF750"):
            tifffile.imwrite(second / f"20X_A594-AF750_A1_{name}_Site-{site}.tif",
                             moved)
        tifffile.imwrite(again / f"20X_GFP_A1_GFP_Site-{site}.tif", base + 7)
    return root


ACQUISITIONS = {"phenotype": "DAPI,GFP,A594", "phenotype_af750": "AF750",
                "register_on": "A594"}


def test_a_repeated_channel_token_is_refused_not_overwritten(tmp_path):
    _plate(tmp_path)
    with pytest.raises(ValueError, match="carry channel A594 for well A1"):
        ops_engine._index_tiles(str(tmp_path), "uncycled")
    single = ops_engine._index_tiles(str(tmp_path / "phenotype"), "uncycled")
    assert sorted(single["A1"][1][0]) == ["A594", "DAPI", "GFP"]


def test_the_listed_acquisitions_index_the_plate(tmp_path):
    _plate(tmp_path)
    acquisitions = ops_engine._ph_acquisitions(
        {"ops_ph_acquisitions": json.dumps(ACQUISITIONS)})
    assert acquisitions == {"folders": [("phenotype", ["DAPI", "GFP", "A594"]),
                                        ("phenotype_af750", ["AF750"])],
                            "register_on": "A594"}
    index, ignored = ops_engine._ph_acquisition_index(str(tmp_path), acquisitions)
    files = index["A1"][1][0]
    assert sorted(files) == ["A594", "A594@1", "AF750", "DAPI", "GFP"]
    assert "/phenotype/" in files["A594"]
    assert "/phenotype_af750/" in files["A594@1"]
    assert "/phenotype_regfp/" not in "".join(files.values())
    assert ignored == {"phenotype": 0, "phenotype_af750": 0}
    sources = ops_engine._plane_sources(files)
    assert sources["A594"][0] == files["A594"]


def test_the_second_acquisition_registers_on_its_own_register_channel(tmp_path):
    _plate(tmp_path)
    index, _ignored = ops_engine._ph_acquisition_index(
        str(tmp_path), ops_engine._ph_acquisitions(
            {"ops_ph_acquisitions": ACQUISITIONS}))
    files = index["A1"][1][0]
    assert ops_engine._ph_secondary_pair(files, ["AF750"]) == ("A594", "A594@1")
    primary, second = ops_engine._ph_split_files(files, ["AF750"])
    assert sorted(primary) == ["A594", "DAPI", "GFP"]
    assert sorted(second) == ["A594@1", "AF750"]
    shift = ops_engine._ph_register_secondary(files, ["AF750"], 48, [])
    assert (shift["dy"], shift["dx"]) == (-SHIFT[0], -SHIFT[1])
    assert shift["pair"] == "A594/A594@1"
    stack = ops_engine._ph_read_field(
        files, ("DAPI", "A594", "AF750"), [],
        {"channels": ["AF750"], "dy": shift["dy"], "dx": shift["dx"]})
    inner = (slice(4, -4), slice(4, -4))
    assert np.array_equal(stack[..., 2][inner], stack[..., 0][inner])
    assert np.array_equal(stack[..., 1], stack[..., 0])


def test_an_acquisition_left_out_is_not_read(tmp_path):
    _plate(tmp_path)
    index, ignored = ops_engine._ph_acquisition_index(
        str(tmp_path), ops_engine._ph_acquisitions({"ops_ph_acquisitions": {
            str(tmp_path / "phenotype"): "DAPI,A594",
            "phenotype_regfp": ["GFP"]}}))
    files = index["A1"][1][1]
    assert "/phenotype_regfp/" in files["GFP"]
    assert sorted(files) == ["A594", "DAPI", "GFP"]
    assert ignored[str(tmp_path / "phenotype")] == 2


@pytest.mark.parametrize("value, message", [
    ("not json", "is not that"),
    ([1, 2], "must map each acquisition"),
    ({"a": "", "b": "DAPI"}, "names no channel for a"),
    ({"a": "DAPI", "b": "dapi"}, "takes channel DAPI from both a and b"),
    ({"register_on": "DAPI"}, "names no acquisition"),
])
def test_a_malformed_acquisition_setting_is_refused(value, message):
    with pytest.raises(ValueError, match=message):
        ops_engine._ph_acquisitions({"ops_ph_acquisitions": value})


def test_an_empty_acquisition_setting_means_one_acquisition():
    assert ops_engine._ph_acquisitions({}) is None
    assert ops_engine._ph_acquisitions({"ops_ph_acquisitions": ""}) is None


def test_missing_and_empty_acquisition_folders_are_refused(tmp_path):
    _plate(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="which is not a folder"):
        ops_engine._ph_acquisition_index(str(tmp_path), {
            "folders": [("nowhere", ["DAPI"])], "register_on": None})
    with pytest.raises(ValueError, match="holds no phenotype tiles"):
        ops_engine._ph_acquisition_index(str(tmp_path), {
            "folders": [("empty", ["DAPI"])], "register_on": None})


def test_two_listed_files_of_one_token_are_refused(tmp_path):
    _plate(tmp_path)
    copy = tmp_path / "phenotype_copy"
    copy.mkdir()
    tifffile.imwrite(copy / "20X_DAPI-GFP-A594_A1_GFP_Site-0.tif",
                     _disc_field(0))
    with pytest.raises(ValueError, match="two GFP files"):
        ops_engine._ph_acquisition_index(str(tmp_path), {
            "folders": [("phenotype", ["DAPI", "GFP"]),
                        ("phenotype_copy", ["GFP"])], "register_on": None})


def test_run_ops_takes_the_second_acquisitions_channels_from_the_setting(
        tmp_path, monkeypatch):
    _plate(tmp_path)
    seen = {}

    def secondary(db, plate, well, files, settings):
        """Record what the run would register."""
        seen["channels"] = settings.get("ops_ph_secondary_channels")
        seen["files"] = sorted(files[1][0])
        return {"registered": 0}

    monkeypatch.setattr(ops_engine, "_ph_secondary_shifts", secondary)
    monkeypatch.setattr(ops_engine, "_ph_fields",
                        lambda *a, **k: {"fields": 0})
    run = tmp_path / "run"
    run.mkdir()
    from tests.test_ops_phenotype_run_v2 import _placed

    _placed(str(run / "measurements.db"))
    out = ops_engine.run_ops({
        "plate": "P_1", "dst_root": str(run), "phenotype_source": str(tmp_path),
        "ops_ph_acquisitions": ACQUISITIONS,
        "ops_ph_secondary_registration": "phase_correlation"},
        phases=("ph_segment",))
    assert seen["channels"] == "AF750"
    assert seen["files"] == ["A594", "A594@1", "AF750", "DAPI", "GFP"]
    assert out["ph_acquisitions"]["register_on"] == "A594"
    assert out["wells"]["A1"]["ph_secondary"] == {"registered": 0}
