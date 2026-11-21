"""Item 1007 lane B: the lossless OME-Zarr 0.5 to 0.6 converter, ``upgrade_ngff``.

T8: every chunk file keeps its md5, the metadata maps back to the original, and the
hard-link and copy modes give the same verdict. T9: what has no lossless 0.6 form is
refused before anything is written, the source is never changed, and a stopped run resumes.
T10: plate, well and labels groups convert and read.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from spacr import ome_zarr_convert as convert
from spacr.layers import Spacing
from spacr.ome_zarr import read_ome_zarr, read_ome_zarr_array, write_ome_zarr

SPACING = Spacing.from_map({'y': 0.65, 'x': 0.65}, units='um')


def _image(seed=0, shape=(2, 48, 40)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=shape).astype(np.uint16)


def _store(root, name="field", seed=0, **over):
    """A 0.5 CYX image with two levels and a physical spacing."""
    options = dict(spacing=SPACING, axes=("c", "y", "x"), levels=2, chunks=(1, 16, 16),
                   ngff_version="0.5", channel_names=["DAPI", "GFP"])
    options.update(over)
    path = root / name
    write_ome_zarr(path, _image(seed), **options)
    return path


def _snapshot(root):
    """Every file under ``root`` with its md5."""
    return {p.relative_to(root).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _ome(path):
    return json.loads((path / "zarr.json").read_text(encoding="utf-8"))["attributes"]["ome"]


@pytest.mark.parametrize("link", [True, False])
def test_the_upgrade_keeps_every_chunk_and_maps_the_metadata_back(tmp_path, link):
    src = _store(tmp_path)
    before = _snapshot(src)
    report = convert.upgrade_ngff(src, tmp_path / "out.zarr", to="0.6", link=link)
    dst = tmp_path / "out.zarr"
    assert report["verdict"] == "verified" and report["to"] == "0.6"
    assert report["arrays"] == 2 and report["groups"] == 1 and report["chunks"] > 0
    assert (report["linked"], report["copied"]) == ((report["chunks"], 0) if link
                                                    else (0, report["chunks"]))
    assert _snapshot(src) == before
    chunks = [k for k in before if not k.endswith("zarr.json")]
    for relative in chunks:
        assert hashlib.md5((dst / relative).read_bytes()).hexdigest() == before[relative]
        same = os.stat(src / relative).st_ino == os.stat(dst / relative).st_ino
        assert same is link
    ome = _ome(dst)
    assert ome["version"] == "0.6" and "version" not in ome["multiscales"][0]
    assert ome["multiscales"][0]["coordinateSystems"][0]["axes"] == _ome(src)["multiscales"][0]["axes"]
    metadata = read_ome_zarr(dst)
    assert metadata.ngff_version == "0.6"
    assert metadata.channel_names == ("DAPI", "GFP")
    assert np.array_equal(read_ome_zarr_array(dst), read_ome_zarr_array(src))
    back = convert.upgrade_ngff(dst, tmp_path / "back.zarr", to="0.5", link=link)
    assert back["verdict"] == "verified"
    assert json.loads((tmp_path / "back.zarr" / "zarr.json").read_text()) == json.loads(
        (src / "zarr.json").read_text())
    assert back["chunk_list_md5"] == report["chunk_list_md5"]
    assert json.loads((dst / "ngff_upgrade.json").read_text())["status"] == "done"


def test_the_link_mode_comes_from_the_settings_unless_link_is_given(tmp_path):
    src = _store(tmp_path)
    copied = convert.upgrade_ngff(src, tmp_path / "a.zarr", settings={"ome_zarr_upgrade_link": False})
    assert copied["copied"] == copied["chunks"] and copied["linked"] == 0
    linked = convert.upgrade_ngff(src, tmp_path / "b.zarr", settings={})
    assert linked["linked"] == linked["chunks"]
    forced = convert.upgrade_ngff(src, tmp_path / "c.zarr", link=True,
                                  settings={"ome_zarr_upgrade_link": False})
    assert forced["linked"] == forced["chunks"]


def test_link_and_copy_give_the_same_verdict_and_chunk_list(tmp_path):
    src = _store(tmp_path)
    linked = convert.upgrade_ngff(src, tmp_path / "linked.zarr", link=True)
    copied = convert.upgrade_ngff(src, tmp_path / "copied.zarr", link=False)
    keys = ("verdict", "chunk_list_md5", "arrays", "groups", "chunks")
    assert {k: linked[k] for k in keys} == {k: copied[k] for k in keys}


def _rewrite(path, change):
    document = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    change(document["attributes"]["ome"])
    (path / "zarr.json").write_text(json.dumps(document), encoding="utf-8")


def test_an_unmappable_attribute_is_refused_and_nothing_is_written(tmp_path):
    src = _store(tmp_path)
    _rewrite(src, lambda ome: ome["multiscales"][0].update(
        coordinateTransformations=[{"type": "scale", "scale": [1.0, 1.0, 1.0]}]))
    before = _snapshot(src)
    with pytest.raises(ValueError, match="coordinateTransformations"):
        convert.upgrade_ngff(src, tmp_path / "out.zarr")
    assert not (tmp_path / "out.zarr").exists()
    assert _snapshot(src) == before


def test_other_versions_v2_stores_and_bad_destinations_are_refused(tmp_path):
    src = _store(tmp_path)
    with pytest.raises(ValueError, match="0.5 to 0.6 only|converts 0.6 to 0.5 only"):
        convert.upgrade_ngff(src, tmp_path / "out.zarr", to="0.5")
    with pytest.raises(ValueError, match="'0.6' or '0.5'"):
        convert.upgrade_ngff(src, tmp_path / "out.zarr", to="0.4")
    old = _store(tmp_path, "old", ngff_version="0.4")
    with pytest.raises(ValueError, match="Zarr v3|Zarr v2"):
        convert.upgrade_ngff(old, tmp_path / "old_out.zarr")
    with pytest.raises(ValueError, match="outside the source"):
        convert.upgrade_ngff(src, src / "inner.zarr")
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not an earlier upgrade"):
        convert.upgrade_ngff(src, other)
    assert (other / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_an_inner_version_is_dropped_when_it_repeats_the_store_and_refused_otherwise(tmp_path):
    src = _store(tmp_path)
    _rewrite(src, lambda ome: ome["multiscales"][0].update(version="0.5"))
    report = convert.upgrade_ngff(src, tmp_path / "out.zarr")
    assert report["removed_version_keys"] == ["zarr.json#multiscales[0]"]
    assert "version" not in _ome(tmp_path / "out.zarr")["multiscales"][0]
    clash = _store(tmp_path, "clash")
    _rewrite(clash, lambda ome: ome["multiscales"][0].update(version="0.4"))
    with pytest.raises(ValueError, match="version 0.4 inside a 0.5 store"):
        convert.upgrade_ngff(clash, tmp_path / "clash_out.zarr")


def test_a_stopped_run_resumes_and_a_different_upgrade_is_refused(tmp_path, monkeypatch):
    src = _store(tmp_path)
    real = convert._write_group

    def _stop(path, document):
        raise KeyboardInterrupt

    monkeypatch.setattr(convert, "_write_group", _stop)
    with pytest.raises(KeyboardInterrupt):
        convert.upgrade_ngff(src, tmp_path / "out.zarr", link=False)
    assert json.loads((tmp_path / "out.zarr" / "ngff_upgrade.json").read_text())["status"] == "running"
    monkeypatch.setattr(convert, "_write_group", real)
    report = convert.upgrade_ngff(src, tmp_path / "out.zarr", link=False)
    assert report["verdict"] == "verified" and report["kept"] == report["chunks"]
    other = _store(tmp_path, "other", seed=1)
    with pytest.raises(ValueError, match="holds an upgrade of"):
        convert.upgrade_ngff(other, tmp_path / "out.zarr")


def test_plate_well_and_label_groups_convert_and_read(tmp_path):
    """T10: the version key moves to ome.version alone; the rest is unchanged."""
    plate = tmp_path / "plate.zarr"
    plate.mkdir()
    plate_ome = {"version": "0.5", "plate": {"columns": [{"name": "1"}], "rows": [{"name": "A"}],
                                             "wells": [{"path": "A/1", "rowIndex": 0,
                                                        "columnIndex": 0}],
                                             "field_count": 1}}
    (plate / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"ome": plate_ome}}), encoding="utf-8")
    well = plate / "A" / "1"
    (plate / "A").mkdir()
    (plate / "A" / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {}}), encoding="utf-8")
    well.mkdir()
    well_ome = {"version": "0.5", "well": {"images": [{"path": "0"}]}}
    (well / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"ome": well_ome}}), encoding="utf-8")
    field = _store(well, "0")
    labels = field / "labels"
    labels.mkdir()
    (labels / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group",
         "attributes": {"ome": {"version": "0.5", "labels": ["cells"]}}}), encoding="utf-8")
    cells = (_image(2, (48, 40)) // 1000).astype(np.uint16)
    write_ome_zarr(labels / "cells", cells, spacing=SPACING, levels=2, downsample="stride",
                   chunks=(16, 16), ngff_version="0.5")
    _rewrite(labels / "cells", lambda ome: ome.update({"image-label": {"source": {"image": "../../"}}}))
    before = _snapshot(plate)
    report = convert.upgrade_ngff(plate, tmp_path / "plate06.zarr")
    out = tmp_path / "plate06.zarr"
    assert report["verdict"] == "verified" and report["groups"] == 6
    assert _snapshot(plate) == before
    assert _ome(out) == {**plate_ome, "version": "0.6"}
    assert _ome(out / "A" / "1") == {**well_ome, "version": "0.6"}
    assert _ome(out / "A" / "1" / "0" / "labels") == {"version": "0.6", "labels": ["cells"]}
    label_ome = _ome(out / "A" / "1" / "0" / "labels" / "cells")
    assert label_ome["version"] == "0.6" and label_ome["image-label"] == {"source": {"image": "../../"}}
    assert read_ome_zarr(out / "A" / "1" / "0").ngff_version == "0.6"
    assert np.array_equal(read_ome_zarr_array(out / "A" / "1" / "0" / "labels" / "cells"), cells)
    assert json.loads((out / "A" / "zarr.json").read_text()) == json.loads(
        (plate / "A" / "zarr.json").read_text())


def test_the_link_setting_is_alpha_and_on_by_default():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    assert OPS_DEFAULTS["ome_zarr_upgrade_link"] is True
    assert "ome_zarr_upgrade_link" in ALPHA_FEATURES[1007]["settings"]
    assert len(OPS_TOOLTIPS["ome_zarr_upgrade_link"]) <= 600
    assert "Alpha" in OPS_TOOLTIPS["ome_zarr_upgrade_link"]
