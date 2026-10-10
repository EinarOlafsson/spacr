"""Figure integrity (F572): the refusals of the traced-export helpers.

The traced paths -- source-crop windows, verified source regions, the
folder scan of earlier exports, merged-stack overlay replay and the index
lookup -- each abstain or raise on input they cannot vouch for. These tests
reach each refusal with the smallest input that triggers it.
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from PIL import Image  # noqa: E402

from spacr import plot  # noqa: E402
from spacr.run_journal import hash_file  # noqa: E402

HEX = "a" * 64


@pytest.fixture(autouse=True)
def _no_run(monkeypatch):
    monkeypatch.setattr("spacr.run_journal.current_run", lambda: None)
    monkeypatch.delenv(plot._INDEX_ENV, raising=False)


def _png_panel(path, box, number=0, digest=HEX, size=10, displayed="d"):
    return {"panel": number, "displayed_sha256": displayed,
            "source": [{"path": str(path), "sha256": digest, "bytes": size}],
            "steps": [{"op": "read_crop_png", "path": str(path), "format": 1},
                      {"op": "crop", "box": list(box)}]}


def test_source_crop_window_refuses_malformed_recipes(tmp_path, monkeypatch):
    png = str(tmp_path / "a.png")
    assert plot._source_crop_window(_png_panel(png, (0, 64, 0, 64))) == (
        {"path": png, "sha256": HEX, "bytes": 10}, (0, 64, 0, 64))
    assert plot._source_crop_window(
        _png_panel(png, (0, 64, 0, 64), digest="z" * 64)) is None
    moved = _png_panel(png, (0, 64, 0, 64))
    moved["steps"].reverse()
    assert plot._source_crop_window(moved) is None
    assert plot._source_crop_window(_png_panel(png, "box")) is None
    assert plot._source_crop_window(_png_panel(png, (0, 10, 0, 64))) is None
    npy = str(tmp_path / "m.npy")
    np.save(npy, np.zeros((200, 200, 4), np.uint16))
    merged = {"panel": 0, "displayed_sha256": "d",
              "source": [{"path": npy, "sha256": HEX, "bytes": 10}],
              "steps": [{"op": "merged_crop", "spec": "not a spec"}]}
    assert plot._source_crop_window(merged) is None
    spec = {"merged_path": npy, "object_type": "cell", "label": 1,
            "channels": [0], "size": [64, 64], "bbox": [10, 50, 10, 50]}
    merged["steps"] = [{"op": "merged_crop", "spec": dict(spec, unknown=1)}]
    assert plot._source_crop_window(merged) is None
    merged["steps"] = [{"op": "merged_crop", "spec": spec}]
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: np.asarray(real_load(
        *a, **k)))
    assert plot._source_crop_window(merged) is None


def _texture_png(path, rgb=False):
    rng = np.random.default_rng(3)
    shape = (96, 96, 3) if rgb else (96, 96)
    Image.fromarray(rng.integers(0, 255, shape).astype(np.uint8)).save(path)
    record = plot._source_record(str(path))
    return {"path": str(path), "sha256": record["sha256"],
            "bytes": record["bytes"]}


def _exact_record(path, number):
    image = np.array(Image.open(path))
    return {"panel": number, "displayed_sha256": plot._array_digest(image),
            "source": [plot._source_record(str(path))], "steps": []}


def test_verified_source_region_refusals(tmp_path, monkeypatch):
    source = _texture_png(tmp_path / "s.png")
    first, second = (source, (0, 80, 0, 80)), (source, (4, 84, 4, 84))
    panel = _exact_record(tmp_path / "s.png", 0)
    previous = _exact_record(tmp_path / "s.png", 1)
    later = time.monotonic() + 60
    assert plot._verified_source_region(panel, previous, first, second, later)
    assert not plot._verified_source_region(panel, previous, first, second, 0)
    assert not plot._verified_source_region(
        panel, previous, first, (source, (0, 80, 0, 200)), later)
    assert not plot._verified_source_region(
        panel, previous, (dict(source, sha256=HEX), (0, 80, 0, 80)), second, later)
    flat = tmp_path / "flat.png"
    Image.fromarray(np.zeros((96, 96), np.uint8)).save(flat)
    flat_source = plot._source_record(str(flat))
    assert not plot._verified_source_region(
        panel, previous, (flat_source, (0, 80, 0, 80)),
        (flat_source, (4, 84, 4, 84)), later)
    rgb = _texture_png(tmp_path / "rgb.png", rgb=True)
    assert not plot._verified_source_region(
        dict(panel, displayed_sha256="x"), previous,
        (rgb, (0, 80, 0, 80)), (rgb, (4, 84, 4, 84)), later)
    clock = itertools.chain([0.0] * 3, itertools.repeat(100.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    assert not plot._verified_source_region(panel, previous, first, second, 50.0)
    monkeypatch.undo()
    monkeypatch.setattr("spacr.run_journal.current_run", lambda: None)
    real = plot._source_record
    calls = {"n": 0}

    def drifting(path):
        calls["n"] += 1
        record = real(path)
        return dict(record, sha256=HEX) if calls["n"] > 2 else record

    monkeypatch.setattr(plot, "_source_record", drifting)
    assert not plot._verified_source_region(panel, previous, first, second, later)
    monkeypatch.setattr(plot, "_source_record", real)
    monkeypatch.setattr(plot, "_SOURCE_REGION_PIXELS", 10)
    assert not plot._verified_source_region(panel, previous, first, second, later)

    def bomb(*_args, **_kwargs):
        raise Image.DecompressionBombError("too big")

    monkeypatch.setattr(Image, "open", bomb)
    assert not plot._verified_source_region(panel, previous, first, second, later)


def _old_export(folder, name, panels, *, figure_bytes=b"figure"):
    figure = folder / name
    figure.write_bytes(figure_bytes)
    sidecar = folder / (name + plot._PROVENANCE_SUFFIX)
    sidecar.write_text(json.dumps({
        "schema": plot._PROVENANCE_SCHEMA, "panels": panels,
        "figure_sha256": hash_file(str(figure), full=True)}))
    os.utime(figure, (1, 1))
    return figure


def test_the_folder_scan_skips_unusable_panels(tmp_path):
    png = str(tmp_path / "a.png")
    current = [
        {"panel": 0, "displayed_sha256": "c0", "source": ["text"],
         "steps": [{"op": "crop"}]},
        {"panel": 1, "displayed_sha256": "c1",
         "source": [{"path": png, "sha256": HEX}],
         "steps": [{"op": "read_crop_png", "path": "/elsewhere.png"}]},
        _png_panel(png, (0, 80, 0, 80), number=2, displayed="c2"),
    ]
    old = [
        {"panel": 0, "displayed_sha256": "o0", "source": ["text"],
         "steps": [{"op": "crop"}]},
        {"panel": 1, "displayed_sha256": "o1",
         "source": [{"path": png, "sha256": HEX}],
         "steps": [{"op": "read_crop_png", "path": "/elsewhere.png"}]},
        "not a panel",
    ]
    _old_export(tmp_path, "old.png", old)
    assert plot._prior_figure_findings(current, str(tmp_path / "new.png")) == []


def _regions(findings):
    return [f for f in findings if f["check"] == "cross_figure_source_region"]


def test_the_folder_scan_skips_large_figures_and_its_spent_budget(
        tmp_path, monkeypatch):
    png = str(tmp_path / "a.png")
    current = [_png_panel(png, (0, 80, 0, 80), displayed="c")]
    old = [_png_panel(png, (4, 84, 4, 84), displayed="o")]
    big = _old_export(tmp_path, "big.png", old)
    os.truncate(big, 65 * 1024 ** 2)
    sidecar = tmp_path / ("big.png" + plot._PROVENANCE_SUFFIX)
    os.utime(sidecar, None)
    os.utime(big, (1, 1))
    assert not _regions(plot._prior_figure_findings(current,
                                                    str(tmp_path / "new.png")))
    os.remove(big)
    os.remove(sidecar)
    _old_export(tmp_path, "old.png", old)
    clock = itertools.chain([0.0] * 4, itertools.repeat(100.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    assert not _regions(plot._prior_figure_findings(current,
                                                    str(tmp_path / "new.png")))


def test_the_folder_scan_survives_a_failing_region_check(tmp_path, monkeypatch):
    png = str(tmp_path / "a.png")
    current = [_png_panel(png, (0, 80, 0, 80), displayed="c")]
    old = [_png_panel(png, (4, 84, 4, 84), displayed="o")]
    _old_export(tmp_path, "old.png", old)
    calls = []

    def failing(*args):
        calls.append(args)
        raise OSError("gone")

    monkeypatch.setattr(plot, "_verified_source_region", failing)
    assert not _regions(plot._prior_figure_findings(current,
                                                    str(tmp_path / "new.png")))
    assert len(calls) == 1


def test_index_lookup_skips_its_own_figure_and_irregular_sidecars(tmp_path):
    entry = {"v": 1, "figure": str(tmp_path / "f.png"),
             "figure_sha256": HEX, "sidecar": str(tmp_path),
             "panels": [{"panel": 0, "displayed_sha256": "same"}]}
    (tmp_path / "f.png").write_bytes(b"x")
    assert plot._indexed_prior_panel(entry, entry["panels"][0], {}) is None
    (tmp_path / plot._INDEX_NAME).write_text(json.dumps(entry) + "\n")
    panels = [{"panel": 0, "displayed_sha256": "same"}]
    assert plot._index_findings(panels, tmp_path / "f.png", []) == []


def _stack(dtype=np.uint16):
    image = np.zeros((64, 64, 4), dtype)
    image[..., 0] = np.arange(64, dtype=dtype)[None, :] * 3 + 5
    image[10:30, 10:30, 2] = 1
    image[40:60, 40:60, 2] = 2
    image[12:20, 12:20, 3] = 1
    return image


def _overlay_step(**changes):
    step = {"op": "overlay_composite", "channel": 0, "percentiles": [1, 99],
            "mode": "outlines", "thickness": 1, "all_on_all": False,
            "all_outlines": False, "outline_palette": "default",
            "mask_order": ["cell", "nucleus"],
            "mask_channels": {"cell": 0, "nucleus": 0},
            "mask_planes": {"cell": 2, "nucleus": 3}, "filters": None}
    step.update(changes)
    return step


@pytest.mark.parametrize("changes,image,message", [
    ({}, np.zeros((4, 4)), "numeric merged stack"),
    ({"mask_order": "cell"}, None, "invalid overlay mask recipe"),
    ({"mask_order": ["golgi"], "mask_planes": {"golgi": 2},
      "mask_channels": {}}, None, "unknown overlay mask role"),
    ({"filters": "x"}, None, "invalid overlay filter recipe"),
    ({"filters": {"pathogen": [[0, 1], [0, 1]]}}, None,
     "unknown overlay mask role"),
    ({"mask_planes": {"cell": 9, "nucleus": 3}}, None,
     "invalid overlay mask plane"),
    ({"filters": {"cell": [[0, 1]]}}, None,
     "invalid overlay filter or intensity channel"),
    ({"channel": 9}, None, "invalid overlay image channel"),
    ({"percentiles": [99, 1]}, None, "invalid overlay percentiles"),
])
def test_overlay_replay_refuses_invalid_recipes(changes, image, message):
    with pytest.raises(ValueError, match=message):
        plot._replay_overlay(_stack() if image is None else image,
                             _overlay_step(**changes))


def test_overlay_replay_budget_and_empty_combined_panel(monkeypatch):
    huge = np.lib.stride_tricks.as_strided(
        np.zeros(1, np.float64), shape=(20000, 20000, 3), strides=(0, 0, 0))
    with pytest.raises(ValueError, match="replay budget"):
        plot._replay_overlay(huge, _overlay_step(
            mask_order=["cell"], mask_planes={"cell": 2},
            mask_channels={"cell": 0}))
    with pytest.raises(ValueError, match="no masks"):
        plot._replay_overlay(_stack(), {"op": "combined_masks", "mask_order": [],
                                        "mask_planes": {}, "filters": {},
                                        "mask_channels": {}})


def test_overlay_replay_float_stacks_filters_and_slot_colours():
    from spacr.object_roles import ORGANELLE_ROLES

    stack = _stack(np.float32)
    rendered = plot._replay_overlay(stack, _overlay_step(
        filters={"cell": [[0, 50], [0, 1000]]}))
    assert rendered.shape == (64, 64, 3)
    slot = sorted(name for name in ORGANELLE_ROLES
                  if name.startswith("organelle") and name != "organelle")[0]
    filled = plot._replay_overlay(stack, _overlay_step(
        mode="filled", all_outlines=True, channel=1,
        mask_order=["cell", slot], mask_planes={"cell": 2, slot: 3},
        mask_channels={"cell": 0, slot: 0}))
    assert filled.shape == (64, 64, 3)
    empty = stack.copy()
    empty[..., 3] = 0
    combined = plot._replay_overlay(empty, {
        "op": "combined_masks", "mask_order": ["cell", "nucleus"],
        "mask_planes": {"cell": 2, "nucleus": 3}, "filters": {},
        "mask_channels": {}})
    assert combined.shape == (64, 64, 3)


def test_overlay_colour_table_bounds():
    with pytest.raises(ValueError, match="colour table"):
        plot._overlay_replay_colored(np.full((2, 2), 70000), 1)
    black = plot._overlay_replay_colored(np.full((2, 2), -1), 1)
    assert black.shape == (2, 2, 4) and not black[..., 3].any()


def _overlay_record(path, **changes):
    record = {"panel": 0, "displayed_sha256": "x",
              "source": [plot._source_record(str(path))],
              "steps": [_overlay_step()]}
    record.update(changes)
    return {"panels": [record]}


def test_overlay_reproduction_refusals(tmp_path, monkeypatch):
    png = tmp_path / "a.png"
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(png)
    with pytest.raises(ValueError, match="one merged NPY"):
        plot._reproduce_panel(_overlay_record(png), 0)
    npy = tmp_path / "m.npy"
    np.save(npy, _stack())
    sidecar = _overlay_record(npy)
    rebuilt, matches = plot._reproduce_panel(sidecar, 0)
    assert rebuilt.shape == (64, 64, 3) and not matches
    real_load = np.load

    class _Closable:
        closed = False

        def close(self):
            _Closable.closed = True

    monkeypatch.setattr(np, "load", lambda *a, **k: _Closable())
    with pytest.raises(ValueError, match="not a merged NPY"):
        plot._reproduce_panel(sidecar, 0)
    assert _Closable.closed
    monkeypatch.setattr(np, "load", lambda *a, **k: np.asarray(real_load(
        *a, **k)))
    with pytest.raises(ValueError, match="not a merged NPY"):
        plot._reproduce_panel(sidecar, 0)
    monkeypatch.setattr(np, "load", real_load)
    real = plot._source_record
    calls = {"n": 0}

    def drifting(path):
        calls["n"] += 1
        record = real(path)
        return dict(record, sha256=HEX) if calls["n"] > 1 else record

    monkeypatch.setattr(plot, "_source_record", drifting)
    with pytest.raises(ValueError, match="changed during replay"):
        plot._reproduce_panel(sidecar, 0)


def test_region_search_skips_panels_it_cannot_grey():
    panels = [{"panel": 0, "shape": [100, 100, 2], "source": []},
              {"panel": 1, "shape": [100, 100], "source": []}]
    arrays = [np.zeros((100, 100, 2)), np.zeros((100, 100))]
    findings, statistics = plot._region_reuse_findings(panels, arrays)
    assert findings == [] and statistics["pairs_checked"] == 0


def test_region_search_stops_at_its_panel_cap(monkeypatch):
    monkeypatch.setattr(plot, "_REGION_PANELS", 1)
    panels = [{"panel": n, "shape": [100, 100], "source": []} for n in range(3)]
    arrays = [np.random.default_rng(n).normal(size=(100, 100)) for n in range(3)]
    assert plot._region_reuse_findings(panels, arrays)[1]["pairs_checked"] == 0


def test_clone_search_abstains_below_its_peak_and_area_limits(monkeypatch):
    rng = np.random.default_rng(1)
    image = rng.normal(100, 5, (128, 128)).astype(np.float32)
    image[80:110, 70:100] = image[10:40, 10:40]
    assert plot._clone_region(image) is not None
    monkeypatch.setattr(plot, "_CLONE_AREA", 0.9)
    assert plot._clone_region(image) is None
    monkeypatch.setattr(plot, "_CLONE_PEAK", 2.0)
    assert plot._clone_region(image) is None


def test_matched_noise_needs_enough_pixels_per_brightness_bin():
    strip = np.random.default_rng(2).normal(0, 1, (10, 8)).astype(np.float32)
    assert plot._noise_by_intensity(strip, strip) is None


def test_work_coordinates_map_back_past_the_last_line():
    mapping = np.array([0, 2, 4])
    assert plot._work_to_displayed(mapping, 1.0, 3) == 5
    assert plot._work_to_displayed(mapping, 1.0, 1) == 2
    assert plot._work_to_displayed(mapping, 0.5, -2) == 0


def _signature(grey):
    import base64
    import hashlib
    import zlib

    raw = np.ascontiguousarray(grey, dtype=np.uint8).tobytes()
    return {"version": 1, "side": plot._SPATIAL_SIDE,
            "codec": "zlib+base64 gray-u8",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(zlib.compress(raw)).decode("ascii")}


def test_spatial_signature_refuses_unusable_panels():
    rng = np.random.default_rng(4)
    assert plot._spatial_signature(np.array([["a"] * 80] * 80)) is None
    assert plot._spatial_signature(np.zeros((40, 40))) is None
    assert plot._spatial_signature(np.zeros((80, 80, 2))) is None
    assert plot._spatial_signature(rng.normal(size=(64, 6000))) is None
    smooth = np.add.outer(np.arange(80.0), np.arange(80.0))
    assert plot._spatial_signature(smooth) is None
    assert plot._spatial_signature(
        rng.integers(0, 255, (80, 80, 3)).astype(np.uint8)) is not None


def test_spatial_signature_decoding_refuses_oversized_or_corrupt_data():
    import base64

    good = _signature(np.zeros((128, 128)))
    with pytest.raises(ValueError, match="encoded size"):
        plot._decode_spatial_signature(dict(good, data="A" * 40000))
    with pytest.raises(ValueError, match="malformed"):
        plot._decode_spatial_signature(dict(
            good, data=base64.b64encode(b"not zlib").decode("ascii")))


def test_similar_region_skips_flat_signatures_and_structure_mismatch():
    import cv2

    rng = np.random.default_rng(5)
    flat = _signature(np.full((128, 128), 100))
    texture = cv2.GaussianBlur(rng.normal(0, 1, (128, 128)), (0, 0), 1.0)
    textured = _signature(np.clip(128 + 60 * texture / texture.std(), 0, 255))
    assert plot._similar_displayed_region(flat, textured) is None
    ramp = np.tile(np.linspace(-90, 90, 128), (128, 1))
    detail = 12 * texture / texture.std()
    source = np.clip(128 + detail + ramp, 0, 255)
    side = 96
    region_detail = detail[16:16 + side, 16:16 + side]
    region_ramp = ramp[16:16 + side, 16:16 + side]
    candidate = cv2.resize(np.clip(128 + region_detail - region_ramp, 0, 255),
                           (128, 128), interpolation=cv2.INTER_LINEAR)
    assert plot._similar_displayed_region(
        _signature(source), _signature(candidate)) is None


def test_attaching_signatures_survives_failures_and_keeps_sidecars_small(
        monkeypatch):
    rng = np.random.default_rng(6)
    arrays = [rng.integers(0, 255, (128, 128)).astype(np.uint8)
              for _ in range(3)]

    def broken(_array):
        raise ValueError("bad panel")

    report = {"panels": [{"panel": n} for n in range(3)]}
    real = plot._spatial_signature
    monkeypatch.setattr(plot, "_spatial_signature", broken)
    plot._attach_spatial_signatures(report, arrays)
    assert not any("spatial_v1" in panel for panel in report["panels"])
    monkeypatch.setattr(plot, "_spatial_signature", real)
    monkeypatch.setattr(plot, "_SPATIAL_SIDECAR_LIMIT", 30000)
    monkeypatch.setattr(plot, "_SPATIAL_SIDECAR_MARGIN", 0)
    plot._attach_spatial_signatures(report, arrays)
    assert 0 < sum("spatial_v1" in panel for panel in report["panels"]) < 3
    report["padding"] = "x" * 40000
    plot._attach_spatial_signatures(report, ())
    assert not any("spatial_v1" in panel for panel in report["panels"])


def test_a_panel_record_without_a_cache_reads_its_source(tmp_path):
    from matplotlib.figure import Figure

    source = tmp_path / "s.png"
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(source)
    fig = Figure()
    axes = fig.add_subplot(111)
    artist = axes.imshow(np.zeros((20, 20)))
    plot._tag_panel(artist, source=str(source))
    record, _data = plot._panel_record(0, 0, axes, artist, fig, 100)
    assert record["source"][0]["sha256"] == plot._source_record(
        str(source))["sha256"]
