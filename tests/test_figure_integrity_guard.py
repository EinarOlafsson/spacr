"""The figure integrity guard on the one figure writer.

Pinned here:

* off by default: a figure written without asking is written exactly as
  before, with no sidecar and no stamp;
* panels meant for comparison that use different display ranges raise a
  warning, and panels sharing one range do not;
* clipped highlights and detector saturation raise a warning;
* a repeated panel -- flipped, rotated or contrast-changed -- is flagged,
  and distinct panels are not;
* a lossy format asked for is reported;
* provenance is stamped into the PNG and PDF metadata and a sidecar, and
  the sidecar rebuilds every tile of an image grid bit for bit from its
  source files;
* warnings reach the open run journal.
"""
from __future__ import annotations

import hashlib
import json
import os

import cv2
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from spacr import plot  # noqa: E402


@pytest.fixture(autouse=True)
def _guard_follows_the_argument(monkeypatch):
    monkeypatch.delenv(plot._INTEGRITY_ENV, raising=False)
    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: ("png", 100))


def _field(seed, size=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.zeros((size, size))
    for _ in range(8):
        cy, cx = rng.uniform(0, size, 2)
        r = rng.uniform(4, 10)
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    img = img * 2000 + 300 + rng.normal(0, 30, img.shape)
    return np.clip(img, 0, 60000).astype(np.uint16)


def _figure(panels, ranges, cmap="gray"):
    fig = Figure(figsize=(2 * len(panels), 2), dpi=100)
    for i, (img, rg) in enumerate(zip(panels, ranges)):
        ax = fig.add_subplot(1, len(panels), i + 1)
        kwargs = {} if rg is None else {"vmin": rg[0], "vmax": rg[1]}
        ax.imshow(img, cmap=cmap, **kwargs)
    return fig


def _shared(panels):
    return (0.0, float(max(int(p.max()) for p in panels)))


def _warnings(report, check):
    return [f for f in report["integrity"]["findings"]
            if f["severity"] == "warning" and f["check"] == check]


def _report(fig, fmt="png", requested=None):
    return plot._integrity_report(fig, fmt=fmt, requested_fmt=requested,
                                  dpi=100)


def test_off_by_default_writes_no_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr("spacr.qt.preferences._get_figure_integrity",
                        lambda: False)
    fig = _figure([_field(1), _field(2)], [None, None])
    written = plot.save_figure(fig, tmp_path / "plain.png")
    assert os.path.isfile(written)
    assert not os.path.exists(plot._provenance_sidecar_path(written))
    from PIL import Image
    with Image.open(written) as image:
        assert plot._PNG_PROVENANCE_KEY not in image.info


def test_the_environment_switches_it_on(tmp_path, monkeypatch):
    monkeypatch.setenv(plot._INTEGRITY_ENV, "1")
    written = plot.save_figure(_figure([_field(1)], [(0, 3000)]),
                               tmp_path / "env.png")
    assert os.path.isfile(plot._provenance_sidecar_path(written))


def test_a_figure_without_image_panels_is_left_alone(tmp_path):
    fig = Figure()
    fig.add_subplot().plot([1, 2, 3])
    written = plot.save_figure(fig, tmp_path / "line.png", integrity=True)
    assert not os.path.exists(plot._provenance_sidecar_path(written))


def test_exact_panel_reuse_across_exports_is_recorded(tmp_path):
    first = _field(302)
    plot.save_figure(_figure([first], [(0, 3000)]),
                     tmp_path / "first.png", integrity=True)
    second = plot.save_figure(_figure([first], [(0, 3000)]),
                              tmp_path / "second.png", integrity=True)
    with open(plot._provenance_sidecar_path(second), encoding="utf-8") as handle:
        report = json.load(handle)
    repeats = _warnings(report, "cross_figure_duplicate")
    assert len(repeats) == 1
    assert repeats[0]["prior_figure"] == "first.png"
    assert repeats[0]["prior_panel"] == 0

    distinct = plot.save_figure(_figure([_field(303)], [(0, 3000)]),
                                tmp_path / "distinct.png", integrity=True)
    with open(plot._provenance_sidecar_path(distinct), encoding="utf-8") as handle:
        assert not _warnings(json.load(handle), "cross_figure_duplicate")


def test_same_source_crop_with_changed_display_is_a_bounded_note(tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    field = np.clip(_field(309) // 16, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([field, field // 2, field // 3], axis=-1)).save(source)
    step = {"op": "read_crop_png", "path": str(source), "format": 1}

    def export(name, recipe):
        data = plot._replay_steps(None, recipe)
        figure = _figure([data], [(0, 255)])
        plot._tag_panel(figure.axes[0].images[0], source=source,
                        steps=recipe)
        written = plot.save_figure(figure, tmp_path / name, integrity=True)
        with open(plot._provenance_sidecar_path(written), encoding="utf-8") as handle:
            return json.load(handle)

    original = export("first.png", [step])
    report = export("changed.png", [step, {"op": "select_channel", "index": 0}])
    assert plot._reproduce_panel(original, 0)[1]
    assert plot._reproduce_panel(report, 0)[1]
    found = [item for item in report["integrity"]["findings"]
             if item["check"] == "cross_figure_source_crop"]
    assert len(found) == 1
    assert found[0]["severity"] == "note"
    assert found[0]["prior_figure"] == "first.png"
    assert found[0]["identical"] is False
    assert not _warnings(report, "cross_figure_duplicate")

    different_recipe = dict(step, format=2)
    report = export("other-recipe.png", [different_recipe,
                                         {"op": "select_channel", "index": 0}])
    assert not any(item["check"] == "cross_figure_source_crop"
                   for item in report["integrity"]["findings"])

    with source.open("ab") as handle:
        handle.write(b"changed source bytes")
    report = export("other-source.png", [step, {"op": "select_channel", "index": 0}])
    assert not any(item["check"] == "cross_figure_source_crop"
                   for item in report["integrity"]["findings"])


def _source_crop_export(folder, source, name, box):
    steps = [{"op": "crop", "box": list(box)}]
    data = plot._replay_steps(plot._read_panel_source(str(source)), steps)
    figure = _figure([data], [None])
    plot._tag_panel(figure.axes[0].images[0], source=source, steps=steps)
    written = plot.save_figure(figure, folder / name, integrity=True)
    with open(plot._provenance_sidecar_path(written), encoding="utf-8") as handle:
        report = json.load(handle)
    assert plot._reproduce_panel(report, 0)[1]
    return report


def test_verified_raw_source_crops_note_a_seventy_five_percent_region(tmp_path):
    from PIL import Image

    source = tmp_path / "field.png"
    pixels = np.clip(_field(601, size=256) // 16, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(source)
    first = _source_crop_export(tmp_path, source, "first.png",
                                (0, 256, 0, 256))
    later = _source_crop_export(tmp_path, source, "later.png",
                                (32, 224, 32, 224))
    findings = [item for item in later["integrity"]["findings"]
                if item["check"] == "cross_figure_source_region"]
    assert len(findings) == 1
    assert findings[0]["severity"] == "note"
    assert findings[0]["source_boxes"] == [[32, 224, 32, 224],
                                            [0, 256, 0, 256]]
    assert findings[0]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "same verified source pixels" in findings[0]["message"]
    assert later["integrity"]["warnings"] == 0
    assert not any(item["check"] == "similar_displayed_region"
                   for item in later["integrity"]["findings"])
    assert first["panels"][0]["displayed_sha256"] != later["panels"][0]["displayed_sha256"]


def test_source_region_abstains_for_distinct_blank_and_changed_sources(tmp_path):
    from PIL import Image

    distinct = tmp_path / "distinct"
    distinct.mkdir()
    left, right = distinct / "left.png", distinct / "right.png"
    first = np.clip(_field(602, size=256) // 16, 0, 255).astype(np.uint8)
    second = np.clip(_field(603, size=256) // 16, 0, 255).astype(np.uint8)
    first[10:20, :] = 255
    second[10:20, :] = 255
    Image.fromarray(first).save(left)
    Image.fromarray(second).save(right)
    _source_crop_export(distinct, left, "first.png", (0, 256, 0, 256))
    report = _source_crop_export(distinct, right, "second.png", (32, 224, 32, 224))
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])

    blank = tmp_path / "blank"
    blank.mkdir()
    source = blank / "source.png"
    Image.fromarray(np.full((256, 256), 44, dtype=np.uint8)).save(source)
    _source_crop_export(blank, source, "first.png", (0, 256, 0, 256))
    report = _source_crop_export(blank, source, "second.png", (32, 224, 32, 224))
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])

    changed = tmp_path / "changed"
    changed.mkdir()
    source = changed / "source.png"
    Image.fromarray(first).save(source)
    _source_crop_export(changed, source, "first.png", (0, 256, 0, 256))
    Image.fromarray(second).save(source)
    report = _source_crop_export(changed, source, "second.png", (32, 224, 32, 224))
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])


def test_source_region_abstains_for_disjoint_or_annotated_same_source(tmp_path):
    from PIL import Image

    source = tmp_path / "field.png"
    Image.fromarray(np.clip(_field(605, size=256) // 16, 0, 255)
                    .astype(np.uint8)).save(source)
    separate = tmp_path / "separate"
    separate.mkdir()
    _source_crop_export(separate, source, "first.png", (0, 96, 0, 96))
    report = _source_crop_export(separate, source, "second.png",
                                 (160, 256, 160, 256))
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])

    marked = tmp_path / "marked"
    marked.mkdir()
    _source_crop_export(marked, source, "first.png", (0, 256, 0, 256))
    recipe = [{"op": "crop", "box": [32, 224, 32, 224]}]
    pixels = plot._replay_steps(plot._read_panel_source(str(source)), recipe).copy()
    pixels[10:20, :] = 255
    figure = _figure([pixels], [None])
    plot._tag_panel(figure.axes[0].images[0], source=source, steps=recipe)
    written = plot.save_figure(figure, marked / "annotation.png", integrity=True)
    with open(plot._provenance_sidecar_path(written), encoding="utf-8") as handle:
        report = json.load(handle)
    assert not plot._reproduce_panel(report, 0)[1]
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])


def test_source_region_abstains_on_invalid_recipe_or_candidate_cap(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.fromarray(np.clip(_field(604, size=256) // 16, 0, 255)
                    .astype(np.uint8)).save(source)
    first = _source_crop_export(tmp_path, source, "first.png", (0, 256, 0, 256))
    sidecar = plot._provenance_sidecar_path(tmp_path / "first.png")
    first["panels"][0]["steps"][0]["box"] = [20, 212, 20, 212]
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(first, handle)
    report = _source_crop_export(tmp_path, source, "tampered.png", (32, 224, 32, 224))
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])
    stale = tmp_path / "stale"
    stale.mkdir()
    prior = stale / "first.png"
    _source_crop_export(stale, source, "first.png", (0, 256, 0, 256))
    with prior.open("ab") as handle:
        handle.write(b"a later figure replacement")
    report = _source_crop_export(stale, source, "later.png", (32, 224, 32, 224))
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])
    monkeypatch.setattr(plot, "_SOURCE_REGION_PAIRS", 0)
    report = _source_crop_export(tmp_path, source, "capped.png", (40, 232, 40, 232))
    assert not any(item["check"] == "cross_figure_source_region"
                   for item in report["integrity"]["findings"])
    assert plot._source_crop_window({"source": [{"path": str(source),
                "sha256": "a" * 64, "bytes": plot._SOURCE_REGION_BYTES + 1}],
                "steps": [{"op": "crop", "box": [0, 256, 0, 256]}]}) is None
    assert plot._source_crop_window({"source": [{"path": str(source),
                "sha256": "a" * 64, "bytes": 1024}],
                "steps": [{"op": "merged_crop", "spec": {}},
                          {"op": "crop", "box": [0, 256, 0, 256]}]}) is None


def test_spatial_signature_is_opt_in_bounded_and_refuses_malformed_data(tmp_path):
    import base64
    import zlib
    from PIL import Image

    array = _field(310, size=128)
    signature = plot._spatial_signature(array)
    assert signature["version"] == 1
    assert plot._decode_spatial_signature(signature).shape == (128, 128)
    assert plot._spatial_signature(np.zeros_like(array)) is None
    nonfinite = array.astype(float)
    nonfinite[0, 0] = np.nan
    assert plot._spatial_signature(nonfinite) is None

    with pytest.raises(ValueError, match="digest"):
        plot._decode_spatial_signature(dict(signature, sha256="0" * 64))
    with pytest.raises(ValueError, match="malformed"):
        plot._decode_spatial_signature(dict(signature, data="bad!"))
    with pytest.raises(ValueError, match="raw size"):
        plot._decode_spatial_signature(dict(
            signature, data=base64.b64encode(zlib.compress(
                b"X" * (128 * 128 * 100))).decode("ascii")))
    with pytest.raises(ValueError, match="version"):
        plot._decode_spatial_signature(dict(signature, version=2))

    random = np.random.default_rng(311)
    arrays = [random.integers(0, 256, (128, 128), dtype=np.uint8)
              for _ in range(40)]
    report = {"panels": [{"panel": index} for index in range(40)]}
    plot._attach_spatial_signatures(report, arrays)
    signed = [p["spatial_v1"] for p in report["panels"]
              if "spatial_v1" in p]
    assert 0 < len(signed) <= plot._SPATIAL_ATTEMPTS
    assert sum(len(s["data"]) for s in signed) <= plot._SPATIAL_REPORT_LIMIT
    assert len(json.dumps(report, indent=2).encode()) <= (
        plot._SPATIAL_SIDECAR_LIMIT - plot._SPATIAL_SIDECAR_MARGIN)

    first = plot.save_figure(_figure([array], [(0, 3000)]),
                             tmp_path / "unchecked.png", integrity=False)
    second = plot.save_figure(_figure([array], [(0, 3000)]),
                              tmp_path / "checked.png", integrity=True)
    with Image.open(first) as plain, Image.open(second) as checked:
        assert np.array_equal(np.asarray(plain), np.asarray(checked))
    with open(plot._provenance_sidecar_path(second), encoding="utf-8") as handle:
        saved = json.load(handle)
    assert "spatial_v1" in saved["panels"][0]
    assert plot._decode_spatial_signature(saved["panels"][0]["spatial_v1"]).shape == (128, 128)
    assert not any(item["check"] == "possible_reused_region"
                   for item in saved["integrity"]["findings"])


def test_checked_exports_note_only_a_bounded_similar_displayed_region(
        tmp_path, monkeypatch):
    original = _field(572, size=128)
    plot.save_figure(_figure([original], [(0, 3000)]),
                     tmp_path / "original.png", integrity=True)
    cropped = cv2.resize(original[12:108, 16:112], (128, 128),
                         interpolation=cv2.INTER_LINEAR)
    later = plot.save_figure(_figure([cropped], [(0, 3000)]),
                             tmp_path / "crop.png", integrity=True)
    with open(plot._provenance_sidecar_path(later), encoding="utf-8") as handle:
        report = json.load(handle)
    matches = [item for item in report["integrity"]["findings"]
               if item["check"] == "similar_displayed_region"]
    assert len(matches) == 1
    assert matches[0]["severity"] == "note"
    assert matches[0]["prior_figure"] == "original.png"
    assert "not an acquisition or image manipulation verdict" in matches[0]["message"]
    assert report["integrity"]["warnings"] == 0

    other = plot.save_figure(_figure([_field(573, size=128)], [(0, 3000)]),
                             tmp_path / "different.png", integrity=True)
    with open(plot._provenance_sidecar_path(other), encoding="utf-8") as handle:
        assert not any(item["check"] == "similar_displayed_region"
                       for item in json.load(handle)["integrity"]["findings"])
    annotated = _field(574, size=128)
    cv2.rectangle(annotated, (8, 8), (36, 30), 3000, -1)
    annotated_path = plot.save_figure(_figure([annotated], [(0, 3000)]),
                                      tmp_path / "annotated-other.png",
                                      integrity=True)
    with open(plot._provenance_sidecar_path(annotated_path), encoding="utf-8") as handle:
        assert not any(item["check"] == "similar_displayed_region"
                       for item in json.load(handle)["integrity"]["findings"])
    complex_other = np.block([[_field(575, 64), _field(576, 64)],
                              [_field(577, 64), _field(578, 64)]])
    complex_path = plot.save_figure(_figure([complex_other], [(0, 3000)]),
                                    tmp_path / "complex-other.png", integrity=True)
    with open(plot._provenance_sidecar_path(complex_path), encoding="utf-8") as handle:
        assert not any(item["check"] == "similar_displayed_region"
                       for item in json.load(handle)["integrity"]["findings"])
    assert plot._spatial_signature(np.zeros((128, 128), np.uint8)) is None

    prior_sidecar = plot._provenance_sidecar_path(tmp_path / "original.png")
    with open(prior_sidecar, encoding="utf-8") as handle:
        damaged = json.load(handle)
    damaged["panels"][0]["spatial_v1"]["data"] = "bad!"
    with open(prior_sidecar, "w", encoding="utf-8") as handle:
        json.dump(damaged, handle)
    after_damage = plot.save_figure(_figure([cropped], [(0, 3000)]),
                                    tmp_path / "after-damage.png", integrity=True)
    with open(plot._provenance_sidecar_path(after_damage), encoding="utf-8") as handle:
        assert not any(item["prior_figure"] == "original.png"
                       for item in json.load(handle)["integrity"]["findings"]
                       if item["check"] == "similar_displayed_region")

    monkeypatch.setattr(plot, "_SPATIAL_PAIRS", 0)
    capped = plot.save_figure(_figure([cropped], [(0, 3000)]),
                              tmp_path / "capped.png", integrity=True)
    with open(plot._provenance_sidecar_path(capped), encoding="utf-8") as handle:
        assert not any(item["check"] == "similar_displayed_region"
                       for item in json.load(handle)["integrity"]["findings"])

def test_a_stale_prior_sidecar_cannot_flag_an_overwritten_figure(tmp_path):
    first = plot.save_figure(_figure([_field(304)], [(0, 3000)]),
                             tmp_path / "first.png", integrity=True)
    # An image changed after its sidecar was written is no longer evidence.
    with open(first, "ab") as handle:
        handle.write(b"newer content")
    second = plot.save_figure(_figure([_field(304)], [(0, 3000)]),
                              tmp_path / "second.png", integrity=True)
    with open(plot._provenance_sidecar_path(second), encoding="utf-8") as handle:
        assert not _warnings(json.load(handle), "cross_figure_duplicate")


@pytest.mark.parametrize("newest_first", [False, True])
def test_prior_panel_checks_use_only_recent_exports_in_any_directory_order(
        tmp_path, monkeypatch, newest_first):
    from contextlib import contextmanager
    from pathlib import Path

    original = _figure([_field(305)], [(0, 3000)])
    other = _figure([_field(306)], [(0, 3000)])
    templates = []
    for index, figure in enumerate((original, other)):
        written = Path(plot.save_figure(
            figure, tmp_path / f"template{index}.png", integrity=True))
        templates.append((written.read_bytes(), Path(
            plot._provenance_sidecar_path(written)).read_bytes()))

    history = tmp_path / "history"
    history.mkdir()
    for index in range(66):
        image, record = templates[0 if index < 2 else 1]
        destination = history / f"prior{index:02d}.png"
        destination.write_bytes(image)
        sidecar = Path(plot._provenance_sidecar_path(destination))
        sidecar.write_bytes(record)
        timestamp = 1_000_000_000 + index * 10
        os.utime(destination, ns=(timestamp, timestamp))
        os.utime(sidecar, ns=(timestamp + 1, timestamp + 1))

    scandir = os.scandir

    @contextmanager
    def ordered_entries(folder):
        with scandir(folder) as entries:
            yield iter(sorted(entries, key=lambda entry: entry.name,
                              reverse=newest_first))

    monkeypatch.setattr(os, "scandir", ordered_entries)
    panels = _report(original)["panels"]
    assert plot._prior_figure_findings(panels, history / "current.png") == []
    recent_panels = _report(other)["panels"]
    findings = plot._prior_figure_findings(recent_panels, history / "current.png")
    assert len(findings) == 1
    assert findings[0]["prior_figure"] == "prior65.png"
    assert findings[0]["check"] == "cross_figure_duplicate"


def test_mismatched_display_ranges_warn(capsys):
    panels = [_field(i) for i in range(3)]
    shared = _report(_figure(panels, [(300, 2500)] * 3))
    assert _warnings(shared, "display_range") == []
    mismatched = _report(_figure(panels, [(300, 2500), (300, 2500),
                                          (300, 1400)]))
    found = _warnings(mismatched, "display_range")
    assert len(found) == 1 and found[0]["panels"] == [0, 1, 2]
    autoscaled = _report(_figure(panels, [None, None, None]))
    assert _warnings(autoscaled, "display_range")


def test_panels_in_different_channels_are_not_compared():
    panels = [_field(1), _field(2)]
    fig = _figure(panels, [(300, 2500), (300, 900)])
    fig.axes[1].images[0].set_cmap("magma")
    assert _warnings(_report(fig), "display_range") == []


def test_clipped_and_saturated_pixels_warn():
    bright = np.clip(_field(3).astype(float) * 3, 0, 60000).astype(np.uint16)
    rg = _shared([_field(3), _field(4)])
    report = _report(_figure([_field(4), bright], [rg] * 2))
    flagged = _warnings(report, "saturation")
    assert [f["panels"] for f in flagged] == [[1]]

    sensor = _field(5)
    sensor.ravel()[::50] = np.iinfo(np.uint16).max
    report = _report(_figure([sensor], [(0, 65535)]))
    assert any("acquisition" in f["message"]
               for f in _warnings(report, "saturation"))


@pytest.mark.parametrize("transform", [
    lambda x: x.copy(),
    lambda x: x[:, ::-1].copy(),
    lambda x: np.rot90(x).copy(),
    lambda x: (x.astype(float) * 0.6 + 400).astype(np.uint16),
], ids=["exact", "flip", "rot90", "contrast"])
def test_a_repeated_panel_is_flagged(transform):
    panels = [_field(i) for i in range(4)]
    panels[3] = transform(panels[1])
    report = _report(_figure(panels, [(300, 2500)] * 4))
    pairs = [set(f["panels"]) for f in _warnings(report, "duplicate")]
    assert pairs == [{1, 3}]


def test_distinct_panels_are_not_flagged_as_repeats():
    panels = [_field(i) for i in range(6)]
    report = _report(_figure(panels, [_shared(panels)] * 6))
    assert report["integrity"]["findings"] == []


def _cell(seed, size=64):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] - size / 2
    ry, rx = rng.uniform(10, 16, 2)
    body = np.exp(-((xx / rx) ** 2 + (yy / ry) ** 2) ** 2)
    img = body * rng.uniform(800, 1500) + 200
    img = rng.poisson(img).astype(float) + rng.normal(0, 40, img.shape)
    return np.clip(img, 0, 60000).astype(np.uint16)


def test_similar_but_different_cells_are_not_repeats():
    panels = [_cell(i) for i in range(12)]
    fig = Figure(figsize=(12, 1), dpi=100)
    for i, img in enumerate(panels):
        fig.add_subplot(1, 12, i + 1).imshow(img, cmap="gray", vmin=0,
                                             vmax=60000)
    report = _report(fig)
    assert _warnings(report, "duplicate") == []
    panels[7] = panels[2][::-1].copy()
    fig.axes[7].images[0].set_data(panels[7])
    pairs = [set(f["panels"]) for f in _warnings(_report(fig), "duplicate")]
    assert pairs == [{2, 7}]


def test_a_lossy_format_is_reported():
    fig = _figure([_field(1)], [(300, 2500)])
    assert _warnings(_report(fig, fmt="jpg"), "lossy_format")
    notes = [f for f in _report(fig, fmt="png", requested="jpg")
             ["integrity"]["findings"] if f["check"] == "lossy_format"]
    assert notes and notes[0]["severity"] == "note"


def test_provenance_rebuilds_every_grid_tile(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    paths = []
    for i in range(5):
        tile = (rng.random((40, 48, 3)) * 50 * (i + 1)).astype(np.uint8)
        paths.append(str(tmp_path / f"tile_{i}.png"))
        Image.fromarray(tile).save(paths[-1])
    fig = plot.plot_image_grid(paths, (2, 98))
    written = plot.save_figure(fig, tmp_path / "grid.png", integrity=True)
    sidecar = plot._provenance_sidecar_path(written)
    with open(sidecar, encoding="utf-8") as handle:
        record = json.load(handle)

    assert record["schema"] == plot._PROVENANCE_SCHEMA
    assert record["software"]["spacr"]
    assert len(record["figure_sha256"]) == 64
    assert len(record["panels"]) == 5
    assert _warnings(record, "display_range")
    for panel, path in zip(record["panels"], paths):
        assert panel["source"][0]["path"] == os.path.abspath(path)
        assert len(panel["source"][0]["sha256"]) == 64
        assert panel["display_range_units"] == "source"
        assert panel["reproducible"] is True
        rebuilt, matches = plot._reproduce_panel(sidecar, panel["panel"])
        assert matches, panel["panel"]
        assert rebuilt.dtype == np.uint8

    with Image.open(written) as image:
        stamped = json.loads(image.info[plot._PNG_PROVENANCE_KEY])
    assert stamped["panels"][0]["steps"] == record["panels"][0]["steps"]


def test_a_changed_source_no_longer_matches(tmp_path):
    from PIL import Image

    path = tmp_path / "tile.png"
    Image.fromarray((np.arange(40 * 40).reshape(40, 40) % 251)
                    .astype(np.uint8)).save(path)
    fig = plot.plot_image_grid([str(path)], (2, 98))
    written = plot.save_figure(fig, tmp_path / "one.png", integrity=True)
    Image.fromarray(np.zeros((40, 40), np.uint8) + 7).save(path)
    _rebuilt, matches = plot._reproduce_panel(
        plot._provenance_sidecar_path(written), 0)
    assert matches is False


def test_pdf_metadata_carries_the_provenance(tmp_path):
    fig = _figure([_field(1)], [(300, 2500)])
    written = plot.save_figure(fig, tmp_path / "page.pdf", fmt="pdf",
                               integrity=True)
    assert written.endswith(".pdf")
    assert b"spaCR provenance" in open(written, "rb").read()
    assert os.path.isfile(plot._provenance_sidecar_path(written))


def test_warnings_reach_the_open_run(tmp_path, monkeypatch, capsys):
    from spacr import run_journal

    class _Run:
        dir = tmp_path / "run"
        app_key = "measure"

        def __init__(self):
            self.warnings, self.outputs = [], []

        def record_warning(self, text):
            self.warnings.append(text)

        def record_output(self, path, setting_key=""):
            self.outputs.append(str(path))

    run = _Run()
    monkeypatch.setattr(run_journal, "current_run", lambda: run)
    panels = [_field(1), _field(2)]
    written = plot.save_figure(_figure(panels, [None, None]),
                               tmp_path / "run.png", integrity=True)
    with open(plot._provenance_sidecar_path(written),
              encoding="utf-8") as handle:
        record = json.load(handle)
    assert record["run"]["app"] == "measure"
    assert record["run"]["manifest"].endswith("manifest.json")
    assert any("display ranges" in text for text in run.warnings)
    assert run.outputs == [plot._provenance_sidecar_path(written)]
    assert "Figure integrity" in capsys.readouterr().out


def test_overlay_provenance_names_immutable_source_and_actual_compositing_recipe(
        tmp_path, monkeypatch):
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    source = tmp_path / "field.npy"
    stack = np.zeros((36, 40, 3), dtype=np.uint16)
    stack[..., 0] = np.arange(40, dtype=np.uint16)[None, :] * 100
    stack[..., 1] = np.arange(36, dtype=np.uint16)[:, None] * 70
    stack[8:20, 10:25, 2] = 1
    np.save(source, stack)
    limits = {"cell": ((1, 1000), (0, 65000))}
    figure = plot.plot_image_mask_overlay(
        str(source), [0, 1], 0, None, None, save_pdf=False,
        filter_dict=limits, mode="outlines", figuresize=1)
    before = [plot._array_digest(artist.get_array())
              for axes in figure.axes for artist in axes.images]
    limits["cell"] = ((1000, 2000), (0, 65000))
    report = plot._integrity_report(figure, fmt="png", dpi=100)
    assert len(report["panels"]) == 3
    assert [panel["displayed_sha256"] for panel in report["panels"]] == before
    for panel in report["panels"]:
        assert panel["source"][0]["path"] == str(source)
        assert panel["source"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert panel["reproducible"] is True
        rebuilt, matches = plot._reproduce_panel(report, panel["panel"])
        assert matches, panel["steps"]
        assert plot._array_digest(rebuilt) == panel["displayed_sha256"]
        assert panel["steps"][0]["filters"]["cell"] == [[1.0, 1000.0], [0.0, 65000.0]]
    assert [panel["steps"][0]["channel"] for panel in report["panels"][:2]] == [0, 1]
    assert report["panels"][0]["display_range_units"] == "source"
    written = plot.save_figure(figure, tmp_path / "overlay.png", fmt="png",
                               dpi=100, integrity=True)
    with open(plot._provenance_sidecar_path(written), encoding="utf-8") as handle:
        saved = json.load(handle)
    assert [panel["displayed_sha256"] for panel in saved["panels"]] == before
    assert all(plot._reproduce_panel(saved, panel["panel"])[1]
               for panel in saved["panels"])
    np.testing.assert_array_equal(np.load(source), stack)
    stack[8:20, 10:25, 2] = 2
    np.save(source, stack)
    with pytest.raises(ValueError, match="overlay source changed"):
        plot._reproduce_panel(report, 0)


@pytest.mark.parametrize("mode,all_on_all,all_outlines", [
    ("filled", False, False), ("filled", True, False),
    ("outlines", False, True),
])
def test_overlay_replay_preserves_each_compositing_choice_and_mask_identity(
        tmp_path, monkeypatch, mode, all_on_all, all_outlines):
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    source = tmp_path / "merged.npy"
    stack = np.zeros((64, 66, 5), dtype=np.uint16)
    stack[..., 0] = np.arange(66, dtype=np.uint16)[None, :] * 90
    stack[..., 1] = np.arange(64, dtype=np.uint16)[:, None] * 80
    stack[7:30, 9:32, 2] = 1
    stack[30:50, 34:55, 3] = 2
    stack[15:40, 18:43, 4] = 3
    np.save(source, stack)
    figure = plot.plot_image_mask_overlay(
        str(source), [0, 1], 0, 1, None, organelle_channel=0,
        save_pdf=False, mode=mode, all_on_all=all_on_all,
        all_outlines=all_outlines, filter_dict={
            "cell": ((1, 1000), (0, 65000)),
            "nucleus": ((1, 1000), (0, 65000))})
    report = plot._integrity_report(figure, fmt="png", dpi=100)
    assert len(report["panels"]) == 3
    for panel in report["panels"]:
        assert panel["reproducible"]
        rebuilt, matches = plot._reproduce_panel(report, panel["panel"])
        assert matches
        assert plot._array_digest(rebuilt) == panel["displayed_sha256"]
    altered = json.loads(json.dumps(report))
    altered["panels"][0]["steps"][0]["mask_planes"]["cell"] = 999
    with pytest.raises(ValueError, match="mask plane"):
        plot._reproduce_panel(altered, 0)
    altered = json.loads(json.dumps(report))
    del altered["panels"][0]["source"][0]["sha256"]
    with pytest.raises(ValueError, match="full export digest"):
        plot._reproduce_panel(altered, 0)
