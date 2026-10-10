"""Figure integrity (F572): reused regions, splices, the cross-figure index,
camera bit depth and the writers outside save_figure.

Pinned here, on synthetic fields with planted manipulations:

* a cropped and re-zoomed region of one panel shown as another panel is a
  warning (a note when both panels trace to the same source), and distinct
  fields or similar single cells are not;
* a panel assembled from two images -- a noise-level change, a background
  step, or a copied patch -- is a splice warning, while padding, smooth
  pictures and pixel-replicated enlargements are not;
* the cross-figure index finds a flipped, contrast-changed repeat in a
  figure written to another folder, never trusts a replaced figure, and
  stays bounded;
* Micro-Manager and caller-declared camera bit depths set the saturation
  ceiling, and pixels above a declared depth are a bit-depth warning;
* writers that keep their own savefig call are checked too.
"""
from __future__ import annotations

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
def _guard_follows_the_argument(monkeypatch, tmp_path):
    monkeypatch.delenv(plot._INTEGRITY_ENV, raising=False)
    monkeypatch.delenv(plot._INDEX_ENV, raising=False)
    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: ("png", 100))
    monkeypatch.setattr("spacr.run_journal.current_run", lambda: None)


def _field(seed, size=256, noise=4.0, background=20.0, smooth=False):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), background, np.float32)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(12):
        cy, cx = rng.uniform(0, size, 2)
        radius = rng.uniform(8, 22)
        img += rng.uniform(60, 160) * np.exp(
            -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius * radius)))
    if not smooth:
        texture = cv2.GaussianBlur(
            rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.5)
        img = img * (1 + 0.25 * texture) + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def _cell(seed, size=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    radius = rng.uniform(20, 30)
    img = 15 + 140 * np.exp(-(((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
                              / (2 * radius * radius)))
    texture = cv2.GaussianBlur(
        rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.2)
    img = img * (1 + 0.2 * texture) + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def _zoomed(image, top, left, side):
    crop = image[top:top + side, left:left + side]
    return cv2.resize(crop, image.shape[::-1], interpolation=cv2.INTER_LINEAR)


def _figure(panels):
    fig = Figure(figsize=(2 * len(panels), 2), dpi=100)
    for index, panel in enumerate(panels):
        axes = fig.add_subplot(1, len(panels), index + 1)
        axes.imshow(panel, cmap="gray", vmin=0, vmax=255)
        axes.set_axis_off()
    return fig


def _report(fig, **kwargs):
    return plot._integrity_report(fig, fmt="png", dpi=100, **kwargs)


def _findings(report, check, severity="warning"):
    return [finding for finding in report["integrity"]["findings"]
            if finding["check"] == check and finding["severity"] == severity]


@pytest.mark.parametrize("side", [128, 192, 230])
def test_a_rezoomed_region_of_another_panel_is_flagged(side):
    fields = [_field(seed) for seed in range(3)]
    fields.append(_zoomed(fields[0], 20, 30, side))
    report = _report(_figure(fields))
    found = _findings(report, "region_reuse")
    assert [sorted(f["panels"]) for f in found] == [[0, 3]]
    finding = found[0]
    assert finding["panels"] == [3, 0]
    assert finding["zoom"] == pytest.approx(256 / side, rel=0.05)
    top, bottom, left, right = finding["box"]
    assert abs(top - 20) <= 4 and abs(left - 30) <= 4
    assert abs((bottom - top) - side) <= 6 and abs((right - left) - side) <= 6
    assert "magnification" in finding["message"]
    assert report["integrity"]["region_search"]["pairs_checked"] == 11


def test_a_noisy_mirrored_region_is_still_found():
    rng = np.random.default_rng(9)
    fields = [_field(seed) for seed in (4, 5)]
    zoom = _zoomed(fields[0], 40, 40, 192)[:, ::-1]
    fields.append(zoom + rng.normal(0, 4, zoom.shape).astype(np.float32))
    found = _findings(_report(_figure(fields)), "region_reuse")
    assert len(found) == 1 and found[0]["variant"] == "mirror"


def test_distinct_fields_and_similar_cells_carry_no_region_or_splice_warning():
    for panels in ([_field(seed) for seed in range(10, 16)],
                   [_cell(seed) for seed in range(12)]):
        report = _report(_figure(panels))
        assert not _findings(report, "region_reuse")
        assert not _findings(report, "splice")


def test_a_region_shared_by_panels_traced_to_one_source_is_a_note(tmp_path):
    source = tmp_path / "field.npy"
    full = _field(21)
    np.save(source, full)
    fig = _figure([full, _zoomed(full, 10, 10, 160)])
    for artist in (axes.images[0] for axes in fig.axes):
        plot._tag_panel(artist, source=str(source))
    report = _report(fig)
    assert not _findings(report, "region_reuse")
    notes = _findings(report, "region_reuse", "note")
    assert len(notes) == 1


def test_a_whole_panel_repeat_is_not_reported_again_as_a_region():
    fields = [_field(30), _field(31)]
    fields.append(fields[0][:, ::-1].copy())
    report = _report(_figure(fields))
    assert _findings(report, "duplicate")
    assert not _findings(report, "region_reuse")


def test_the_region_search_reports_what_its_time_budget_left_out(monkeypatch):
    monkeypatch.setattr(plot, "_REGION_SECONDS", -1.0)
    report = _report(_figure([_field(seed) for seed in range(3)]))
    assert report["integrity"]["region_search"] == {
        "pairs_checked": 0, "pairs_skipped": 6}


def test_region_helpers_abstain_on_unusable_input():
    assert plot._panel_grey(np.zeros((8, 8)), 64) is None
    assert plot._panel_grey(np.array(["a"] * 400).reshape(20, 20), 64) is None
    assert plot._panel_grey(np.zeros((20, 20, 2)), 64) is None
    assert plot._panel_grey(np.zeros((4, 4, 4, 4)), 64) is None
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[..., 0] = 90
    grey, scale = plot._panel_grey(rgb, 64)
    assert scale == 1.0 and grey.shape == (40, 40) and grey.max() == 30
    holes = np.ones((32, 32), np.float32)
    holes[0, 0] = np.nan
    assert np.isfinite(plot._panel_grey(holes, 64)[0]).all()
    assert np.isfinite(plot._panel_grey(np.full((32, 32), np.nan), 64)[0]).all()
    small, big = np.zeros((20, 20), np.float32), np.zeros((100, 100), np.float32)
    assert plot._region_scales(small, big) == []
    assert plot._region_scales(big, big, low=0.5, high=0.5, count=3) == [0.5]
    assert plot._best_region(small, small, [1.0], ["none"]) is None
    textured = _field(40, size=100)
    assert plot._best_region(textured, small, [1.0], ["none"]) is None
    assert plot._best_region(textured, textured, [2.0], ["none"]) is None
    coarse = plot._panel_grey(textured, 128)
    blank = plot._panel_grey(np.zeros((64, 64), np.float32), 128)
    assert plot._region_in_panel((coarse, coarse), (blank, blank)) is None
    tiny = plot._panel_grey(np.zeros((300, 20), np.float32), 128)
    assert plot._region_in_panel((coarse, coarse), (tiny, tiny)) is None


def test_region_in_panel_abstains_when_fine_checks_fail(monkeypatch):
    big = _field(41)
    small = _zoomed(big, 20, 30, 128)
    pairs = [(plot._panel_grey(a, plot._REGION_WORK),
              plot._panel_grey(a, plot._REGION_REFINE)) for a in (big, small)]
    assert plot._region_in_panel(*pairs) is not None
    monkeypatch.setattr(plot, "_REGION_DETAIL", 1.01)
    assert plot._region_in_panel(*pairs) is None
    real = plot._region_scales
    monkeypatch.setattr(plot, "_region_scales",
                        lambda b, s, low=None, high=None, count=0, minimum=0:
                        [] if low is not None
                        else real(b, s, count=count, minimum=minimum))
    assert plot._region_in_panel(*pairs) is None


def test_aligned_residuals_abstain_without_alignment_or_texture():
    rng = np.random.default_rng(8)
    textured = _field(43, size=96)
    assert plot._aligned_residual_correlation(textured, textured) > 0.9
    assert plot._aligned_residual_correlation(textured, _field(44, size=96)) is None
    flat = np.full((96, 96), 7, np.float32)
    assert plot._aligned_residual_correlation(flat, flat) is None
    tiny = rng.normal(size=(12, 12)).astype(np.float32)
    assert plot._aligned_residual_correlation(tiny, tiny) is None
    ramp = np.tile(np.arange(96, dtype=np.float32), (96, 1))
    noisy_ramp = ramp + rng.normal(0, 1, ramp.shape).astype(np.float32)
    assert plot._aligned_residual_correlation(ramp, noisy_ramp) is None


def _spliced(kind, seed):
    rng = np.random.default_rng(seed)
    first = _field(seed, noise=3)
    if kind == "noise":
        second = _field(seed + 1, noise=10)
        out = first.copy()
        out[:, 140:] = second[:, 140:]
    elif kind == "offset":
        second = _field(seed + 1, background=45)
        out = first.copy()
        out[110:, :] = second[110:, :]
    else:
        out = _field(seed)
        out[170:210, 160:200] = out[30:70, 40:80]
    return out, rng


def test_two_fields_with_different_noise_are_a_splice():
    image, _rng = _spliced("noise", 50)
    found = _findings(_report(_figure([image])), "splice")
    assert len(found) == 1
    seam = found[0]
    assert seam["kind"] == "seam" and seam["axis"] == "vertical"
    assert abs(seam["position"] - 140) <= 32
    assert (seam["noise_ratio"] or 0) > 1.6 or seam["offset"] >= 1.0
    assert "seam" in seam["message"]


def test_a_background_step_is_a_horizontal_splice():
    image, _rng = _spliced("offset", 60)
    found = _findings(_report(_figure([image])), "splice")
    assert [(f["kind"], f["axis"], f["position"]) for f in found] == [
        ("seam", "horizontal", 110)]
    assert found[0]["offset"] >= 1.0


def test_a_copied_patch_is_a_splice():
    image, _rng = _spliced("clone", 70)
    found = _findings(_report(_figure([image])), "splice")
    assert [f["kind"] for f in found] == ["clone"]
    clone = found[0]
    assert clone["shift"] == [140, 120]
    top, bottom, left, right = clone["box"]
    assert 20 <= top <= 40 and 30 <= left <= 50 and bottom - top >= 20


def test_padding_smooth_and_replicated_pictures_are_not_splices():
    padded = _field(80)
    padded[:, :60] = 0
    replicated = cv2.resize(_field(81, size=48), (240, 240),
                            interpolation=cv2.INTER_NEAREST)
    smooth = [_field(seed, smooth=True) for seed in range(82, 86)]
    report = _report(_figure([padded, replicated] + smooth))
    assert not _findings(report, "splice")


def test_splice_findings_map_back_through_replication_and_resizing():
    image, _rng = _spliced("clone", 70)
    doubled = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    big = cv2.resize(image, (1100, 1100), interpolation=cv2.INTER_LINEAR)
    grey, rows, columns, scale = plot._splice_work(doubled)
    assert grey.shape == (256, 256) and scale == 1.0
    assert list(rows[:3]) == [0, 2, 4]
    found = plot._splice_findings(
        [{"panel": 0, "shape": list(doubled.shape)}], [doubled])
    assert found and abs(found[0]["shift"][0] - 280) <= 2
    assert abs(found[0]["shift"][1] - 240) <= 2
    grey, rows, columns, scale = plot._splice_work(big)
    assert grey.shape == (512, 512) and scale == pytest.approx(512 / 1100)
    step, _rng = _spliced("offset", 60)
    wide = np.pad(step, ((0, 0), (0, 0)))
    found = plot._splice_findings([{"panel": 0, "shape": [256, 256]}], [wide])
    assert found[0]["position"] == 110


def test_splice_helpers_abstain_on_small_flat_or_ambiguous_input():
    assert plot._splice_work(np.zeros((8, 8))) is None
    assert plot._splice_work(np.zeros((40, 200), np.float32)) is None
    flat = np.zeros((128, 128), np.float32)
    assert plot._clone_region(flat) is None
    assert plot._clone_region(np.zeros((20, 20), np.float32)) is None
    assert plot._seam(flat) is None
    assert plot._seam(np.zeros((20, 50), np.float32)) is None
    ramp = np.tile(np.arange(128, dtype=np.float32), (128, 1))
    assert plot._seam(ramp) is None
    assert not plot._noise_like(flat)
    assert plot._noise_like(_field(90))
    assert plot._splice_findings([{"panel": 0, "shape": [32, 32]}],
                                 [np.zeros((32, 32))]) == []
    assert plot._splice_findings([{"panel": 0, "shape": [100, 30]}],
                                 [np.zeros((100, 30))]) == []
    left = np.zeros((64, 64), np.float32)
    assert plot._noise_by_intensity(left, left) is None
    rng = np.random.default_rng(3)
    mixed_left = rng.normal(100, 2, (64, 64)).astype(np.float32)
    mixed_right = rng.normal(100, 2, (64, 64)).astype(np.float32)
    mixed_left[:, :32] += rng.normal(0, 8, (64, 32)).astype(np.float32)
    mixed_right[:, 32:] += rng.normal(0, 8, (64, 32)).astype(np.float32)
    assert plot._noise_by_intensity(mixed_left, mixed_right) is None or abs(
        plot._noise_by_intensity(mixed_left, mixed_right)) < np.log(1.6)


def test_clone_confirmation_rejects_broad_and_periodic_peaks():
    texture = np.random.default_rng(5).normal(0, 1, (64, 64))
    corr = np.zeros((128, 128))
    corr[64 + 20, 64 + 20] = 0.5
    corr[64 + 20, 64 + 21] = 0.4
    assert plot._confirm_clone(texture, corr, (84, 84), 0.5) is None
    corr[64 + 20, 64 + 21] = 0
    corr[64 + 40, 64 + 40] = 0.4
    assert plot._confirm_clone(texture, corr, (84, 84), 0.5) is None
    corr[64 + 40, 64 + 40] = 0
    corr[64 + 20, 64] = 0.4
    assert plot._confirm_clone(texture, corr, (84, 84), 0.5) is None
    corr[64 + 20, 64] = 0
    assert plot._confirm_clone(texture, corr, (84, 84), 0.5) is None
    corr[64 + 21, 64 + 20] = 0
    assert plot._confirm_clone(texture, corr, (85, 85), 0.5) is None
    agreeing = np.zeros((64, 64))
    agreeing[5:15, 5:15] = 1.0
    assert plot._confirm_clone(agreeing, np.zeros((128, 128)),
                               (64 + 10, 64 + 10), 0.5) is None


def _export(fig, path):
    return plot.save_figure(fig, str(path), fmt="png", integrity=True)


def test_the_index_finds_a_repeat_written_to_another_folder(tmp_path, monkeypatch):
    shared = tmp_path / "project.jsonl"
    monkeypatch.setenv(plot._INDEX_ENV, str(shared))
    first = [_field(seed) for seed in (100, 101, 102)]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _export(_figure(first), tmp_path / "a" / "first.png")
    changed = first[1][:, ::-1] * 0.8 + 10
    out = _export(_figure([_field(103), changed, first[2]]),
                  tmp_path / "b" / "second.png")
    report = json.loads(open(out + ".provenance.json").read())
    found = _findings(report, "cross_figure_index")
    assert [(f["panels"], f["prior_panel"], f["identical"]) for f in found] == [
        ([1], 1, False), ([2], 2, True)]
    assert found[0]["prior_figure_path"] == str(tmp_path / "a" / "first.png")
    assert found[0]["correlation"] >= plot._DUPLICATE_CORRELATION
    lines = shared.read_text().splitlines()
    assert len(lines) == 2
    assert (tmp_path / "a" / plot._INDEX_NAME).exists()
    assert (tmp_path / "b" / plot._INDEX_NAME).exists()


def test_the_index_does_not_repeat_a_folder_finding_and_notes_one_source(
        tmp_path, monkeypatch):
    source = tmp_path / "field.npy"
    image = _field(110)
    np.save(source, image)

    def tagged():
        fig = _figure([image, _field(111)])
        plot._tag_panel(fig.axes[0].images[0], source=str(source))
        return fig

    _export(tagged(), tmp_path / "one.png")
    out = _export(tagged(), tmp_path / "two.png")
    report = json.loads(open(out + ".provenance.json").read())
    assert [f["check"] for f in report["integrity"]["findings"]
            if f["check"].startswith("cross_figure")] == [
        "cross_figure_duplicate", "cross_figure_duplicate"]
    monkeypatch.setattr(plot, "_prior_figure_findings", lambda *a: [])
    report = _report(tagged(), destination=str(tmp_path / "three.png"))
    notes = _findings(report, "cross_figure_index", "note")
    assert len(notes) == 1 and notes[0]["identical"]


def test_the_index_never_trusts_a_replaced_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "_prior_figure_findings", lambda *a: [])
    first = [_field(120), _field(121)]
    old = _export(_figure(first), tmp_path / "old.png")
    with open(old, "ab") as handle:
        handle.write(b"edited")
    report = _report(_figure([first[0]]), destination=str(tmp_path / "new.png"))
    assert not _findings(report, "cross_figure_index")


def test_the_index_skips_entries_it_cannot_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "_prior_figure_findings", lambda *a: [])
    first = [_field(130), _field(131)]
    old = _export(_figure(first), tmp_path / "old.png")
    sidecar = old + ".provenance.json"
    index = tmp_path / plot._INDEX_NAME
    entry = json.loads(index.read_text().splitlines()[-1])
    current = _figure([first[0], first[1][::-1] * 0.9])
    variants = []
    for change in ({"sidecar": None}, {"sidecar": str(tmp_path / "gone.json")},
                   {"figure": str(tmp_path)},
                   {"figure_sha256": "0" * 64}):
        variants.append(dict(entry, **change))
    lines = ["not json", json.dumps([1]), json.dumps({"v": 2})]
    lines += [json.dumps(dict(variant, figure=variant.get("figure")
                              if "figure" in variant and variant["figure"] != old
                              else str(tmp_path / f"f{n}.png")))
              for n, variant in enumerate(variants)]
    index.write_text("\n".join(lines) + "\n")
    report = _report(current, destination=str(tmp_path / "new.png"))
    assert not _findings(report, "cross_figure_index")
    big = dict(entry, figure=str(tmp_path / "huge.json"), sidecar=sidecar)
    (tmp_path / "huge.json").write_bytes(b"x")
    os.truncate(tmp_path / "huge.json", 257 * 1024 ** 2)
    index.write_text(json.dumps(big) + "\n")
    assert not _findings(_report(current, destination=str(tmp_path / "n.png")),
                         "cross_figure_index")
    record = json.loads(open(sidecar).read())
    record["panels"][0]["displayed_sha256"] = "changed"
    record["panels"][1].pop("spatial_v1")
    with open(sidecar, "w") as handle:
        json.dump(record, handle)
    index.write_text(json.dumps(dict(entry, figure_sha256=record[
        "figure_sha256"])) + "\n")
    report = _report(current, destination=str(tmp_path / "m.png"))
    assert not _findings(report, "cross_figure_index")
    with open(sidecar, "w") as handle:
        handle.write("[1]")
    assert plot._indexed_prior_panel(entry, entry["panels"][0], {}) is None
    with open(sidecar, "w") as handle:
        handle.write("{broken")
    assert plot._indexed_prior_panel(entry, entry["panels"][0], {}) is None


def test_index_candidates_need_a_signature_and_pass_the_texture_test(
        tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "_prior_figure_findings", lambda *a: [])
    image = _field(140)
    _export(_figure([image]), tmp_path / "old.png")
    index = tmp_path / plot._INDEX_NAME
    entry = json.loads(index.read_text())
    entry["panels"].append("not a panel")
    entry["panels"][0]["hash"] = "zz"
    index.write_text(json.dumps(entry) + "\n")
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    assert not _findings(_report(_figure([blurred]),
                                 destination=str(tmp_path / "n.png")),
                         "cross_figure_index")
    entry["panels"][0]["hash"] = None
    index.write_text(json.dumps(entry) + "\n")
    monkeypatch.setattr(plot, "_INDEX_HAMMING", 64)
    assert not _findings(_report(_figure([blurred]),
                                 destination=str(tmp_path / "n.png")),
                         "cross_figure_index")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    scene = _field(141, smooth=True)
    noise = np.random.default_rng(2)
    first = scene + noise.normal(0, 12, scene.shape).astype(np.float32)
    second = scene + noise.normal(0, 12, scene.shape).astype(np.float32)
    _export(_figure([first]), fresh / "old.png")
    assert not _findings(_report(_figure([second]),
                                 destination=str(fresh / "n.png")),
                         "cross_figure_index")
    assert _findings(_report(_figure([cv2.GaussianBlur(first, (0, 0), 2)]),
                             destination=str(fresh / "n.png")),
                     "cross_figure_index")
    monkeypatch.setattr(plot, "_panel_thumbnail", lambda array: None)
    assert not _findings(_report(_figure([second]),
                                 destination=str(fresh / "n.png")),
                         "cross_figure_index")


def test_index_query_survives_a_bad_panel_signature(tmp_path, monkeypatch):
    fig = _figure([_field(150)])
    real = plot._decode_spatial_signature
    calls = {"n": 0}

    def flaky(record):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("bad")
        return real(record)

    monkeypatch.setattr(plot, "_decode_spatial_signature", flaky)
    report = _report(fig, destination=str(tmp_path / "x.png"))
    assert "index_hash" in report["panels"][0]
    monkeypatch.setattr(plot, "_decode_spatial_signature",
                        lambda record: (_ for _ in ()).throw(ValueError("x")))
    report = _report(fig, destination=str(tmp_path / "x.png"))
    assert "index_hash" not in report["panels"][0]
    monkeypatch.setattr(plot, "_index_findings",
                        lambda *a: (_ for _ in ()).throw(OSError("x")))
    assert _report(fig, destination=str(tmp_path / "x.png")) is not None


def test_the_index_stays_bounded_and_survives_unwritable_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "_INDEX_LIMIT", 4096)
    report = {"figure_sha256": "a" * 64, "created": "now",
              "panels": [{"panel": 0, "displayed_sha256": "b" * 64,
                          "index_hash": "0" * 16,
                          "source": [{"path": "/x"}, "bad"]}]}
    figure = tmp_path / "f.png"
    for _ in range(40):
        plot._record_in_index(report, figure, str(figure) + ".json")
    index = tmp_path / plot._INDEX_NAME
    assert index.stat().st_size <= 4096
    entries = plot._read_figure_index(str(index))
    assert len(entries) == 1 and entries[0]["panels"][0]["sources"] == ["/x"]
    with open(index, "ab") as handle:
        handle.write(b"x" * 5000 + b"\n")
    assert plot._read_figure_index(str(index)) == []
    assert plot._read_figure_index(str(tmp_path / "missing")) == []
    report["panels"] = report["panels"] * 200
    plot._record_in_index(report, tmp_path / "g.png", "s")
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(plot._INDEX_ENV, str(blocker / "index.jsonl"))
    plot._record_in_index({"panels": []}, tmp_path / "h.png", "s")


def test_index_paths_follow_the_run_and_the_environment(tmp_path, monkeypatch):
    class _Run:
        dir = tmp_path / "run"

    monkeypatch.setattr("spacr.run_journal.current_run", lambda: _Run())
    monkeypatch.setenv(plot._INDEX_ENV, str(tmp_path / "out" / plot._INDEX_NAME))
    paths = plot._figure_index_paths(tmp_path / "out" / "f.png")
    assert paths == [str(tmp_path / "out" / plot._INDEX_NAME),
                     str(tmp_path / "run" / plot._INDEX_NAME)]
    monkeypatch.setattr("spacr.run_journal.current_run",
                        lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert len(plot._figure_index_paths(tmp_path / "f.png")) == 2


def test_hashes_cover_every_flip_and_rotation():
    grey = np.random.default_rng(7).integers(0, 255, (128, 128)).astype(np.uint8)
    hashes = plot._index_hashes(grey)
    assert plot._index_hash(np.rot90(grey)[:, ::-1]) in hashes
    assert len(next(iter(hashes))) == 16


def _micromanager_tiff(path, image, metadata):
    import tifffile

    tifffile.imwrite(path, image, extratags=[(51123, "s", 0, metadata, True)])


@pytest.mark.parametrize("metadata,ceiling,source", [
    (json.dumps({"BitDepth": 12}), 4095, "Micro-Manager BitDepth"),
    (json.dumps({"Summary": {"BitDepth": "14"}}), 16383,
     "Micro-Manager BitDepth"),
    (json.dumps({"Camera": "x"}), 65535, "storage dtype"),
])
def test_micromanager_bit_depth_sets_the_saturation_ceiling(
        tmp_path, metadata, ceiling, source):
    from PIL import Image

    image = np.full((32, 32), 100, np.uint16)
    image[:4] = 4095
    path = tmp_path / "mm.tif"
    _micromanager_tiff(path, image, metadata)
    with Image.open(path) as opened:
        record = plot._source_sensor_range(opened, np.array(opened))
    assert record["ceiling"] == ceiling and record["source"] == source


@pytest.mark.parametrize("metadata,reason", [
    ("{broken", "Expecting"),
    (json.dumps([1]), "not an object"),
    (json.dumps({"BitDepth": "twelve"}), "whole number"),
    (json.dumps({"BitDepth": True}), "whole number"),
])
def test_malformed_micromanager_metadata_keeps_the_storage_ceiling(
        tmp_path, metadata, reason):
    from PIL import Image

    image = np.full((32, 32), 100, np.uint16)
    path = tmp_path / "mm.tif"
    _micromanager_tiff(path, image, metadata)
    with Image.open(path) as opened:
        record = plot._source_sensor_range(opened, np.array(opened))
    assert record["ceiling"] == 65535
    assert reason in record["micromanager_reason"]


def test_micromanager_tag_forms_and_bounds():
    assert plot._micromanager_bits({}) is None
    assert plot._micromanager_bits({51123: (json.dumps({"BitDepth": 10}),)}) == 10
    assert plot._micromanager_bits({51123: b'{"BitDepth": 11}'}) == 11
    with pytest.raises(ValueError):
        plot._micromanager_bits({51123: b"x" * (1024 * 1024 + 1)})
    with pytest.raises(ValueError):
        plot._micromanager_bits({51123: 5})


def test_pixels_above_a_declared_depth_are_a_bit_depth_warning(tmp_path):
    from PIL import Image

    image = np.full((32, 32), 100, np.uint16)
    image[0, 0] = 5000
    path = tmp_path / "mm.tif"
    _micromanager_tiff(path, image, json.dumps({"BitDepth": 12}))
    with Image.open(path) as opened:
        raw = np.array(opened)
        record = plot._source_sensor_range(opened, raw)
    assert record["contradiction"] and record["observed_max"] == 5000
    assert record["ceiling"] == 65535
    fig = Figure(figsize=(2, 2), dpi=100)
    artist = fig.add_subplot(111).imshow(raw, cmap="gray")
    plot._tag_panel(artist, source=str(path), raw=raw, sensor_range=record)
    found = _findings(_report(fig), "bit_depth")
    assert len(found) == 1 and found[0]["declared_bits"] == 12
    assert "Micro-Manager BitDepth" in found[0]["message"]


def test_a_caller_declared_bit_depth_is_validated_and_named():
    raw = np.full((32, 32), 100, np.uint16)
    raw[:2] = 1023
    fig = Figure(figsize=(2, 2), dpi=100)
    artist = fig.add_subplot(111).imshow(raw, cmap="gray")
    plot._tag_panel(artist, raw=raw, significant_bits=10)
    report = _report(fig)
    saturation = _findings(report, "saturation")
    assert saturation and "caller-declared bit depth" in saturation[0]["message"]
    assert report["panels"][0]["sensor_range"]["ceiling"] == 1023
    signed = plot._declared_sensor_range(raw.astype(np.int16), 10, "x")
    assert signed["ceiling"] == 32767 and "unsigned" in signed["reason"]
    wide = plot._declared_sensor_range(raw, 20, "x")
    assert "outside" in wide["reason"]
    floats = plot._declared_sensor_range(raw.astype(np.float32), 10, "x")
    assert floats["ceiling"] is None
    empty = plot._declared_sensor_range(np.zeros((0, 0), np.uint16), 8, "x")
    assert empty["ceiling"] == 255


def test_an_ome_declaration_the_pixels_exceed_is_a_contradiction(tmp_path):
    import tifffile
    from PIL import Image

    image = np.full((32, 32), 100, np.uint16)
    image[0, 0] = 5000
    path = tmp_path / "ome.tif"
    tifffile.imwrite(path, image, ome=True,
                     metadata={"axes": "YX", "SignificantBits": 12})
    with Image.open(path) as opened:
        record = plot._source_sensor_range(opened, np.array(opened))
    assert record.get("contradiction") is True
    assert record["metadata"] == "OME Pixels SignificantBits"


def test_checked_savefig_stamps_jpeg_and_eps_writers(tmp_path, capsys):
    fig = _figure([_field(160)])
    out = plot._checked_savefig(fig, tmp_path / "f.jpg", dpi=50,
                                integrity=True)
    report = json.loads(open(str(out) + ".provenance.json").read())
    assert report["format"] == "jpg" and _findings(report, "lossy_format")
    plot._checked_savefig(fig, tmp_path / "f.png", integrity=True,
                          metadata={"Title": "t"})
    assert (tmp_path / "f.png.provenance.json").exists()
    plot._checked_savefig(fig, tmp_path / "plain.png", integrity=False)
    assert not (tmp_path / "plain.png.provenance.json").exists()
    plot._checked_savefig(Figure(), tmp_path / "empty.eps", integrity=True)
    assert not (tmp_path / "empty.eps.provenance.json").exists()


def test_checked_savefig_writes_even_when_the_check_fails(tmp_path, monkeypatch,
                                                          capsys):
    fig = _figure([_field(161)])
    monkeypatch.setattr(plot, "_integrity_report",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    plot._checked_savefig(fig, tmp_path / "a.png", integrity=True)
    assert (tmp_path / "a.png").exists()
    assert "could not run" in capsys.readouterr().out
    monkeypatch.undo()
    monkeypatch.setattr(plot, "_finish_integrity",
                        lambda *a: (_ for _ in ()).throw(OSError("disk")))
    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 100))
    plot._checked_savefig(fig, tmp_path / "b.png", integrity=True)
    assert "could not be written" in capsys.readouterr().out


def test_the_figure_settings_fallback_and_the_sweep_writer_are_checked(
        tmp_path, monkeypatch):
    from spacr import gene_measurement_sweep
    from spacr.qt.widgets import figure_settings

    monkeypatch.setenv(plot._INTEGRITY_ENV, "1")
    fig = _figure([_field(170)])
    written = figure_settings.save_figure_as(None, fig, str(tmp_path / "f.jpg"))
    assert written and (tmp_path / "f.jpg.provenance.json").exists()
    gene_measurement_sweep._write(_figure([_field(171)]),
                                  str(tmp_path / "sweep.png"))
    assert (tmp_path / "sweep.png.provenance.json").exists()
