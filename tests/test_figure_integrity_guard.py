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

import json
import os

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
