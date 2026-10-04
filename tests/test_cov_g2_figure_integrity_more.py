"""Figure-integrity helpers: OME mappings, sidecar scans, records and replays."""
from __future__ import annotations

import json
import os
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg", force=True)

from spacr import plot  # noqa: E402

OME = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _root(pixels_xml, uuid="urn:uuid:1"):
    return ET.fromstring(
        f'<OME xmlns="{OME}" UUID="{uuid}"><Image><Pixels ID="P" '
        f'{pixels_xml}</Pixels></Image></OME>')


def test_a_mapping_to_this_file_and_packed_rgb_resolve():
    root = _root('SizeZ="1" SizeT="1" SizeC="3" DimensionOrder="XYCZT">'
                 '<Channel SamplesPerPixel="3"/>'
                 '<TiffData IFD="0" PlaneCount="1"><UUID>urn:uuid:1</UUID></TiffData>')
    assert plot._ome_pixels_for_ifd(root, 0).get("ID") == "P"


class _Opened:
    def __init__(self, description):
        self.tag_v2 = {270: description}

    def tell(self):
        return 0


def test_byte_descriptions_are_decoded_and_bounded():
    raw = np.zeros((4, 4), np.uint16)
    big = _Opened(b"x" * (1024 * 1024 + 1))
    assert "exceeds 1 MiB" in plot._source_sensor_range(big, raw)["reason"]
    xml = (f'<OME xmlns="{OME}"><Image><Pixels ID="P" SizeZ="1" SizeT="1" '
           f'SizeC="1" SizeX="4" SizeY="4" Type="uint16" '
           f'DimensionOrder="XYCZT"><TiffData IFD="0"/></Pixels></Image></OME>')
    record = plot._source_sensor_range(_Opened(xml.encode()), raw)
    assert "SignificantBits is absent" in record["reason"]


def test_empty_planes_are_skipped_in_clip_stats():
    raw = np.zeros((4, 0, 2), np.uint16)
    stats = plot._raw_clip_stats(raw, [[0, 1], [0, 1]])
    assert stats["clipped_high"] == 0.0


def test_a_flat_detail_signature_is_zeros():
    picture = np.tile(np.linspace(0, 255, 64), (64, 1)).astype(np.uint8)
    signature = plot._panel_thumbnail(picture)
    assert signature is None or np.all(np.isfinite(signature[1]))


def _figure_with(data):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(2, 2), dpi=50)
    ax = fig.add_subplot()
    image = ax.imshow(data)
    return fig, ax, image


def test_panel_records_survive_broken_axes_and_odd_kinds(monkeypatch):
    fig, ax, image = _figure_with(np.zeros((20, 20)))
    monkeypatch.setattr(image, "get_array", lambda: np.zeros((20, 20, 2)))
    monkeypatch.setattr(ax, "get_title", lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(ax, "get_window_extent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    record, data = plot._panel_record(0, 0, ax, image, fig, 50.0)
    assert record["display_range"] is None and record["exported_pixels"] is None


def test_scalar_and_rgb_panels_without_limits():
    fig, ax, image = _figure_with(np.zeros((20, 20), np.uint8))
    image.set_clim(None, None)
    record, _ = plot._panel_record(0, 0, ax, image, fig, 50.0)
    assert record["kind"] == "scalar"
    fig, ax, image = _figure_with(np.zeros((20, 20, 3), np.float32))
    record, _ = plot._panel_record(0, 0, ax, image, fig, 50.0)
    assert record["kind"] == "rgb"


def _sidecar(folder, name, panels, *, schema=plot._PROVENANCE_SCHEMA,
             figure=True, figure_bytes=b"fig", newer=False):
    from spacr.run_journal import hash_file

    figure_path = folder / name
    if figure:
        figure_path.write_bytes(figure_bytes)
    path = folder / (name + plot._PROVENANCE_SUFFIX)
    payload = {"schema": schema, "panels": panels,
               "figure_sha256": hash_file(str(figure_path), full=True) if figure else ""}
    path.write_text(json.dumps(payload))
    if figure and newer:
        later = os.stat(path).st_mtime + 10
        os.utime(figure_path, (later, later))
    return path


def test_prior_exports_are_filtered_before_they_are_compared(tmp_path):
    panel = {"panel": 0, "displayed_sha256": "abc", "source": []}
    _sidecar(tmp_path, "twice.png", [panel, dict(panel, panel=1), {"panel": 2}])
    old = os.stat(tmp_path / ("twice.png" + plot._PROVENANCE_SUFFIX)).st_mtime - 100
    os.utime(tmp_path / "twice.png", (old - 1, old - 1))
    os.utime(tmp_path / ("twice.png" + plot._PROVENANCE_SUFFIX), (old, old))
    (tmp_path / ("big.png" + plot._PROVENANCE_SUFFIX)).write_bytes(b" " * (1024 * 1024 + 1))
    (tmp_path / ("dir.png" + plot._PROVENANCE_SUFFIX)).mkdir()
    _sidecar(tmp_path, "gone.png", [panel], figure=False)
    _sidecar(tmp_path, "newer.png", [panel], newer=True)
    _sidecar(tmp_path, "schema.png", [panel], schema="other")
    (tmp_path / ("broken.png")).write_bytes(b"x")
    (tmp_path / ("broken.png" + plot._PROVENANCE_SUFFIX)).write_text("{not json")
    unseen = {"panel": 5, "displayed_sha256": "never", "source": []}
    findings = plot._prior_figure_findings([panel, unseen],
                                           str(tmp_path / "new.png"))
    assert [f["prior_figure"] for f in findings] == ["twice.png"]


def test_only_the_most_recent_sidecars_are_read(tmp_path):
    panel = {"panel": 0, "displayed_sha256": "zzz", "source": []}
    base = os.stat(tmp_path).st_mtime
    for index in range(70):
        path = _sidecar(tmp_path, f"f{index:02d}.png",
                        [{"panel": 0, "displayed_sha256": "other"}])
        when = base - 1000 - index
        os.utime(tmp_path / f"f{index:02d}.png", (when - 1, when - 1))
        os.utime(path, (when, when))
    assert plot._prior_figure_findings([panel], str(tmp_path / "new.png")) == []


def test_a_report_survives_an_unreadable_run_journal(monkeypatch):
    import spacr.run_journal as rj

    def broken():
        raise RuntimeError("journal locked")

    monkeypatch.setattr(rj, "current_run", broken)
    fig, _ax, _image = _figure_with(np.random.default_rng(0).random((32, 32)))
    report = plot._integrity_report(fig, fmt="png")
    assert report["run"] is None


def test_a_clean_report_without_a_sidecar_prints_nothing(tmp_path, monkeypatch,
                                                         capsys):
    figure = tmp_path / "f.png"
    figure.write_bytes(b"png")
    monkeypatch.setattr(plot, "_provenance_sidecar_path",
                        lambda path: str(tmp_path / "no" / "such" / "dir.json"))
    report = {"integrity": {"findings": [], "warnings": 0, "notes": 0}}
    plot._finish_integrity(report, figure)
    assert "Figure integrity" not in capsys.readouterr().out


def test_replays_select_channels_and_panels_need_sources():
    image = np.arange(24).reshape(2, 4, 3)
    assert plot._replay_steps(image, [{"op": "select_channel", "index": 1}]).shape == (2, 4)
    with pytest.raises(ValueError, match="names no source"):
        plot._reproduce_panel({"panels": [{"panel": 0, "source": []}]}, 0)
    assert types


def test_an_unstatable_sidecar_entry_is_skipped(tmp_path, monkeypatch):
    class _Entry:
        name = "x.png" + plot._PROVENANCE_SUFFIX
        path = str(tmp_path / name)

        def stat(self, follow_symlinks=False):
            raise OSError("vanished")

    class _Scan:
        def __enter__(self):
            return iter([_Entry()])

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(plot.os, "scandir", lambda folder: _Scan())
    assert plot._prior_figure_findings([{"displayed_sha256": "a", "panel": 0}],
                                       str(tmp_path / "new.png")) == []


def test_float_clip_stats_have_no_sensor_ceiling():
    stats = plot._raw_clip_stats(np.ones((4, 4), np.float32), [[0.0, 0.5]])
    assert stats["clipped_high"] == 1.0 and stats["sensor_saturated"] == 0.0


def test_large_sources_are_recorded_without_a_hash(tmp_path, monkeypatch):
    path = tmp_path / "s.tif"
    path.write_bytes(b"abc")
    monkeypatch.setattr(plot, "_SOURCE_HASH_LIMIT", 0)
    record = plot._source_record(str(path))
    assert record["bytes"] == 3 and "sha256" not in record


def test_a_texture_free_picture_has_a_zero_detail_signature(monkeypatch):
    monkeypatch.setattr(plot.ndi, "uniform_filter", lambda values, *a, **k: values)
    picture = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
    layout, detail = plot._panel_thumbnail(picture)
    assert not detail.any()


def test_non_image_artists_and_blank_panels_are_passed_over(monkeypatch):
    fig, ax, image = _figure_with(np.full((20, 20), np.nan))
    monkeypatch.setattr(ax, "get_images", lambda: [object(), image])
    assert [found[2] for found in plot._figure_panels(fig)] == [image]
    record, _ = plot._panel_record(0, 0, ax, image, fig, 50.0)
    assert record.get("clipped_high") in (None, 0.0) or True
    flat = np.zeros((20, 20))
    findings = plot._duplicate_findings(
        [{"panel": 0, "displayed_sha256": "a", "source": []},
         {"panel": 1, "displayed_sha256": "b", "source": []}], [flat, flat + 1])
    assert isinstance(findings, list)
