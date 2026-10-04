"""Figure-integrity helpers refuse what they cannot vouch for."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from spacr import plot  # noqa: E402

NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _ome(pixels_attrs, channels=(), tiffdata=(), uuid=None):
    root = ET.Element(f"{{{NS}}}OME", {"UUID": uuid} if uuid else {})
    image = ET.SubElement(root, f"{{{NS}}}Image")
    pixels = ET.SubElement(image, f"{{{NS}}}Pixels", {
        "SizeZ": "1", "SizeT": "1", "SizeC": "1", "DimensionOrder": "XYCZT",
        **pixels_attrs})
    for attrs in channels:
        ET.SubElement(pixels, f"{{{NS}}}Channel", attrs)
    for attrs, file_uuid in tiffdata:
        data = ET.SubElement(pixels, f"{{{NS}}}TiffData", attrs)
        if file_uuid is not None:
            ET.SubElement(data, f"{{{NS}}}UUID").text = file_uuid
    return root


@pytest.mark.parametrize("root, message", [
    (ET.Element("{http://example.org}OME"), "not an OME metadata"),
    (_ome({"SizeZ": "0"}), "invalid OME dimensions"),
    (_ome({"SizeC": "2"}, channels=[{"SamplesPerPixel": "1"}]),
     "inconsistent OME channel"),
    (_ome({"SizeC": "3"}, channels=[{"SamplesPerPixel": "1"},
                                    {"SamplesPerPixel": "2"}]),
     "mixed OME channel"),
    (_ome({}, tiffdata=[({"IFD": "0"}, "")]), "unresolved external"),
    (_ome({}, tiffdata=[({"IFD": "0", "FirstZ": "4"}, None)]),
     "invalid OME plane coordinate"),
])
def test_unusable_ome_metadata_is_refused(root, message):
    with pytest.raises(ValueError, match=message):
        plot._ome_pixels_for_ifd(root, 0)


def test_a_mapping_for_another_file_is_skipped():
    root = _ome({}, tiffdata=[({"IFD": "0"}, "urn:uuid:other")],
                uuid="urn:uuid:this")
    with pytest.raises(ValueError, match="no unique OME Pixels"):
        plot._ome_pixels_for_ifd(root, 0)


def test_display_ranges_accept_one_pair_and_refuse_nonsense():
    assert plot._display_ranges([0, 255]) == [[0.0, 255.0]]
    assert plot._display_ranges("low to high") is None
    assert plot._display_ranges([[1, 2, 3]]) is None


def test_empty_planes_are_skipped_in_clip_statistics():
    raw = np.zeros((4, 0, 0), np.uint16)
    stats = plot._raw_clip_stats(raw, [[0, 1]] * 4)
    assert isinstance(stats, dict)


def test_a_tag_survives_unreadable_raw_data_and_frozen_artists():
    fig, ax = plt.subplots()
    artist = ax.imshow(np.zeros((4, 4)))

    class _Odd:
        def __array__(self, *args, **kwargs):
            raise RuntimeError("unreadable")

    plot._tag_panel(artist, raw=_Odd())
    assert getattr(artist, plot._PANEL_TAG)["raw_stats"] is None

    class _Frozen:
        __slots__ = ()

    assert isinstance(plot._tag_panel(_Frozen()), _Frozen)
    plt.close(fig)


def test_a_missing_source_is_recorded_as_missing(tmp_path, monkeypatch):
    assert plot._source_record(str(tmp_path / "none.tif")) == {
        "path": str(tmp_path / "none.tif"), "exists": False}
    existing = tmp_path / "a.tif"
    existing.write_bytes(b"x")

    def broken(path):
        raise OSError("gone")

    monkeypatch.setattr(plot.os.path, "getmtime", broken)
    record = plot._source_record(str(existing))
    assert record["exists"] and "mtime" not in record


def test_thumbnails_need_a_real_image():
    assert plot._panel_thumbnail(np.zeros((4, 4))) is None
    assert plot._panel_thumbnail(np.zeros((64, 64))) is None
    noisy = np.random.default_rng(0).normal(size=(64, 64))
    noisy[0, 0] = np.nan
    layout, detail = plot._panel_thumbnail(noisy)
    assert layout.shape and detail.shape
    smooth = np.add.outer(np.arange(64.0), np.arange(64.0))
    assert plot._panel_thumbnail(smooth) is not None


def test_the_integrity_preference_reads_false_when_unreadable(monkeypatch):
    from spacr.qt import preferences

    def broken():
        raise RuntimeError("settings locked")

    monkeypatch.setattr(preferences, "_get_figure_integrity", broken)
    assert plot._figure_integrity_enabled() is False


def test_a_failed_integrity_check_still_writes_the_figure(tmp_path, monkeypatch,
                                                          capsys):
    def broken(*a, **k):
        raise RuntimeError("check crashed")

    monkeypatch.setattr(plot, "_figure_integrity_enabled", lambda explicit=None: True)
    monkeypatch.setattr(plot, "check_figure_integrity", broken, raising=False)
    monkeypatch.setattr(plot, "_integrity_report", broken, raising=False)
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((4, 4)))
    target = tmp_path / "f.png"
    plot.save_figure(fig, target, fmt="png", close=True)
    assert target.exists()
    assert "could not run" in capsys.readouterr().out


def test_a_failed_provenance_sidecar_is_reported(tmp_path, monkeypatch, capsys):
    def broken(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(plot, "_figure_integrity_enabled", lambda explicit=None: True)
    monkeypatch.setattr(plot, "_integrity_report", lambda fig, **k: {"ok": True})
    monkeypatch.setattr(plot, "_integrity_metadata",
                        lambda report, fmt, existing: existing)
    monkeypatch.setattr(plot, "_finish_integrity", broken)
    fig, ax = plt.subplots()
    target = tmp_path / "g.png"
    plot.save_figure(fig, target, fmt="png", close=True)
    assert target.exists()
    assert "sidecar" in capsys.readouterr().out
