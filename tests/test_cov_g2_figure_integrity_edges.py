"""Figure-integrity helpers refuse what they cannot vouch for.

Two parametrised contracts: unusable OME metadata and unreplayable steps
are refused by name, and every other helper reports what it observed
(statistics, records, findings, printed notices) for the input it was
given. Each row is a module-level helper that does the setup and returns
the observed value, and the test compares it to the expected one.
"""
from __future__ import annotations

import functools
import inspect
import re
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



def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _refusal(call, fragment):
    """The part of the ValueError message that matches ``fragment``."""
    with pytest.raises(ValueError) as caught:
        call()
    found = re.search(fragment, str(caught.value))
    return found.group(0) if found else str(caught.value)


def _ome_refusal(root, fragment):
    return _refusal(lambda: plot._ome_pixels_for_ifd(root, 0), fragment)


def _mapping_for_another_file_is_skipped():
    root = _ome({}, tiffdata=[({"IFD": "0"}, "urn:uuid:other")],
                uuid="urn:uuid:this")
    return _refusal(lambda: plot._ome_pixels_for_ifd(root, 0),
                    "no unique OME Pixels")


def _unknown_replay_step_is_refused():
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    return _refusal(lambda: plot._replay_steps(image, [{"op": "rotate"}]),
                    "cannot be replayed")


def _empty_planes_are_skipped_in_clip_statistics():
    raw = np.zeros((4, 0, 0), np.uint16)
    return plot._raw_clip_stats(raw, [[0, 1]] * 4)


def _a_tag_survives_unreadable_raw_data_and_frozen_artists():
    fig, ax = plt.subplots()
    artist = ax.imshow(np.zeros((4, 4)))

    class _Odd:
        def __array__(self, *args, **kwargs):
            raise RuntimeError("unreadable")

    plot._tag_panel(artist, raw=_Odd())
    raw_stats = getattr(artist, plot._PANEL_TAG)["raw_stats"]

    class _Frozen:
        __slots__ = ()

    frozen_returned = isinstance(plot._tag_panel(_Frozen()), _Frozen)
    plt.close(fig)
    return raw_stats, frozen_returned


def _a_missing_source_is_recorded_as_missing(tmp_path, monkeypatch):
    missing = plot._source_record(str(tmp_path / "none.tif"))
    assert missing == {"path": str(tmp_path / "none.tif"), "exists": False}
    existing = tmp_path / "a.tif"
    existing.write_bytes(b"x")

    def broken(path):
        raise OSError("gone")

    monkeypatch.setattr(plot.os.path, "getmtime", broken)
    record = plot._source_record(str(existing))
    return missing["exists"], record["exists"], "mtime" in record


def _thumbnails_need_a_real_image():
    tiny = plot._panel_thumbnail(np.zeros((4, 4)))
    blank = plot._panel_thumbnail(np.zeros((64, 64)))
    noisy = np.random.default_rng(0).normal(size=(64, 64))
    noisy[0, 0] = np.nan
    layout, detail = plot._panel_thumbnail(noisy)
    smooth = np.add.outer(np.arange(64.0), np.arange(64.0))
    return (tiny, blank, layout.shape, detail.shape,
            plot._panel_thumbnail(smooth) is not None)


def _the_integrity_preference_reads_false_when_unreadable(monkeypatch):
    from spacr.qt import preferences

    def broken():
        raise RuntimeError("settings locked")

    monkeypatch.setattr(preferences, "_get_figure_integrity", broken)
    return plot._figure_integrity_enabled()


def _a_failed_integrity_check_still_writes_the_figure(tmp_path, monkeypatch,
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
    return target.exists(), "could not run" in capsys.readouterr().out


def _a_failed_provenance_sidecar_is_reported(tmp_path, monkeypatch, capsys):
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
    return target.exists(), "sidecar" in capsys.readouterr().out


def _panel(number, **fields):
    record = {"panel": number, "kind": "scalar", "shape": [4, 4],
              "display_range": [[0.0, 1.0]], "tagged": False}
    record.update(fields)
    return record


def _range_findings_compare_only_comparable_panels():
    panels = [
        _panel(1, display_range=None),
        _panel(2, compare="pair", display_range=[[0.0, 1.0]]),
        _panel(3, compare="pair", display_range=[[0.0, 2.0]]),
        _panel(4),
        _panel(5, tagged=True, channel="DAPI", display_range=[[1.0, 1.0]]),
        _panel(6, tagged=True, channel="DAPI", display_range=[[1.0, 1.0]]),
    ]
    return [f["panels"] for f in plot._range_findings(panels)]


def _lossy_and_resampling_checks_need_panels_to_judge():
    lossy = plot._lossy_findings("jpg", "jpg", [])
    panels = [{"panel": 1, "shape": [400, 400], "exported_pixels": [40, 40]},
              {"panel": 2, "shape": [4], "exported_pixels": [10, 10]},
              {"panel": 3, "shape": [4, 4], "exported_pixels": None}]
    notes = plot._resampling_findings(panels)
    return lossy, [n["panels"] if "panels" in n else n.get("panel")
                   for n in notes]


def _the_version_reads_unknown_without_a_version_module(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_version(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "_version" and level:
            raise ImportError("no version file")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_version)
    return plot._spacr_version()


def _finishing_without_a_writable_sidecar_still_reports(tmp_path, monkeypatch,
                                                        capsys):
    from spacr import run_journal

    figure = tmp_path / "f.png"
    figure.write_bytes(b"png")

    class _Run:
        def record_warning(self, line):
            raise RuntimeError("journal closed")

        def record_output(self, path, setting_key=None):
            raise RuntimeError("journal closed")

    monkeypatch.setattr(run_journal, "current_run", lambda: _Run())
    report = {"integrity": {"findings": [
        {"severity": "note", "message": "fine"},
        {"severity": "warning", "message": "ranges differ"}],
        "warnings": 1, "notes": 1}}
    first = plot._finish_integrity(report, figure) is not None
    real_open = open

    def refuse(path, *a, **k):
        if str(path).endswith(".tmp"):
            raise OSError("read-only")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", refuse)

    def broken():
        raise RuntimeError("no journal")

    monkeypatch.setattr(run_journal, "current_run", broken)
    second = plot._finish_integrity(report, figure) is None
    return first, second, "ranges differ" in capsys.readouterr().out


def _steps_replay_and_npy_sources_are_read(tmp_path):
    np.save(tmp_path / "a.npy", np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    image = plot._read_panel_source(tmp_path / "a.npy")
    out = plot._replay_steps(image, [
        {"op": "max_project", "axis": 0}, {"op": "crop", "box": [0, 2, 0, 2]},
        {"op": "rescale", "ranges": [0, 30]}, {"op": "to_uint8"}])
    return str(out.dtype), out.shape


@pytest.mark.parametrize("helper, expected", [
    pytest.param(functools.partial(
        _ome_refusal, ET.Element("{http://example.org}OME"),
        "not an OME metadata"), "not an OME metadata",
        id="ome-wrong-root-element"),
    pytest.param(functools.partial(
        _ome_refusal, _ome({"SizeZ": "0"}), "invalid OME dimensions"),
        "invalid OME dimensions", id="ome-zero-dimension"),
    pytest.param(functools.partial(
        _ome_refusal,
        _ome({"SizeC": "2"}, channels=[{"SamplesPerPixel": "1"}]),
        "inconsistent OME channel"),
        "inconsistent OME channel", id="ome-channel-count-mismatch"),
    pytest.param(functools.partial(
        _ome_refusal,
        _ome({"SizeC": "3"}, channels=[{"SamplesPerPixel": "1"},
                                       {"SamplesPerPixel": "2"}]),
        "mixed OME channel"),
        "mixed OME channel", id="ome-mixed-channel-samples"),
    pytest.param(functools.partial(
        _ome_refusal, _ome({}, tiffdata=[({"IFD": "0"}, "")]),
        "unresolved external"),
        "unresolved external", id="ome-unresolved-external-file"),
    pytest.param(functools.partial(
        _ome_refusal,
        _ome({}, tiffdata=[({"IFD": "0", "FirstZ": "4"}, None)]),
        "invalid OME plane coordinate"),
        "invalid OME plane coordinate", id="ome-plane-coordinate-out-of-range"),
    pytest.param(_mapping_for_another_file_is_skipped, "no unique OME Pixels",
                 id="ome-mapping-for-another-file-skipped"),
    pytest.param(_unknown_replay_step_is_refused, "cannot be replayed",
                 id="replay-unknown-step"),
])
def test_unusable_inputs_are_refused_by_name(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(functools.partial(plot._display_ranges, [0, 255]),
                 [[0.0, 255.0]], id="display-range-one-pair"),
    pytest.param(functools.partial(plot._display_ranges, "low to high"), None,
                 id="display-range-text-refused"),
    pytest.param(functools.partial(plot._display_ranges, [[1, 2, 3]]), None,
                 id="display-range-triple-refused"),
    pytest.param(_empty_planes_are_skipped_in_clip_statistics,
                 {"clipped_high": 0.0, "clipped_low": 0.0,
                  "sensor_saturated": 0.0},
                 id="empty-planes-skipped-in-clip-stats"),
    pytest.param(_a_tag_survives_unreadable_raw_data_and_frozen_artists,
                 (None, True), id="tag-survives-unreadable-and-frozen"),
    pytest.param(_a_missing_source_is_recorded_as_missing,
                 (False, True, False), id="source-record-missing-and-no-mtime"),
    pytest.param(_thumbnails_need_a_real_image,
                 (None, None, (32, 32), (96, 96), True),
                 id="snap-thumbnail-shapes"),
    pytest.param(_the_integrity_preference_reads_false_when_unreadable, False,
                 id="integrity-preference-unreadable-is-false"),
    pytest.param(_a_failed_integrity_check_still_writes_the_figure,
                 (True, True), id="failed-check-still-writes-figure"),
    pytest.param(_a_failed_provenance_sidecar_is_reported, (True, True),
                 id="failed-sidecar-reported"),
    pytest.param(_range_findings_compare_only_comparable_panels, [[2, 3]],
                 id="range-findings-comparable-only"),
    pytest.param(_lossy_and_resampling_checks_need_panels_to_judge,
                 ([], [[1]]), id="lossy-resampling-need-panels"),
    pytest.param(_the_version_reads_unknown_without_a_version_module,
                 "unknown", id="version-unknown-without-module"),
    pytest.param(_finishing_without_a_writable_sidecar_still_reports,
                 (True, True, True), id="finish-without-sidecar-reports"),
    pytest.param(_steps_replay_and_npy_sources_are_read, ("uint8", (2, 2)),
                 id="replay-steps-and-npy-source"),
])
def test_integrity_helpers_report_what_they_observed(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected
