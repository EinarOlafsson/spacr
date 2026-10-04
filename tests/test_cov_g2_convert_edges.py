"""Converter edges: legacy CZI scenes and plate-barcode inputs."""
from __future__ import annotations

import sys
import types

import pytest

from spacr import convert as cv


def _legacy_handle(entries, axes="SCZYX0"):
    return types.SimpleNamespace(
        filtered_subblock_directory=[types.SimpleNamespace(
            axes=a, start=s, shape=sh) for a, s, sh in entries],
        axes=axes, dtype="uint16")


def test_legacy_czi_scenes_are_bounded_per_scene():
    handle = _legacy_handle([
        ("SCZYX0", (0, 0, 0, 0, 0, 0), (1, 1, 1, 8, 8, 1)),
        ("SCZYX0", (0, 1, 0, 0, 0, 0), (1, 1, 1, 8, 8, 1)),
        ("SCZYX0", (1, 0, 2, 0, 0, 0), (1, 1, 1, 8, 8, 1)),
    ])
    series = cv._legacy_czi_series(handle)
    assert [s["czi_scene"] for s in series] == [0, 1]
    assert series[0]["n_c"] == 2
    assert series[0]["czi_bounds"]["Y"] == (0, 8)


def test_a_legacy_czi_is_described_by_its_first_scene(monkeypatch):
    handle = _legacy_handle([
        ("SCZYX0", (0, 0, 0, 0, 0, 0), (1, 1, 1, 8, 8, 1))])

    class _Czi:
        def __init__(self, path):
            pass

        def __enter__(self):
            return handle

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cv, "_import_reader",
                        lambda ext: types.SimpleNamespace(CziFile=_Czi))
    described = cv._describe_czi("/data/x.czi")
    assert described["n_series"] == 1 and described["reader"] == "czifile"


def test_an_oversized_barcode_input_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "barcodes.csv"
    path.write_bytes(b"x" * 10)
    real_open = type(path).open

    class _Huge:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            return b"x" * size

    monkeypatch.setattr(type(path), "open", lambda self, *a, **k: (
        _Huge() if self == path else real_open(self, *a, **k)))
    with pytest.raises(cv.ConfigurationError, match="exceeds 16 MiB"):
        cv._barcode_input_bytes(path)


@pytest.mark.parametrize("text, message", [
    (b"p1=A;p1 B", "source_plate=barcode"),
    (b" ; ", "no assignments"),
])
def test_barcode_sidecars_need_one_assignment_per_entry(text, message):
    if message == "no assignments":
        text = b"=;"
    with pytest.raises(cv.ConfigurationError):
        cv._conversion_sidecar_assignments(text, {"p1"}, "sidecar.txt")


def test_barcode_sidecars_skip_blank_entries():
    assert cv._conversion_sidecar_assignments(b"p1=BC1;;\n", {"p1"},
                                              "s.txt") == {"p1": "BC1"}


def test_finishing_barcodes_needs_completed_wells(monkeypatch):
    result = types.SimpleNamespace(is_complete=True,
                                   rows=lambda: [{"status": "failed"}])
    monkeypatch.setattr(cv, "_check_conversion_barcode_sidecars", lambda s: None)
    with pytest.raises(cv.ConfigurationError, match="No completed imported"):
        cv._finish_conversion_barcodes({"inputs": {}}, result)
