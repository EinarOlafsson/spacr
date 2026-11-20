"""Conversion map and barcode refusals explain recovery and preserve evidence."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from spacr import convert
from spacr.errors import ConfigurationError


def test_missing_conversion_map_names_the_path_and_recovery(tmp_path):
    """A missing map stays missing while the error explains which file to select."""
    path = tmp_path / "conversion_map.csv"
    with pytest.raises(ConfigurationError) as caught:
        convert.read_map(str(path))
    assert str(path) in str(caught.value)
    assert "Select an existing conversion map CSV" in str(caught.value)
    assert not path.exists()


def test_unreadable_conversion_map_keeps_the_cause_and_original(tmp_path, monkeypatch):
    """A reader failure keeps its cause and does not rewrite the selected map."""
    from spacr import tabular

    path = tmp_path / "conversion_map.csv"
    content = "target,source\nconverted.tif,original.tif\n"
    path.write_text(content, encoding="utf-8")
    cause = OSError("fixture map read failed")

    def refuse_read(*args, **kwargs):
        """Supply a controlled reader failure without touching the map."""
        raise cause

    monkeypatch.setattr(tabular, "read_table", refuse_read)
    with pytest.raises(ConfigurationError) as caught:
        convert.read_map(str(path))
    assert caught.value.__cause__ is cause
    assert str(path) in str(caught.value)
    assert "Check that the file is readable" in str(caught.value)
    assert path.read_text(encoding="utf-8") == content


def test_sidecar_encoding_failure_keeps_the_cause_and_bytes(tmp_path):
    """Invalid UTF-8 is refused with a repair instruction and no implicit conversion."""
    path = tmp_path / "barcode.txt"
    content = b"\xff00123"
    path.write_bytes(content)
    with pytest.raises(ConfigurationError) as caught:
        convert._conversion_sidecar_assignments(content, {"source"}, path)
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)
    assert str(path) in str(caught.value)
    assert "UTF-8" in str(caught.value)
    assert "Save barcode.txt as UTF-8 plain text" in str(caught.value)
    assert path.read_bytes() == content


@pytest.mark.parametrize("content, plates, diagnostic", [
    (b"", {"source"}, "empty or contains control characters"),
    (b"00123", {"source", "second"}, "source_plate=barcode entries"),
    (b"source=00123=extra", {"source"}, "source_plate=barcode entries"),
    (b"unknown=00123", {"source"}, "unknown, empty or duplicate assignments"),
    (b"source=00123\nsource=00456", {"source"}, "unknown, empty or duplicate assignments"),
    (b"source=", {"source"}, "unknown, empty or duplicate assignments"),
])
def test_malformed_sidecar_names_a_correction_without_rewriting(
        tmp_path, content, plates, diagnostic):
    """Ambiguous or malformed assignments remain refusals with actionable advice."""
    path = tmp_path / "barcode.txt"
    path.write_bytes(content)
    with pytest.raises(ConfigurationError) as caught:
        convert._conversion_sidecar_assignments(content, plates, path)
    message = str(caught.value)
    assert diagnostic in message
    assert str(path) in message
    assert "preview" in message.lower()
    assert path.read_bytes() == content


@pytest.mark.needs_symlink
def test_linked_sidecar_refusal_keeps_the_link_and_target(tmp_path):
    """The link guard still refuses linked input and preserves its target."""
    target = tmp_path / "original.txt"
    target.write_text("00123", encoding="utf-8")
    sidecar = tmp_path / "barcode.txt"
    sidecar.symlink_to(target)
    with pytest.raises(ConfigurationError) as caught:
        convert._conversion_barcode_sidecar_bytes(sidecar)
    assert "symbolic links" in str(caught.value)
    assert "Copy the intended barcode text into a regular barcode.txt file" in str(caught.value)
    assert sidecar.is_symlink()
    assert target.read_text(encoding="utf-8") == "00123"


def test_sidecar_read_failure_keeps_permission_cause_and_recovery(tmp_path, monkeypatch):
    """The original permission failure stays chained to the actionable refusal."""
    path = tmp_path / "barcode.txt"
    path.write_text("00123", encoding="utf-8")
    cause = PermissionError("fixture sidecar read denied")

    def refuse_open(*args, **kwargs):
        """Refuse only the isolated sidecar open used by this test."""
        raise cause

    with monkeypatch.context() as patch:
        patch.setattr(convert.os, "open", refuse_open)
        with pytest.raises(ConfigurationError) as caught:
            convert._conversion_barcode_sidecar_bytes(path)
    assert caught.value.__cause__ is cause
    assert str(path) in str(caught.value)
    assert "Check that barcode.txt is readable and is a regular file" in str(caught.value)
    assert path.read_text(encoding="utf-8") == "00123"


def test_conflicting_barcode_evidence_names_the_plate_and_recovery(tmp_path):
    """Conflicting evidence remains blocked without changing either assignment."""
    source = tmp_path / "A01" / "field.tif"
    sidecar = tmp_path / "barcode.txt"
    sidecar.write_text("00123", encoding="utf-8")
    plan = SimpleNamespace(
        sources=[SimpleNamespace(path=str(source), meta={"layout": "well"})],
        mappings=[SimpleNamespace(source=str(source), source_plate="source", plate="plate1")])
    explicit = {"plate1": "00456"}
    with pytest.raises(ConfigurationError) as caught:
        convert._discover_conversion_barcodes(plan, tmp_path, explicit)
    assert "output plate plate1" in str(caught.value)
    assert str(sidecar) in str(caught.value)
    assert "Make the explicit output plate barcode" in str(caught.value)
    assert explicit == {"plate1": "00456"}
    assert sidecar.read_text(encoding="utf-8") == "00123"


def test_changed_sidecar_requires_new_review_and_keeps_current_bytes(tmp_path):
    """A sidecar changed since discovery stays refused without restoring old data."""
    path = tmp_path / "barcode.txt"
    reviewed = hashlib.sha256(b"00123").hexdigest()
    path.write_bytes(b"00456")
    with pytest.raises(ConfigurationError) as caught:
        convert._check_conversion_barcode_sidecars({str(path): reviewed})
    assert "changed during conversion" in str(caught.value)
    assert "Finish editing barcode.txt, preview the input again" in str(caught.value)
    assert str(path) in str(caught.value)
    assert path.read_bytes() == b"00456"


@pytest.mark.parametrize("problem", ["overlapping_destination", "existing_bundle"])
def test_barcode_destination_refusal_preserves_existing_data(tmp_path, problem):
    """Preflight still refuses unsafe output placement before any conversion write."""
    source = tmp_path / "raw"
    source.mkdir()
    records = tmp_path / "records.csv"
    content = "barcode,well\n00123,A01\n"
    records.write_text(content, encoding="utf-8")
    destination = source / "converted" if problem == "overlapping_destination" else tmp_path / "out"
    if problem == "existing_bundle":
        bundle = destination / "plate_barcode_linkage"
        bundle.mkdir(parents=True)
        (bundle / "keep.txt").write_text("existing results", encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        convert._prepare_conversion_barcodes(
            {"plate_barcode_source": str(records)}, SimpleNamespace(), source, destination)
    assert "Choose" in str(caught.value)
    assert records.read_text(encoding="utf-8") == content
    assert list(source.iterdir()) == []
    if problem == "existing_bundle":
        assert (bundle / "keep.txt").read_text(encoding="utf-8") == "existing results"
    else:
        assert not destination.exists()


@pytest.mark.parametrize("problem", ["incomplete_conversion", "changed_records"])
def test_barcode_publication_refusal_names_recovery_without_a_bundle(tmp_path, problem):
    """Incomplete or changed evidence cannot publish a successful linkage bundle."""
    records = tmp_path / "records.csv"
    content = b"barcode,well\n00456,A01\n"
    records.write_bytes(content)
    bundle = tmp_path / "plate_barcode_linkage"
    prepared = {
        "bundle": bundle,
        "inputs": {str(records): hashlib.sha256(b"barcode,well\n00123,A01\n").hexdigest()},
    }
    result = SimpleNamespace(is_complete=problem == "changed_records")
    with pytest.raises(ConfigurationError) as caught:
        convert._finish_conversion_barcodes(prepared, result)
    message = str(caught.value)
    if problem == "incomplete_conversion":
        assert "Conversion is incomplete" in message
        assert "failed" in message.lower()
    else:
        assert "Plate barcode input changed during conversion" in message
        assert str(records) in message
        assert "preview" in message.lower()
    assert not bundle.exists()
    assert records.read_bytes() == content
