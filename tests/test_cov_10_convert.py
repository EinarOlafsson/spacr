"""Conversion helpers on the inputs that are not the happy path.

A plate format nobody supports, a folder name that parses as a position no
plate has, a source file that vanished between planning and resume, and a
TIFF that is on disk but carries no image: each of these decides whether a
conversion refuses loudly, skips a field, or silently resumes onto an
artifact that cannot be read. They are the branches a good run never takes,
which is exactly why they need a test of their own.
"""
from __future__ import annotations

import os

import pytest

from spacr import convert
from spacr.errors import ConfigurationError


def test_a_plate_format_spacr_does_not_know_is_refused_by_name():
    """``well_sequence`` must not invent wells for an unsupported plate.

    Silently falling back to 384 would name wells that the microscope never
    imaged, and every downstream table would carry those names as fact.
    """
    with pytest.raises(ConfigurationError) as excinfo:
        convert.well_sequence(7)
    message = str(excinfo.value)
    assert "7" in message
    assert "384" in message


def test_a_real_well_name_is_not_reported_as_off_plate():
    """``off_plate_reason`` stays silent for a name that IS a well.

    The function exists to catch ``ZZ99``-style typos; a warning on ``A01``
    would train users to ignore it.
    """
    assert convert.off_plate_reason("A01") is None
    assert convert.off_plate_reason("P24") is None
    assert convert.off_plate_reason("ZZ99") is not None


def test_a_source_that_cannot_be_stat_ed_records_the_error_not_a_size():
    """A missing source file yields an identity that says so.

    The resume guard compares source identities. If a vanished file produced
    the same identity as a present one, a half-converted plate would resume
    against a source that is no longer there.
    """
    identity = convert._source_identity("/no/such/file/for/spacr.tif")
    assert identity["path"] == os.path.abspath("/no/such/file/for/spacr.tif")
    assert "error" in identity
    assert "size" not in identity and "mtime_ns" not in identity


def test_a_truncated_tiff_is_not_a_resumable_artifact(tmp_path):
    """A file too short to hold a TIFF header is rejected before it is opened."""
    stub = tmp_path / "short.tif"
    stub.write_bytes(b"II*")
    assert convert._valid_converted_tiff(str(stub)) is False


def test_a_tiff_with_no_pages_is_not_a_resumable_artifact(tmp_path, monkeypatch):
    """A readable TIFF container holding zero pages must not count as done.

    tifffile opens such a file without complaint, so only the page check
    stands between a resumed run and a plate whose fields are empty files.
    """
    import tifffile

    target = tmp_path / "empty.tif"
    target.write_bytes(b"II*\x00" + b"\x00" * 64)

    class _NoPages:
        pages = []
        series = []

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(tifffile, "TiffFile", _NoPages)
    assert convert._valid_converted_tiff(str(target)) is False
