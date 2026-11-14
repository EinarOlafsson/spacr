"""Convert's barcode linkage refuses every input it cannot trust."""
from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from spacr import convert as cv
from spacr.errors import ConfigurationError


def test_a_sidecar_that_is_not_a_regular_file_is_refused(tmp_path):
    if hasattr(os, "mkfifo"):
        sidecar = tmp_path / "barcode.txt"
        os.mkfifo(sidecar)
    else:
        sidecar = os.devnull
    with pytest.raises(ConfigurationError, match="stable regular file"):
        cv._conversion_barcode_sidecar_bytes(str(sidecar))


def test_a_sidecar_that_grows_while_read_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "barcode.txt"
    path.write_text("00123", encoding="utf-8")
    real = os.fstat
    calls = []

    def growing(fd):
        result = real(fd)
        calls.append(1)
        if len(calls) == 2:
            return os.stat_result((result.st_mode, result.st_ino, result.st_dev,
                                   result.st_nlink, result.st_uid, result.st_gid,
                                   result.st_size + 1, 0, 0, 0))
        return result

    monkeypatch.setattr(cv.os, "fstat", growing)
    with pytest.raises(ConfigurationError, match="changed while being read"):
        cv._conversion_barcode_sidecar_bytes(str(path))


def _plan(tmp_path, *, layout="well", source=None, outputs=("plate1",),
          relative=("A01", "f.tif")):
    root = tmp_path / "raw"
    image = Path(source) if source else root.joinpath(*relative)
    sources = [types.SimpleNamespace(path=str(image), meta={"layout": layout})]
    mappings = [types.SimpleNamespace(source_plate="raw", plate=plate,
                                      source=str(image)) for plate in outputs]
    return types.SimpleNamespace(sources=sources, mappings=mappings), root


@pytest.mark.parametrize("kwargs, message", [
    ({"source": "/elsewhere/f.tif"}, "inside the source tree"),
    ({"layout": "plate_well", "relative": ("f.tif",)}, "identify this source plate"),
    ({"layout": None}, "unambiguous scanned source layout"),
    ({"outputs": ("plate1", "plate2")}, "multiple output plate identities"),
])
def test_discovery_refuses_what_it_cannot_place(tmp_path, kwargs, message):
    plan, root = _plan(tmp_path, **kwargs)
    root.mkdir()
    with pytest.raises(ConfigurationError, match=message):
        cv._discover_conversion_barcodes(plan, root, {})


def test_a_plan_with_no_wells_has_nothing_to_link(tmp_path):
    records = tmp_path / "r.csv"
    records.write_text("barcode,well\n1,A01\n", encoding="utf-8")
    plan = types.SimpleNamespace(sources=[], mappings=[])
    with pytest.raises(ConfigurationError, match="No planned image wells"):
        cv._prepare_conversion_barcodes(
            {"plate_barcode_source": str(records)}, plan, tmp_path / "raw",
            tmp_path / "out")


