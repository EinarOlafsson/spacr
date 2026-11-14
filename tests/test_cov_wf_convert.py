"""A conversion write that dies halfway through a field.

When a source dies mid-field, the planes already on disk are reported as
written, not swept into the failed list along with the rest, and the field
is only marked complete once a resumed run has finished it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import tifffile

from spacr import convert as cv


def _write(path, value=1, shape=(6, 6), **kwargs):
    """Write a synthetic TIFF whose pixels say which file it is."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.full(shape, value, np.uint16), **kwargs)
    return path


# ---------------------------------------------------------------------------
# A write that dies halfway through a field
# ---------------------------------------------------------------------------

def test_a_plane_already_written_is_not_also_reported_as_failed(
        tmp_path, monkeypatch):
    """When a source dies mid-field, the planes already on disk stay written.

    The failure sweep walks the whole group for the source that raised. If
    it did not exclude the targets already in ``written``, the ledger would
    claim a file failed while a complete, readable TIFF of it sits in the
    output folder — and a user reconciling the map against the folder would
    find one more image than the run says it produced.
    """
    root = tmp_path / "src"
    stack = np.stack([np.full((6, 6), 10 + z, np.uint16) for z in range(4)])
    os.makedirs(str(root / "run1" / "wt"))
    tifffile.imwrite(str(root / "run1" / "wt" / "fov01.tif"), stack,
                     metadata={"axes": "ZYX"})

    conversion_plan = cv.plan(cv.scan(str(root)))
    assert len(conversion_plan.mappings) == 4, "a 4-plane stack is 4 targets"

    real_write = cv._atomic_write

    def _die_on_the_second_plane(path, array):
        if path.endswith("Z02C01.tif"):
            raise RuntimeError("the disk went away mid-field")
        real_write(path, array)

    monkeypatch.setattr(cv, "_atomic_write", _die_on_the_second_plane)

    out = tmp_path / "out"
    result = cv.convert(conversion_plan, str(out))

    written = sorted(m.target for m in result.written)
    failed = sorted(m.target for m in result.failed)
    assert written == ["plate1_A01_T0001F001L01A01Z01C01.tif"]
    assert failed == [
        "plate1_A01_T0001F001L01A01Z02C01.tif",
        "plate1_A01_T0001F001L01A01Z03C01.tif",
        "plate1_A01_T0001F001L01A01Z04C01.tif",
    ]
    # The one written plane is a real file, and is not in the failed list.
    assert set(written).isdisjoint(failed)
    assert int(tifffile.imread(str(out / written[0]))[0, 0]) == 10
    assert [os.path.basename(p) for p, _ in result.skipped] == ["fov01.tif"]
    assert "disk went away" in result.skipped[0][1]
    # The half-written field is never marked complete on the checkpoint.
    assert result.resumed_fields == []
    rows = pd.read_csv(result.map_path)
    assert sorted(rows.loc[rows["status"] == "failed", "target"]) == failed
    assert sorted(rows.loc[rows["status"] == "converted", "target"]) == written

    # And the counterpart that proves the empty list above came from the
    # half-written field and not from resume being dead: with the disk back,
    # a resumed run finishes the field, and only then does a third run skip
    # it as already done.
    monkeypatch.setattr(cv, "_atomic_write", real_write)
    repaired = cv.convert(conversion_plan, str(out), resume=True)
    assert repaired.resumed_fields == [], "nothing was complete to resume yet"
    assert sorted(m.target for m in repaired.written) == failed
    again = cv.convert(conversion_plan, str(out), resume=True)
    assert again.resumed_fields == ["plate1/A01/f0001"]
    assert again.n_written == 0
    assert sorted(m.target for m in again.existing) == sorted(
        m.target for m in conversion_plan.mappings)
