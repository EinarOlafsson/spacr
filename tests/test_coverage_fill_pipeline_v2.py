"""Coverage-fill for spacr.pipeline_v2 — _record_cellpose_hash with no open
run, and FilenameMapper discovery next to a non-matching file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from spacr import pipeline_v2 as PV

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


class _FakeModel:
    def __init__(self, *a, **k):
        self.pretrained_model = k.get("pretrained_model")

    def eval(self, img, diameter=None, channel_axis=MISSING_CHANNEL_AXIS, **k):
        # channel_axis is named, not swallowed by **k. pipeline_v2 hands
        # Cellpose channels-last (H, W, C) images with a hard-coded axis, and
        # a mock that accepts any axis cannot tell the working -1 from the
        # channel_axis=3 that raised IndexError on every real run.
        check_cellpose_eval_call(img, channel_axis)
        images = img if isinstance(img, list) else [img]
        masks = []
        for image in images:
            m = np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
            m[1:4, 1:4] = 1
            masks.append(m)
        return masks, None, None


def _make_plate(dst: Path, wells=("A01",), fields=(1, 2), channels=3,
                 size=12) -> Path:
    import tifffile
    plate = dst / "plate1"; plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for well in wells:
        for f in fields:
            for c in range(channels):
                arr = rng.integers(0, 2000, size=(size, size)).astype(np.uint16)
                p = plate / f"plate1_{well}_T01F0{f}L01A01Z01C0{c}.tif"
                tifffile.imwrite(str(p), arr)
    return plate


# ---------------------------------------------------------------------------
# _record_cellpose_hash
# ---------------------------------------------------------------------------

class TestRecordHash:


    def test_no_open_run_returns(self, tmp_path, monkeypatch):
        import spacr.run_journal as rj
        # runs_root() is ~/.spacr/runs. Without this redirect the final
        # assertion globbed tmp_path -- a directory the journal would never
        # write to -- so it held for any behaviour whatsoever, including
        # _record_cellpose_hash quietly opening a run in the developer's real
        # home. Pointed at tmp_path, "nothing was written" is a claim about
        # the place the journal actually writes.
        runs = tmp_path / "runs"
        monkeypatch.setattr(rj, "runs_root", lambda: runs)
        ckpt = tmp_path / "m.pth"; ckpt.write_bytes(b"x")
        m = _FakeModel(pretrained_model=[str(ckpt)])
        # No open run → returns cleanly (line 423-424) rather than creating
        # one behind the caller's back.
        assert rj.current_run() is None
        PV._record_cellpose_hash(m, "cyto")
        assert rj.current_run() is None
        assert not runs.exists() or not list(runs.glob("**/manifest.json"))


class TestMapperEdges:
    def test_discover_skips_nonmatching_file(self, tmp_path, caplog):
        p = _make_plate(tmp_path)
        # Drop in a file that won't match the cellvoyager regex.
        (p / "not_a_valid_name.tif").write_bytes(b"x")
        mapper = PV.FilenameMapper.discover(p, metadata_type="cellvoyager")
        # It still discovers the valid files.
        assert mapper.field_ids()


