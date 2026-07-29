"""Coverage-fill for spacr.pipeline_v2 — stream_masks_from_stack,
_record_cellpose_hash, run_v2 driven with a MOCKED CellposeModel (CPU).
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from spacr import pipeline_v2 as PV


class _FakeModel:
    def __init__(self, *a, **k):
        self.pretrained_model = k.get("pretrained_model")
    def eval(self, img, diameter=None, **k):
        images = img if isinstance(img, list) else [img]
        masks = []
        for image in images:
            m = np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
            m[1:4, 1:4] = 1
            masks.append(m)
        return masks, None, None


@pytest.fixture
def _mock_cp(monkeypatch):
    monkeypatch.setattr(
        PV, "cp_models",
        types.SimpleNamespace(CellposeModel=_FakeModel), raising=False)
    # pipeline_v2 imports cp_models INSIDE the function, so patch the
    # cellpose module directly too.
    import cellpose
    monkeypatch.setattr("cellpose.models.CellposeModel", _FakeModel)
    yield


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
    def test_no_checkpoint_paths_returns(self):
        # Model with no real ckpt files → early return (line 415-416).
        m = _FakeModel(pretrained_model=None)
        PV._record_cellpose_hash(m, "cyto")   # no exception

    def test_records_to_open_run(self, tmp_path, monkeypatch):
        # A real ckpt file + an open run → record_model called.
        ckpt = tmp_path / "model.pth"; ckpt.write_bytes(b"weights")
        m = _FakeModel(pretrained_model=[str(ckpt)])
        import spacr.run_journal as rj
        monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
        with rj.open_run("mask", {"src": "/x"}) as run:
            PV._record_cellpose_hash(m, "cyto")
        # Manifest should now carry a model entry.
        import json
        manifest = json.loads((run.dir / "manifest.json").read_text())
        assert manifest.get("model_hashes")

    def test_no_open_run_returns(self, tmp_path):
        ckpt = tmp_path / "m.pth"; ckpt.write_bytes(b"x")
        m = _FakeModel(pretrained_model=[str(ckpt)])
        # No open run → returns cleanly (line 423-424).
        PV._record_cellpose_hash(m, "cyto")

    def test_nested_pretrained_model(self, tmp_path, monkeypatch):
        # model.cp.pretrained_model nested path (lines 406-411).
        ckpt = tmp_path / "n.pth"; ckpt.write_bytes(b"x")
        inner = types.SimpleNamespace(pretrained_model=str(ckpt))
        m = types.SimpleNamespace(pretrained_model=None, cp=inner)
        import spacr.run_journal as rj
        monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
        with rj.open_run("mask", {"src": "/x"}):
            PV._record_cellpose_hash(m, "cyto")


# ---------------------------------------------------------------------------
# _read_plane
# ---------------------------------------------------------------------------

class TestReadPlane:
    def test_read_tif_2d(self, tmp_path):
        import tifffile
        p = tmp_path / "x.tif"
        tifffile.imwrite(str(p), np.ones((6, 6), dtype=np.uint16))
        assert PV._read_plane(str(p)).shape == (6, 6)

    def test_read_png_reduces_3d(self, tmp_path):
        from PIL import Image
        p = tmp_path / "x.png"
        Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(str(p))
        arr = PV._read_plane(str(p))
        assert arr.ndim == 2


# ---------------------------------------------------------------------------
# stream_masks_from_stack + run_v2 (mocked cellpose)
# ---------------------------------------------------------------------------

class TestStreamMasks:
    def test_empty_stacks_returns(self):
        assert PV.stream_masks_from_stack([]) == []

    def test_cellpose_missing_raises(self, tmp_path, monkeypatch):
        # Force the cellpose import inside stream_masks to fail.
        plate = _make_plate(tmp_path)
        mapper = PV.FilenameMapper.discover(plate, metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(plate, mapper,
                                                channels=(0, 1, 2))
        import builtins
        real = builtins.__import__
        def _block(name, *a, **k):
            if name == "cellpose" or name.startswith("cellpose."):
                raise ImportError("no cellpose")
            return real(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", _block)
        with pytest.raises(RuntimeError):
            PV.stream_masks_from_stack(stacks, model_name="cyto")

    def test_run_v2_end_to_end_mocked(self, tmp_path, _mock_cp):
        plate = _make_plate(tmp_path)
        out = PV.run_v2(plate, channels=(0, 1, 2), model_name="cyto",
                        metadata_type="cellvoyager", batch_fields=1)
        assert (plate / "merged").is_dir()
        stacks = list((plate / "merged").glob("stack_*.npy"))
        assert stacks
        # Mask channel appended → last dim grew.
        arr = np.load(stacks[0])
        assert arr.shape[-1] >= 4   # 3 channels + mask

    def test_run_v2_cpsam_model(self, tmp_path, _mock_cp):
        plate = _make_plate(tmp_path)
        PV.run_v2(plate, channels=(0, 1, 2), model_name="cpsam",
                  metadata_type="cellvoyager", batch_fields=2)
        assert (plate / "merged").is_dir()

    def test_stream_masks_keep_npz(self, tmp_path, _mock_cp):
        plate = _make_plate(tmp_path)
        mapper = PV.FilenameMapper.discover(plate, metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(plate, mapper,
                                                channels=(0, 1, 2))
        PV.stream_masks_from_stack(stacks, model_name="cyto",
                                     batch_fields=1, keep_npz=True)
        # keep_npz leaves the scratch NPZ files.
        scratch = stacks[0].path.parent / "_scratch"
        assert scratch.exists()

    def test_v1_normalized_pixels_are_passed_to_cellpose(
            self, tmp_path, monkeypatch):
        received = []

        class _CaptureModel:
            def __init__(self, *args, **kwargs):
                self.pretrained_model = None

            def eval(self, images, **kwargs):
                received.extend(np.asarray(image).copy() for image in images)
                return [
                    np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
                    for image in images
                ], None, None

        monkeypatch.setattr("cellpose.models.CellposeModel", _CaptureModel)
        plate = _make_plate(tmp_path, channels=2)
        mapper = PV.FilenameMapper.discover(
            plate, metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(
            plate, mapper, channels=(0, 1))
        raw = [np.load(stack.path)[..., [1, 0]] for stack in stacks]
        settings = {
            "lower_percentile": 2,
            "background": 100,
            "Signal_to_noise": 10,
            "remove_background": False,
            "cell_background": 100,
            "cell_Signal_to_noise": 10,
            "remove_background_cell": False,
            "nucleus_background": 100,
            "nucleus_Signal_to_noise": 10,
            "remove_background_nucleus": False,
        }

        PV.stream_masks_from_stack(
            stacks,
            model_name="cpsam",
            channels_for_cellpose=(1, 0),
            batch_fields=2,
            postprocess_settings=settings,
            object_type="cell",
        )

        from spacr.io import _normalize_img_batch
        normalization_settings = dict(settings)
        normalization_settings.update({
            "cell_channel": 0,
            "nucleus_channel": 1,
            "pathogen_channel": None,
            "organelle_channel": None,
        })
        expected = _normalize_img_batch(
            np.stack(raw).copy(),
            channels=range(2),
            save_dtype=np.float32,
            settings=normalization_settings,
        )
        assert np.array_equal(np.asarray(received), expected)

    def test_v1_normalization_preserves_heterogeneous_field_shapes(
            self, tmp_path, _mock_cp):
        import tifffile

        plate = tmp_path / "plate1"
        plate.mkdir()
        rng = np.random.default_rng(4)
        for field, size in ((1, 8), (2, 12)):
            for channel in range(2):
                image = rng.integers(
                    1, 2000, size=(size, size), dtype=np.uint16)
                tifffile.imwrite(
                    plate / (
                        f"plate1_A01_T01F0{field}L01A01Z01C0{channel}.tif"
                    ),
                    image,
                )

        mapper = PV.FilenameMapper.discover(
            plate, metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(
            plate, mapper, channels=(0, 1))
        settings = {
            "lower_percentile": 2,
            "background": 100,
            "Signal_to_noise": 10,
            "remove_background": False,
            "cell_background": 100,
            "cell_Signal_to_noise": 10,
            "remove_background_cell": False,
            "nucleus_background": 100,
            "nucleus_Signal_to_noise": 10,
            "remove_background_nucleus": False,
        }

        PV.stream_masks_from_stack(
            stacks,
            model_name="cpsam",
            channels_for_cellpose=(1, 0),
            batch_fields=2,
            postprocess_settings=settings,
            object_type="cell",
        )

        assert [np.load(stack.path).shape for stack in stacks] == [
            (8, 8, 3),
            (12, 12, 3),
        ]


class TestResolveRegex:
    def test_custom_without_regex_raises(self, tmp_path):
        with pytest.raises(ValueError):
            PV._resolve_regex("custom", [Path("x.tif")], None)

    def test_custom_regex_used(self):
        pat, name = PV._resolve_regex("custom", [Path("x.tif")], r".*")
        assert name == "custom" and pat == r".*"

    def test_yokogawa_branch(self, tmp_path):
        p = _make_plate(tmp_path)
        files = list(p.glob("*.tif"))
        pat, name = PV._resolve_regex("yokogawa", files, None)
        assert name == "yokogawa"

    def test_auto_best_fit_fallback(self, tmp_path):
        # Files that match NONE of the patterns → best-fit fallback
        # (lines 283-291).
        files = [Path("garbage_name_1.tif"), Path("garbage_2.tif")]
        pat, name = PV._resolve_regex("auto", files, None)
        assert name in ("cellvoyager", "yokogawa")


class TestMapperEdges:
    def test_discover_skips_nonmatching_file(self, tmp_path, caplog):
        p = _make_plate(tmp_path)
        # Drop in a file that won't match the cellvoyager regex.
        (p / "not_a_valid_name.tif").write_bytes(b"x")
        mapper = PV.FilenameMapper.discover(p, metadata_type="cellvoyager")
        # It still discovers the valid files.
        assert mapper.field_ids()


class TestStreamMasksEdges:
    def test_single_channel_stack(self, tmp_path, _mock_cp):
        # A 1-channel stack triggers the arr.squeeze() branch (line 552).
        plate = _make_plate(tmp_path, channels=1)
        mapper = PV.FilenameMapper.discover(plate,
                                              metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(plate, mapper, channels=(0,))
        PV.stream_masks_from_stack(stacks, model_name="cyto", batch_fields=1)

    def test_mask_returned_as_list(self, tmp_path, monkeypatch):
        # model.eval returns masks as a list → m = m[0] (line 557).
        class _ListModel:
            def __init__(self, *a, **k): self.pretrained_model = None
            def eval(self, img, diameter=None, **k):
                images = img if isinstance(img, list) else [img]
                return [
                    np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
                    for image in images
                ], None, None
        monkeypatch.setattr("cellpose.models.CellposeModel", _ListModel)
        plate = _make_plate(tmp_path, channels=2)
        mapper = PV.FilenameMapper.discover(plate,
                                              metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(plate, mapper, channels=(0, 1))
        PV.stream_masks_from_stack(stacks, model_name="cyto", batch_fields=1)

    def test_sidecar_and_cleanup_exceptions(self, tmp_path, _mock_cp,
                                              monkeypatch):
        # Make unlink + rmtree + sidecar writes raise → except branches
        # (573-581, 590-591) all swallowed.
        plate = _make_plate(tmp_path, channels=2)
        mapper = PV.FilenameMapper.discover(plate,
                                              metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(plate, mapper, channels=(0, 1))
        monkeypatch.setattr(Path, "unlink",
                            lambda self, *a, **k: (_ for _ in ()).throw(OSError()))
        import shutil
        monkeypatch.setattr(shutil, "rmtree",
                            lambda *a, **k: (_ for _ in ()).throw(OSError()))
        PV.stream_masks_from_stack(stacks, model_name="cyto", batch_fields=1)
