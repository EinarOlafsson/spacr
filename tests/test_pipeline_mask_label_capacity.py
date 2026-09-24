"""Mask I/O and pipeline saves must never wrap or truncate object identities."""

import numpy as np
import pytest
import tifffile

from spacr.mask_io import load_mask, save_mask
from tests.test_cellpose4_model_story import _mask_settings, _write_npz
from tests.test_cellpose4_model_story import sam_pipeline as sam_pipeline
from tests.test_cov_object_organelle_sam import _base_settings


@pytest.mark.parametrize("fmt", ["npy", "tif"])
@pytest.mark.parametrize("bad", [
    np.array([[0, 65536]], dtype=np.uint32),
    np.array([[0, 2**63 + 7]], dtype=np.uint64),
    np.array([[0, -1]], dtype=np.int32),
    np.array([[0, 1.5]], dtype=np.float32),
    np.array([[0, np.nan]], dtype=np.float32),
    np.array([[0, np.inf]], dtype=np.float32),
])
def test_invalid_mask_save_preserves_existing_file_and_input(tmp_path, fmt, bad):
    target = tmp_path / f"mask.{fmt}"
    save_mask(target, np.array([[0, 7]], dtype=np.uint16))
    before = target.read_bytes()
    input_before = bad.tobytes()
    with pytest.raises(ValueError):
        save_mask(target, bad)
    assert target.read_bytes() == before
    assert bad.tobytes() == input_before
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("fmt", ["npy", "tif"])
@pytest.mark.parametrize("bad", [
    np.array([[0, 65536]], dtype=np.uint32),
    np.array([[0, -1]], dtype=np.int32),
    np.array([[0, 1.5]], dtype=np.float32),
])
def test_invalid_external_mask_is_refused_on_read(tmp_path, fmt, bad):
    target = tmp_path / f"external.{fmt}"
    if fmt == "npy":
        np.save(target, bad)
    else:
        tifffile.imwrite(target, bad)
    before = target.read_bytes()
    with pytest.raises(ValueError):
        load_mask(target)
    assert target.read_bytes() == before


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3)])
def test_capacity_boundary_and_sparse_ids_roundtrip_without_renumbering(tmp_path, shape):
    labels = np.resize(np.array([0, 7, 65535], dtype=np.int64), shape)
    for fmt in ("npy", "tif"):
        output = save_mask(tmp_path / f"valid.{fmt}", labels)
        actual = load_mask(output)
        assert actual.dtype == np.uint16
        np.testing.assert_array_equal(actual, labels)


@pytest.mark.parametrize("backend", ["sam", "legacy", "organelle"])
def test_pipeline_refuses_bad_second_mask_before_replacing_any_batch_file(
        tmp_path, monkeypatch, sam_pipeline, backend):
    import spacr.object as objects
    import spacr.utils as utils

    src = tmp_path / "stack"
    _write_npz(src)
    good = np.zeros((32, 32), dtype=np.uint32)
    good[2:8, 2:8] = 7
    bad = good.copy()
    bad[2:8, 2:8] = 65536
    masks = [good, bad]
    monkeypatch.setattr(objects, "merge_split_filter_masks", lambda *a, **k: masks)
    monkeypatch.setattr(utils, "_masks_to_masks_stack", lambda *a, **k: masks)
    monkeypatch.setattr(utils, "_choose_model", lambda *a, **k: objects.cp_models.CellposeModel())
    role = "organelle" if backend == "organelle" else "cell"
    folder = src / f"{role}_mask_stack"
    folder.mkdir()
    targets = [folder / f"plate1_A01_{index}.npy" for index in (1, 2)]
    for target in targets:
        target.write_bytes(b"damaged output queued for regeneration")
    before = [target.read_bytes() for target in targets]
    settings = _mask_settings(src, filter=False, cell_merge=False, seg_qc="off")
    if backend == "organelle":
        settings = _base_settings(seg_qc="off")
        generate = objects.generate_organelle_masks_sam
    elif backend == "legacy":
        generate = objects.generate_cellpose_masks
    else:
        generate = objects.generate_cellpose_masks_sam
    with pytest.raises(ValueError, match="uint16"):
        generate(str(src), settings, role)
    assert [target.read_bytes() for target in targets] == before
    assert set(folder.iterdir()) == set(targets)
