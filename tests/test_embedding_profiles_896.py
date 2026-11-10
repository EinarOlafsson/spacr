"""Item 896: raw crop shards, embeddings, well embedding profiles and the well UMAP."""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import embedding_profiles as ep

h5py = pytest.importorskip("h5py")


def _field(seed: int) -> np.ndarray:
    """A 64 x 64 merged field: four intensity planes, then cell (4) and nucleus (5) labels."""
    rng = np.random.default_rng(seed)
    array = np.zeros((64, 64, 6), dtype=np.float32)
    array[..., :4] = rng.integers(100, 1000, size=(64, 64, 4))
    array[10:20, 10:20, 4] = 1
    array[12:16, 12:16, 5] = 1
    array[30:40, 30:42, 4] = 2
    array[33:36, 33:36, 5] = 2
    array[0:6, 50:64, 4] = 3
    return array


def _plate(tmp_path: Path, plate: str = "Wk_1") -> Path:
    merged = tmp_path / "src" / "merged"
    merged.mkdir(parents=True)
    key = plate.replace("_", "%5F")
    for number, (well, site) in enumerate([("B02", 1), ("B02", 2), ("C03", 1)]):
        np.save(merged / f"{key}_{well}_{site}_1.npy", _field(number))
    return merged


def _cells(plate: str = "Wk_1") -> pd.DataFrame:
    rows = [{"plate": plate, "well": w, "site": s, "Metadata_cell_id": c}
            for w, s in [("B02", 1), ("B02", 2), ("C03", 1)] for c in (1, 2, 3)]
    frame = pd.DataFrame(rows)
    frame["moa"] = np.where(frame["well"] == "B02", "DMSO", "taxol-like")
    return frame


def _encoder(batch: np.ndarray) -> np.ndarray:
    return np.stack([batch.mean(axis=(1, 2, 3)), batch.max(axis=(1, 2, 3))], axis=1)


def test_the_draw_is_capped_per_well_and_repeatable():
    cells = _cells()
    one = ep.select_crop_cells(cells, cap_per_well=2, seed=4)
    two = ep.select_crop_cells(cells, cap_per_well=2, seed=4)
    pd.testing.assert_frame_equal(one, two)
    assert one.groupby("well").size().tolist() == [2, 2]
    assert one["n_cells_in_well"].tolist() == [6, 6, 3, 3]
    assert one.equals(one.sort_values(["well", "site", "Metadata_cell_id"]).reset_index(drop=True))
    assert len(ep.select_crop_cells(cells, cap_per_well=0)) == 9
    with pytest.raises(ValueError, match="lacks"):
        ep.select_crop_cells(cells.drop(columns="site"))


def test_a_shard_holds_raw_windows_and_bit_masks(tmp_path):
    merged = _plate(tmp_path)
    info = ep.write_crop_shard(merged, ep.select_crop_cells(_cells(), cap_per_well=0),
                               tmp_path / "crops", plate="Wk_1", size=16)
    assert info["n_crops"] == 9 and info["n_wells"] == 2 and info["clipped_px"] == 0
    with h5py.File(info["shard"]) as handle:
        images, masks = handle["images"][:], handle["masks"][:]
        assert json.loads(handle.attrs["mask_bits"])["8"] == "outside the field"
    assert images.dtype == np.uint16 and images.shape == (9, 16, 16, 3)
    field = _field(0)
    np.testing.assert_array_equal(images[0], field[6:22, 6:22, :3].astype(np.uint16))
    assert (masks[0] & 1).sum() == 100 and ((masks[0] & 2) > 0).sum() == 16
    edge = masks[2]
    assert ((edge & 8) > 0).any() and not ((masks[0] & 8) > 0).any()
    index = pd.read_parquet(info["cells"])
    assert index["row"].tolist() == list(range(9))
    assert index["truncated"].tolist()[:2] == [False, False]
    only = ep.write_crop_shard(merged, ep.select_crop_cells(_cells(), cap_per_well=0),
                               tmp_path / "only", plate="Wk_1", size=8, only_wells=["C03"],
                               mask_flags=False)
    with h5py.File(only["shard"]) as handle:
        assert "masks" not in handle and handle["images"].shape[0] == 3
    assert pd.read_parquet(only["cells"])["truncated"].all()


def test_the_flat_field_divides_the_window(tmp_path, monkeypatch):
    import spacr.illumination as illumination

    merged = _plate(tmp_path)
    stub = types.SimpleNamespace(
        gain_stack=lambda channels: np.full((64, 64, len(channels)), 2.0, np.float32),
        _offset_stack=lambda channels: np.full((64, 64, len(channels)), 100.0, np.float32))
    seen = {}

    def load(path):
        seen["path"] = path
        return types.SimpleNamespace(field_for=lambda key: seen.setdefault("key", key) and stub)

    monkeypatch.setattr(illumination, "load_illumination_model", load)
    cells = ep.select_crop_cells(_cells(), cap_per_well=0).iloc[:1]
    info = ep.write_crop_shard(merged, cells, tmp_path / "c", plate="Wk_1", size=16,
                               illumination="model.npz")
    assert seen == {"path": "model.npz", "key": "Wk%5F1"}
    with h5py.File(info["shard"]) as handle:
        image = handle["images"][0]
    expected = np.clip(np.rint((_field(0)[6:22, 6:22, :3] - 100.0) * 2.0), 0, 65535)
    np.testing.assert_array_equal(image, expected.astype(np.uint16))


def test_embeddings_aggregate_to_wells_and_percentiles(tmp_path):
    merged = _plate(tmp_path)
    shard = ep.write_crop_shard(merged, ep.select_crop_cells(_cells(), cap_per_well=0),
                                tmp_path / "crops", plate="Wk_1", size=16)
    spec = ep.embedding_spec({"crop_channels": "0,1,2"})
    assert spec.input_transform == "none" and spec.normalize and spec.channels == (0, 1, 2)
    info = ep.embed_crop_shard(shard["shard"], spec, tmp_path / "emb", encoder=_encoder, chunk=4)
    assert info["n"] == 9 and info["dims"] == 6 and len(info["channel_scale"]) == 3
    values = np.load(info["path"])
    crops = ep.ShardCrops(shard["shard"])
    assert len(crops) == 9 and crops[np.array([0, 2])].shape == (2, 16, 16, 3)
    assert (crops[0:1][0][..., 0] == 0).sum() == 16 * 16 - 100
    crops.close()
    table = ep.embedding_cells(values, pd.read_parquet(shard["cells"]), keep=("plate", "well", "moa"))
    result = ep.aggregate_embedding_wells([table], tmp_path / "tables")
    assert result["n_cells"] == 9 and result["n_features"] == 6
    median = pd.read_parquet(tmp_path / "tables" / "wells_median.parquet")
    assert median["n_cells"].tolist() == [6, 3]
    assert median["emb_0000"].iloc[0] == pytest.approx(table[table.well == "B02"]["emb_0000"].median())
    limits = pd.read_parquet(tmp_path / "tables" / "dmso_percentiles.parquet")
    assert limits["q"].tolist() == [0.01, 0.99] and "moa" not in limits
    with pytest.raises(ValueError, match="embeddings for"):
        ep.embedding_cells(values[:2], table)
    with pytest.raises(ValueError, match="input_normalisation"):
        ep.embedding_spec({"embedding_input_normalisation": "zscore"})
    with pytest.raises(ValueError, match="channel_scale"):
        ep.embedding_spec({"embedding_channel_scale": "auto"})
    assert ep.embedding_spec({"embedding_input_normalisation": "imagenet"}).input_transform == "pretrained"
    with pytest.raises(ValueError, match="no masks"):
        ep.ShardCrops(_unmasked(tmp_path, merged))


def _unmasked(tmp_path, merged):
    info = ep.write_crop_shard(merged, ep.select_crop_cells(_cells()).iloc[:1], tmp_path / "u",
                               plate="Wk_1", size=8, mask_flags=False)
    return info["shard"]


def test_the_well_umap_is_seeded():
    pytest.importorskip("umap")
    rng = np.random.default_rng(0)
    wells = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"emb_{i:04d}" for i in range(5)])
    wells.insert(0, "well", [f"W{i}" for i in range(30)])
    one = ep.well_umap(wells, n_neighbors=5, seed=0)
    two = ep.well_umap(wells, n_neighbors=5, seed=0)
    pd.testing.assert_frame_equal(one, two)
    assert list(one.columns) == ["well", "x", "y"] and one.attrs["umap_version"]


def test_run_embedding_profiles_from_settings(tmp_path, monkeypatch):
    import spacr.embeddings as embeddings

    merged = _plate(tmp_path)
    monkeypatch.setattr(embeddings, "_backbone_encoder", lambda spec: _encoder)
    mapped = {}

    def fake_umap(wells, *, features, **options):
        mapped.update(options, features=list(features), wells=wells)
        return wells[["plate", "well"]].assign(x=0.0, y=1.0)

    monkeypatch.setattr(ep, "well_umap", fake_umap)
    wells = pd.DataFrame({"plate": ["Wk_1", "Wk_1"], "well": ["B02", "C03"],
                          "compound": ["DMSO", "taxol"]})
    result = ep.run_embedding_profiles({
        "embedding_src": str(merged.parent), "embedding_cells": _cells(), "crop_window": 16,
        "crop_cap_per_well": 4, "embedding_wells": wells, "embedding_umap": True,
        "umap_neighbors": 2})
    assert result["shard"]["n_crops"] == 7 and result["shard"]["n_wells"] == 2
    assert result["embedding"]["dims"] == 6
    out = merged.parent / "embeddings"
    assert sorted(p.name for p in (out / "tables").glob("*.parquet")) == [
        "dmso_percentiles.parquet", "wells_mean.parquet", "wells_median.parquet"]
    assert mapped["n_neighbors"] == 2 and mapped["metric"] == "cosine" and mapped["seed"] == 0
    assert np.isfinite(mapped["wells"][mapped["features"]].to_numpy()).all()
    assert pd.read_csv(result["umap"]["path"])["y"].tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="needed"):
        ep.run_embedding_profiles({})
    with pytest.raises(ValueError, match="crop_store"):
        ep.run_embedding_profiles({"embedding_src": "x", "embedding_cells": _cells(),
                                   "crop_store": "zarr"})
    with pytest.raises(ValueError, match="crop_illumination"):
        ep.run_embedding_profiles({"embedding_src": "x", "embedding_cells": _cells(),
                                   "crop_illumination": "flat"})


def test_the_module_is_registered_and_alpha():
    from spacr import cli
    from spacr.settings import ALPHA_FEATURES

    module = cli.resolve_module("embedding_profiles")
    assert module is not None and module.entry == "spacr.embedding_profiles:run_embedding_profiles"
    assert cli.module_defaults(module)["crop_cap_per_well"] == 300
    assert set(ALPHA_FEATURES[896]["module_settings"]["embedding_profiles"]) == set(ep._DEFAULTS)
