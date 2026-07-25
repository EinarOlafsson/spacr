"""CPU coverage for the spacr.utils "results + augmentation" block:

``merge_regression_res_with_metadata``, ``process_vision_results``,
``get_ml_results_paths``, ``augment_image`` and ``augment_dataset``.

Everything here is offline, CPU-only and works on tiny synthetic arrays so the
whole file runs in well under a second.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_open_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# merge_regression_res_with_metadata
# ---------------------------------------------------------------------------

def _write_regression_inputs(tmp_path):
    """Two small CSVs exercising every branch of the gene-parsing helpers."""
    results = tmp_path / "regression_results.csv"
    metadata = tmp_path / "gene_metadata.csv"
    pd.DataFrame(
        {
            # 'T.' prefix + '_' suffix -> ABC123
            # no 'T.' prefix, '_' suffix   -> DEF456
            # bracket but nothing to strip -> GHI789
            # no bracket at all            -> None (return None branch)
            "feature": [
                "C(gene)[T.ABC123_2]",
                "C(gene)[DEF456_11]",
                "C(gene)[T.GHI789]",
                "Intercept",
            ],
            "coefficient": [1.5, -0.5, 0.25, 10.0],
        }
    ).to_csv(results, index=False)
    pd.DataFrame(
        {
            # every Gene ID carries an underscore so the parsed gene column has
            # no NaNs -- pandas would otherwise join NaN keys to each other.
            "Gene ID": ["TGME49_ABC123", "TGME49_DEF456", "TGME49_ZZZ000"],
            "Description": ["kinase", "transporter", "unrelated"],
        }
    ).to_csv(metadata, index=False)
    return results, metadata


def test_merge_regression_res_with_metadata_parses_and_merges(tmp_path):
    from spacr.utils import merge_regression_res_with_metadata

    results, metadata = _write_regression_inputs(tmp_path)
    merged = merge_regression_res_with_metadata(str(results), str(metadata))

    # left join -> one row per regression feature, nothing dropped/duplicated
    assert len(merged) == 4
    assert list(merged["gene"])[:3] == ["ABC123", "DEF456", "GHI789"]
    assert merged["gene"].isna().tolist() == [False, False, False, True]

    # metadata joined only where the gene actually matched
    assert merged.loc[0, "Description"] == "kinase"
    assert merged.loc[1, "Description"] == "transporter"
    assert pd.isna(merged.loc[2, "Description"])   # GHI789 absent from metadata
    assert pd.isna(merged.loc[3, "Description"])   # Intercept has no gene

    # the unrelated metadata row must not leak into the result
    assert "unrelated" not in set(merged["Description"].dropna())
    # original columns survive the merge
    assert merged.loc[3, "coefficient"] == 10.0


def test_merge_regression_res_with_metadata_writes_suffixed_csv(tmp_path):
    from spacr.utils import merge_regression_res_with_metadata

    results, metadata = _write_regression_inputs(tmp_path)
    merged = merge_regression_res_with_metadata(
        str(results), str(metadata), name="_annotated"
    )

    expected = tmp_path / "regression_results_annotated.csv"
    assert expected.is_file(), os.listdir(tmp_path)
    # default '_metadata' name was NOT used
    assert not (tmp_path / "regression_results_metadata.csv").exists()

    roundtrip = pd.read_csv(expected)
    assert list(roundtrip.columns) == list(merged.columns)
    assert len(roundtrip) == len(merged)
    assert roundtrip.loc[0, "Description"] == "kinase"
    # written without an index column
    assert "Unnamed: 0" not in roundtrip.columns


def test_merge_regression_res_with_metadata_no_matches(tmp_path):
    """Disjoint gene sets still produce one row per feature, all metadata NaN."""
    from spacr.utils import merge_regression_res_with_metadata

    results = tmp_path / "res.csv"
    metadata = tmp_path / "meta.csv"
    pd.DataFrame({"feature": ["C(gene)[T.AAA_1]"], "coefficient": [0.1]}).to_csv(
        results, index=False
    )
    pd.DataFrame(
        {"Gene ID": ["TGME49_BBB"], "Description": ["nope"]}
    ).to_csv(metadata, index=False)

    merged = merge_regression_res_with_metadata(str(results), str(metadata))
    assert len(merged) == 1
    assert merged.loc[0, "gene"] == "AAA"
    assert merged["Description"].isna().all()


def test_merge_regression_res_with_metadata_does_not_join_on_missing_gene(tmp_path):
    from spacr.utils import merge_regression_res_with_metadata

    results = tmp_path / "res.csv"
    metadata = tmp_path / "meta.csv"
    pd.DataFrame(
        {"feature": ["Intercept", "C(gene)[T.AAA_1]"], "coefficient": [1.0, 2.0]}
    ).to_csv(results, index=False)
    pd.DataFrame(
        {"Gene ID": ["NOUNDERSCORE", "TGME49_AAA"],
         "Description": ["SPURIOUS", "real"]}
    ).to_csv(metadata, index=False)

    merged = merge_regression_res_with_metadata(str(results), str(metadata))
    assert merged.loc[1, "Description"] == "real"
    # 'Intercept' has no gene at all -> it must not inherit anyone's metadata
    assert pd.isna(merged.loc[0, "Description"])


# ---------------------------------------------------------------------------
# process_vision_results
# ---------------------------------------------------------------------------

def test_process_vision_results_maps_wells_and_thresholds():
    from spacr.utils import process_vision_results

    df = pd.DataFrame(
        {
            "path": [
                "plate1_A01_3_7.png",
                "plate1_B12_10_2.png",
                "plate2_C03_1_5.png",
            ],
            "pred": [0.20, 0.50, 0.99],
        }
    )
    out = process_vision_results(df, threshold=0.5)

    assert list(out["plateID"]) == ["plate1", "plate1", "plate2"]
    assert list(out["rowID"]) == ["r1", "r2", "r3"]
    assert list(out["columnID"]) == ["c1", "c12", "c3"]
    assert list(out["fieldID"]) == ["f3", "f10", "f1"]
    assert list(out["object"]) == ["7", "2", "5"]
    assert list(out["prc"]) == ["plate1_r1_c1", "plate1_r2_c12", "plate2_r3_c3"]
    # threshold is inclusive (>=), so 0.50 is a positive
    assert list(out["cv_predictions"]) == [0, 1, 1]
    assert out["cv_predictions"].dtype.kind == "i"
    # mutates and returns the same object
    assert out is df


def test_process_vision_results_threshold_is_respected():
    from spacr.utils import process_vision_results

    df = pd.DataFrame(
        {"path": ["p_A01_1_1.png", "p_A02_1_1.png"], "pred": [0.6, 0.95]}
    )
    out = process_vision_results(df, threshold=0.9)
    assert list(out["cv_predictions"]) == [0, 1]


def test_process_vision_results_numeric_well_falls_back_to_raw_well():
    """A non-alphabetic well id is copied verbatim into rowID and columnID."""
    from spacr.utils import process_vision_results

    df = pd.DataFrame({"path": ["plateX_12_4_9.png"], "pred": [1.0]})
    out = process_vision_results(df, threshold=0.5)
    assert out.loc[0, "rowID"] == "12"
    assert out.loc[0, "columnID"] == "12"
    assert out.loc[0, "prc"] == "plateX_12_12"
    assert out.loc[0, "cv_predictions"] == 1


# ---------------------------------------------------------------------------
# get_ml_results_paths
# ---------------------------------------------------------------------------

_EXPECTED_BASENAMES = [
    "results.csv",
    "permutation.csv",
    "feature_importance.csv",
    None,                      # depends on model_type
    "permutation.pdf",
    "feature_importance.pdf",
    "shap.pdf",
    "plate_heatmap.pdf",
    "ml_settings.csv",
    "ml_features.csv",
]


def _check_paths(paths, src, model_type, feature_string):
    assert isinstance(paths, tuple) and len(paths) == 10
    res_fldr = os.path.join(src, "results", model_type, feature_string)
    assert os.path.isdir(res_fldr)
    for path, expected in zip(paths, _EXPECTED_BASENAMES):
        assert os.path.dirname(path) == res_fldr
        if expected is not None:
            assert os.path.basename(path) == expected
    assert os.path.basename(paths[3]) == f"{model_type}_model.csv"
    # all ten paths are distinct
    assert len(set(paths)) == 10


def test_get_ml_results_paths_int_channel(tmp_path):
    from spacr.utils import get_ml_results_paths

    paths = get_ml_results_paths(str(tmp_path), model_type="xgboost",
                                 channel_of_interest=2)
    _check_paths(paths, str(tmp_path), "xgboost", "channel_2")


def test_get_ml_results_paths_list_channels(tmp_path):
    from spacr.utils import get_ml_results_paths

    paths = get_ml_results_paths(str(tmp_path), model_type="random_forest",
                                 channel_of_interest=[0, 1, 3])
    _check_paths(paths, str(tmp_path), "random_forest", "channels_0_1_3")


def test_get_ml_results_paths_morphology(tmp_path):
    from spacr.utils import get_ml_results_paths

    paths = get_ml_results_paths(str(tmp_path), model_type="logistic",
                                 channel_of_interest="morphology")
    _check_paths(paths, str(tmp_path), "logistic", "morphology")


def test_get_ml_results_paths_none_means_all_features(tmp_path):
    from spacr.utils import get_ml_results_paths

    paths = get_ml_results_paths(str(tmp_path), model_type="xgboost",
                                 channel_of_interest=None)
    _check_paths(paths, str(tmp_path), "xgboost", "all_features")


def test_get_ml_results_paths_is_idempotent(tmp_path):
    """Calling twice must not raise on the already-existing folder."""
    from spacr.utils import get_ml_results_paths

    first = get_ml_results_paths(str(tmp_path), channel_of_interest=1)
    second = get_ml_results_paths(str(tmp_path), channel_of_interest=1)
    assert first == second
    assert os.path.isdir(os.path.dirname(first[0]))


@pytest.mark.parametrize("bad", [2.5, "nucleus", (1, 2), {"a": 1}])
def test_get_ml_results_paths_rejects_unsupported_channel(tmp_path, bad):
    from spacr.utils import get_ml_results_paths

    with pytest.raises(ValueError, match="Unsupported channel_of_interest"):
        get_ml_results_paths(str(tmp_path), channel_of_interest=bad)
    # nothing was created for the rejected input
    assert not os.path.exists(os.path.join(str(tmp_path), "results"))


# ---------------------------------------------------------------------------
# augment_image
# ---------------------------------------------------------------------------

def _as_arrays(images):
    from PIL import Image

    assert all(isinstance(im, Image.Image) for im in images)
    return [np.asarray(im) for im in images]


def test_augment_image_grayscale_array_expands_to_rgb():
    from spacr.utils import augment_image

    gray = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    out = augment_image(gray)
    assert len(out) == 8

    arrs = _as_arrays(out)
    # 2D input was promoted to 3 identical channels
    for arr in arrs:
        assert arr.ndim == 3 and arr.shape[2] == 3
        assert np.array_equal(arr[..., 0], arr[..., 1])
        assert np.array_equal(arr[..., 1], arr[..., 2])

    # order: original, hflip, rot90cw, rot90cw+hflip, rot180, ..., rot90ccw, ...
    ch = [a[..., 0] for a in arrs]
    assert np.array_equal(ch[0], gray)
    assert np.array_equal(ch[1], gray[:, ::-1])
    assert np.array_equal(ch[2], np.rot90(gray, -1))
    assert np.array_equal(ch[3], np.rot90(gray, -1)[:, ::-1])
    assert np.array_equal(ch[4], np.rot90(gray, 2))
    assert np.array_equal(ch[5], np.rot90(gray, 2)[:, ::-1])
    assert np.array_equal(ch[6], np.rot90(gray, 1))
    assert np.array_equal(ch[7], np.rot90(gray, 1)[:, ::-1])


def test_augment_image_rotations_transpose_non_square_shape():
    from spacr.utils import augment_image

    gray = np.zeros((3, 5), dtype=np.uint8)
    out = augment_image(gray)
    # PIL .size is (width, height)
    sizes = [im.size for im in out]
    assert sizes[0] == (5, 3) and sizes[1] == (5, 3)     # original / flip
    assert sizes[2] == (3, 5) and sizes[3] == (3, 5)     # 90 cw
    assert sizes[4] == (5, 3) and sizes[5] == (5, 3)     # 180
    assert sizes[6] == (3, 5) and sizes[7] == (3, 5)     # 90 ccw


def test_augment_image_accepts_pil_input():
    from PIL import Image
    from spacr.utils import augment_image

    gray = (np.arange(16, dtype=np.uint8).reshape(4, 4) * 15)
    pil = Image.fromarray(gray, mode="L")
    out = augment_image(pil)
    assert len(out) == 8
    arrs = _as_arrays(out)
    assert arrs[0].shape == (4, 4, 3)
    assert np.array_equal(arrs[0][..., 0], gray)
    assert np.array_equal(arrs[4][..., 0], np.rot90(gray, 2))


def test_augment_image_colour_array_keeps_three_channels():
    from spacr.utils import augment_image

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(4, 6, 3), dtype=np.uint8)
    out = augment_image(rgb)
    arrs = _as_arrays(out)
    assert len(arrs) == 8
    assert arrs[0].shape == (4, 6, 3)
    # untouched original, per-channel
    assert np.array_equal(arrs[0], rgb)
    assert np.array_equal(arrs[1], rgb[:, ::-1, :])
    assert np.array_equal(arrs[6], np.rot90(rgb, 1, axes=(0, 1)))
    # the 8 outputs are genuinely different transforms of the same pixels
    assert sorted(a.sum() for a in arrs) == [rgb.sum()] * 8


def test_augment_image_pil_rgb_input_is_not_re_expanded():
    from PIL import Image
    from spacr.utils import augment_image

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(5, 5, 3), dtype=np.uint8)
    out = augment_image(Image.fromarray(rgb, mode="RGB"))
    arrs = _as_arrays(out)
    assert arrs[0].shape == (5, 5, 3)
    assert np.array_equal(arrs[0], rgb)


# ---------------------------------------------------------------------------
# augment_dataset
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_augment_dataset_expands_eightfold_and_keeps_metadata():
    from spacr.utils import augment_dataset

    a = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    b = torch.zeros(3, 4, 4)
    dataset = [(a, 0, "a.png"), (b, 1, "b.png")]

    out = augment_dataset(dataset)
    assert len(out) == 16
    assert [lbl for _, lbl, _ in out] == [0] * 8 + [1] * 8
    assert [fn for _, _, fn in out] == ["a.png"] * 8 + ["b.png"] * 8
    for img, _, _ in out:
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 4, 4)
        assert img.dtype == torch.float32


def test_augment_dataset_applies_the_expected_transforms():
    from spacr.utils import augment_dataset

    img = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    out = augment_dataset([(img, 7, "x.png")])
    tensors = [t for t, _, _ in out]
    assert len(tensors) == 8

    # angle 0 is the identity, then hflip, then successive CCW rotations
    assert torch.equal(tensors[0], img)
    assert torch.equal(tensors[1], img.flip(-1))
    for k, idx in zip((1, 2, 3), (2, 4, 6)):
        rot = torch.rot90(img, k, dims=(-2, -1))
        assert torch.equal(tensors[idx], rot), f"rotation k={k}"
        assert torch.equal(tensors[idx + 1], rot.flip(-1)), f"flip of k={k}"

    # rotations permute pixels, they do not invent or destroy them
    for t in tensors:
        assert sorted(t.flatten().tolist()) == sorted(img.flatten().tolist())


def test_augment_dataset_empty_input_returns_empty_list():
    from spacr.utils import augment_dataset

    assert augment_dataset([]) == []


def test_augment_dataset_is_grayscale_flag_does_not_change_output():
    from spacr.utils import augment_dataset

    img = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    plain = augment_dataset([(img, 0, "g.png")], is_grayscale=False)
    flagged = augment_dataset([(img, 0, "g.png")], is_grayscale=True)
    assert len(plain) == len(flagged) == 8
    for (p, pl, pf), (f, fl, ff) in zip(plain, flagged):
        assert torch.equal(p, f)
        assert (pl, pf) == (fl, ff)


def test_augment_dataset_rejects_non_tensor_images():
    from spacr.utils import augment_dataset

    bad = np.zeros((1, 4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        augment_dataset([(bad, 0, "bad.png")])


def test_augment_dataset_rejects_non_tensor_after_valid_entries():
    """The type guard fires mid-iteration, not only on the first sample."""
    from spacr.utils import augment_dataset

    good = torch.zeros(1, 3, 3)
    with pytest.raises(TypeError):
        augment_dataset([(good, 0, "ok.png"), ([[0, 1], [2, 3]], 1, "bad.png")])
