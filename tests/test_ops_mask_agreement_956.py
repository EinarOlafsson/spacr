"""Item 956: split and merge counts, the identity control and the displaced
control in the large-site mask agreement, and the score-only run on saved
labels in the 04Align frame."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_large_sites as ols
from tests.test_ops_large_sites_921 import _models, _site, _stand_in_masks, _write_well  # noqa: F401

KEYS = ("ops_mask_agreement_split_merge", "ops_mask_agreement_identity_control",
        "ops_mask_agreement_displace_px", "ops_mask_agreement_labels",
        "ops_mask_agreement_frame")


def _box(labels, value, y0, y1, x0, x1):
    labels[y0:y1, x0:x1] = value


def _reference():
    """Three square nuclei away from the border."""
    ref = np.zeros((120, 120), np.int32)
    _box(ref, 1, 10, 40, 10, 40)
    _box(ref, 2, 10, 40, 60, 90)
    _box(ref, 3, 70, 100, 10, 40)
    return ref


def test_a_known_split_and_a_known_merge_are_counted():
    ref = _reference()
    mine = np.zeros_like(ref)
    _box(mine, 1, 10, 40, 10, 25)
    _box(mine, 2, 10, 40, 25, 40)
    _box(mine, 3, 10, 40, 60, 90)
    assert ols._split_merge(mine, ref) == {"splits": 1, "merges": 0}
    merged = np.zeros_like(ref)
    _box(merged, 1, 8, 102, 8, 42)
    _box(merged, 2, 10, 40, 60, 90)
    assert ols._split_merge(merged, ref) == {"splits": 0, "merges": 1}
    assert ols._split_merge(ref, ref.copy()) == {"splits": 0, "merges": 0}


def test_objects_on_the_border_are_dropped_before_the_assignment():
    ref = _reference()
    ref[0:5, 50:110] = 9
    mine = ref.copy()
    mine[0:5, 50:80], mine[0:5, 80:110] = 9, 10
    assert ols._split_merge(mine, ref) == {"splits": 0, "merges": 0}


def test_the_shift_and_overlap_cut_both_frames_alike():
    image = np.arange(1, 26, dtype=np.int32).reshape(5, 5)
    moved = ols._shift_labels(image, -1, 2)
    assert moved[0, 2] == image[1, 0] and moved[4].sum() == 0 and moved[:, :2].sum() == 0
    assert ols._frame_overlap(image, -1, 2).shape == (3, 1)
    assert np.array_equal(ols._shift_labels(image, 0, 0), image)


def _well(tmp_path, mine, ref, site="5", shift=None):
    """Saved labels and a PERISCOPE-layout reference with an optional Image.csv."""
    import tifffile

    labels = tmp_path / "seg" / "labels"
    labels.mkdir(parents=True)
    np.save(labels / f"{site}_nuclei.npy", mine)
    masks = tmp_path / "ref" / f"CP-Well1-{site}" / "segmentation_masks"
    masks.mkdir(parents=True)
    tifffile.imwrite(masks / f"Plate_CP_Well_Well1_Site_{site}_Nuclei_Objects.tiff", ref)
    if shift is not None:
        pd.DataFrame({"Align_Yshift_DAPI_Painting": [shift[0]],
                      "Align_Xshift_DAPI_Painting": [shift[1]]}).to_csv(
            masks.parent / "Image.csv", index=False)
    return tmp_path / "seg", tmp_path / "ref"


def _score(tmp_path, seg, ref, **extra):
    settings = {"seg_dst": str(tmp_path / "out"), "seg_report_reference": str(ref),
                "ops_mask_agreement_labels": str(seg), "ops_mask_agreement_split_merge": True,
                "ops_mask_agreement_identity_control": True}
    settings.update(extra)
    return ols.run_large_site_segmentation(settings)


def test_the_score_only_run_moves_the_labels_into_the_align_frame(tmp_path):
    ref = _reference()
    mine = ols._shift_labels(ref, 7, -4)
    seg, refdir = _well(tmp_path, mine, ref, shift=(7, -4))
    out = _score(tmp_path, seg, refdir, ops_mask_agreement_frame="periscope_align")
    site = out["mask_agreement"][0]
    assert (site["dy"], site["dx"]) == (-7, 4)
    assert site["f1"] == 1.0 and site["splits"] == 0 and site["merges"] == 0
    assert site["identity_f1"] == 1.0 and site["identity_splits"] == 0
    assert site["displaced_f1"] < 0.5 and site["f1_iou0.7"] == 1.0
    same = _score(tmp_path / "b", seg, refdir)
    assert same["mask_agreement"][0]["f1"] < 1.0
    table = pd.read_csv(tmp_path / "out" / "mask_agreement.csv")
    assert list(table["site"].astype(str)) == ["5", "pooled"]
    assert set(json.loads((tmp_path / "out" / "settings.json").read_text())) >= {
        "ops_mask_agreement_labels", "ops_mask_agreement_frame"}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["mask_agreement.csv",
                                                                     "settings.json"]


def test_the_pooled_row_sums_counts_and_bootstraps_sites(tmp_path):
    import tifffile

    ref = _reference()
    split = ref.copy()
    split[10:40, 25:40] = 4
    seg, refdir = _well(tmp_path, ref, ref, site="1")
    np.save(seg / "labels" / "2_nuclei.npy", split)
    masks = refdir / "CP-Well1-2" / "segmentation_masks"
    masks.mkdir(parents=True)
    tifffile.imwrite(masks / "Plate_CP_Well_Well1_Site_2_Nuclei_Objects.tiff", ref)
    rows = _score(tmp_path, seg, refdir)["mask_agreement"]
    one, two, pooled = rows
    assert (one["site"], two["site"], pooled["site"]) == ("1", "2", "pooled")
    assert two["splits"] == 1 and pooled["splits"] == 1
    assert pooled["n_reference"] == 6 and pooled["matched"] == one["matched"] + two["matched"]
    assert pooled["f1"] == 2 * pooled["matched"] / (pooled["n_spacr"] + pooled["n_reference"])
    assert pooled["split_rate"] == 1 / 6
    frame = pd.DataFrame(rows[:2])
    rng = np.random.default_rng(0)
    draws = [frame.iloc[rng.integers(0, 2, 2)] for _ in range(ols.MASK_AGREEMENT_BOOTSTRAP)]
    rates = [d["splits"].sum() / d["n_reference"].sum() for d in draws]
    assert pooled["split_rate_ci_low"] == float(np.quantile(rates, 0.025))
    assert pooled["split_rate_ci_high"] == float(np.quantile(rates, 0.975))
    assert pooled["identity_f1"] == 1.0 and pooled["identity_merges"] == 0


def test_the_settings_are_checked_and_listed_as_alpha():
    from spacr.settings import ALPHA_FEATURES
    from spacr.validate import _APP_EXTRA_KEYS

    assert set(KEYS) == set(ALPHA_FEATURES[956]["module_settings"]["ops_large_sites"])
    assert set(KEYS) == set(ols.MASK_AGREEMENT_DEFAULTS) <= set(ols.LARGE_SITE_TOOLTIPS)
    assert set(KEYS) == set(ols.LARGE_SITE_CATEGORIES["Mask agreement report"])
    assert set(KEYS) <= _APP_EXTRA_KEYS["ops_large_sites"]
    assert not set(KEYS) & set(ols.large_site_settings())
    for key in KEYS:
        tip = ols.LARGE_SITE_TOOLTIPS[key]
        assert tip.startswith("(") and 80 <= len(tip) <= 600 and tip.endswith("Alpha.")
    reference = {"seg_report_reference": "r"}
    with pytest.raises(ValueError, match="ops_mask_agreement_frame"):
        ols._agreement_settings(dict(reference, ops_mask_agreement_frame="sbs"))
    with pytest.raises(ValueError, match="ops_mask_agreement_displace_px"):
        ols._agreement_settings(dict(reference, ops_mask_agreement_displace_px=-1))
    with pytest.raises(ValueError, match="report"):
        ols._agreement_settings(dict(reference, ops_mask_agreement_labels="x"))
    with pytest.raises(ValueError, match="seg_report_reference"):
        ols._agreement_settings({"ops_mask_agreement_split_merge": True})
    assert not ols._agreement_settings({})["report"]


def test_periscope_align_needs_the_sites_image_table(tmp_path):
    ref = _reference()
    seg, refdir = _well(tmp_path, ref, ref)
    with pytest.raises(FileNotFoundError, match="Image.csv"):
        _score(tmp_path, seg, refdir, ops_mask_agreement_frame="periscope_align")


def test_the_default_run_writes_what_it_wrote_before(tmp_path):
    stack, _ = _site()
    channels = ols.LARGE_SITE_DEFAULTS["seg_site_channels"]
    well, ref = _write_well(tmp_path, stack, channels)
    base = {"seg_preset": "cell_painting_phalloidin", "seg_sites_src": str(well),
            "seg_tile_px": 256, "seg_tile_overlap_px": 64, "seg_report_reference": str(ref),
            "seg_measure_band_px": 200, "seg_diameter_min_objects": 3}
    plain = ols.run_large_site_segmentation(dict(base, seg_dst=str(tmp_path / "a")),
                                            models=_models(), available_bytes=2 ** 40)
    assert "mask_agreement" not in plain
    assert not (tmp_path / "a" / "mask_agreement.csv").exists()
    saved = json.loads((tmp_path / "a" / "settings.json").read_text())
    assert not set(KEYS) & set(saved)
    report = ols.run_large_site_segmentation(
        dict(base, seg_dst=str(tmp_path / "b"), ops_mask_agreement_split_merge=True),
        models=_models(), available_bytes=2 ** 40)
    for name in ("sites.csv", "agreement_well.csv"):
        assert (tmp_path / "a" / name).read_bytes() != b""
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "a" / name).drop(columns=["seconds", "peak_memory_gb"],
                                                    errors="ignore"),
            pd.read_csv(tmp_path / "b" / name).drop(columns=["seconds", "peak_memory_gb"],
                                                    errors="ignore"))
    kinds = {(r["site"], r["kind"]) for r in report["mask_agreement"]}
    assert kinds == {("1", "nuclei"), ("1", "cells"), ("pooled", "nuclei"), ("pooled", "cells")}
    assert "identity_f1" not in report["mask_agreement"][0]
    assert "displaced_f1" in report["mask_agreement"][0]
