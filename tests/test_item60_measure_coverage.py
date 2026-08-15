"""Normal and defensive edge cases for the item-60 measure sweep."""
from __future__ import annotations

from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M


def _mask_three():
    mask = np.zeros((12, 18), dtype=np.int32)
    mask[2:5, 2:5] = 1
    mask[2:5, 7:10] = 2
    mask[7:10, 13:16] = 3
    return mask


def _morph_settings(**over):
    settings = {
        "cell_mask_dim": 0,
        "nucleus_mask_dim": None,
        "pathogen_mask_dim": None,
        "organelle_mask_dim": 1,
        "cytoplasm": False,
        "spatial_measurements": True,
        "spatial_neighbor_radius": "bad",
        "organelle_morphology": "spots",
        "organelle_type": "custom",
    }
    settings.update(over)
    return settings


def test_zernike_empty_region_radius_and_inconsistent_vectors(monkeypatch):
    regions = [SimpleNamespace(image=np.zeros((2, 2), dtype=bool)),
               SimpleNamespace(image=np.ones((2, 2), dtype=bool))]
    monkeypatch.setattr(M, "regionprops", lambda _mask: regions)
    vectors = iter([np.array([1., 2.]), np.array([1.])])
    monkeypatch.setattr(M, "_load_zernike_moments",
                        lambda: lambda *_a, **_k: next(vectors))
    with pytest.raises(ValueError, match="same length"):
        M._calculate_zernike(np.ones((2, 2), dtype=int),
                             pd.DataFrame({"label": [1, 2]}))


def test_analyze_cytoskeleton_single_signal_pixel_gets_zero_skeleton():
    array = np.zeros((8, 8, 1), dtype=float)
    mask = np.zeros((8, 8), dtype=int)
    mask[3, 3] = 1
    array[3, 3, 0] = 5
    out = M._analyze_cytoskeleton(array, mask, 0)
    assert out.to_dict("records") == [{
        "object_label": 1,
        "skeleton_length": 0,
        "skeleton_branch_points": 0,
    }]


def test_spatial_column_names_and_empty_frame_types():
    names = M.spatial_column_names(12.9)
    assert names[0] == "neighbors_within_12"
    empty = M._empty_spatial_frame(12.9)
    assert list(empty.columns) == ["label", *names]
    assert str(empty["label"].dtype) == "int64"


def test_spatial_adjacency_measures_touching_labels():
    mask = np.zeros((8, 8), dtype=int)
    mask[2:6, 1:4] = 1
    mask[2:6, 4:7] = 2
    percent, neighbours = M._spatial_adjacency(mask, expand=0)
    assert percent[1] > 0 and percent[2] > 0
    assert neighbours == {1: 1, 2: 1}


def test_spatial_adjacency_empty_mask_returns_empty_maps():
    assert M._spatial_adjacency(np.zeros((4, 4), dtype=int)) == ({}, {})


def test_spatial_adjacency_old_skimage_refuses_anisotropic_spacing(monkeypatch):
    monkeypatch.setattr(M, "_EXPAND_LABELS_TAKES_SPACING", False)
    with pytest.raises(M.ConfigurationError, match="anisotropic"):
        M._spatial_adjacency(np.zeros((2, 3, 3), dtype=int),
                             spacing=(2., .2, .2))
    assert M._spatial_adjacency(np.zeros((3, 3), dtype=int), spacing=None) == ({}, {})


def test_spatial_measurements_empty_single_pair_and_triple():
    empty = M._spatial_measurements(np.zeros((5, 5), dtype=int), radius=4)
    assert empty.empty

    singleton = np.zeros((6, 6), dtype=int)
    singleton[2:4, 2:4] = 1
    one = M._spatial_measurements(singleton, radius=4)
    assert one.loc[0, "nearest_neighbor_distance"] == M._SPATIAL_NO_NEIGHBOUR
    assert one.loc[0, "second_neighbor_distance"] == M._SPATIAL_NO_NEIGHBOUR

    pair = _mask_three()
    two = M._spatial_measurements(np.where(pair == 3, 0, pair), radius=20)
    assert (two["neighbors_within_20"] == 1).all()
    assert (two["second_neighbor_distance"] == M._SPATIAL_NO_NEIGHBOUR).all()

    three = M._spatial_measurements(pair, spacing=(2., 1.), radius=50)
    assert len(three) == 3
    assert np.isfinite(three.drop(columns="label").to_numpy()).all()


def test_morphology_spatial_invalid_radius_uses_default_and_organelle_gate():
    cell = _mask_three()
    organelle = np.zeros_like(cell)
    organelle[2:3, 2:3] = 1
    empty = np.zeros_like(cell)
    out = M._morphological_measurements(
        cell, empty, empty, organelle, empty, _morph_settings(), zernike=False)
    assert "cell_neighbors_within_50" in out[0]
    assert "organelle_neighbors_within_50" in out[3]
    assert out[0].columns[0] == "label"


def test_summarize_organelle_missing_label_intensity_defaults_zero(monkeypatch):
    fake = pd.DataFrame({"label": [7], "area": [1.], "eccentricity": [0.],
                         "solidity": [1.], "major_axis_length": [1.],
                         "minor_axis_length": [1.]})
    monkeypatch.setattr(M, "_safe_morphology_table", lambda *_a, **_k: fake.copy())
    monkeypatch.setattr(
        M, "_map_child_to_parent",
        lambda *_a, **_k: pd.DataFrame({"organelle_label": [7], "cell": [1]}))
    parent = np.zeros((3, 3), dtype=int)
    parent[1:, 1:] = 1
    out = M._summarize_organelles_per_parent(
        np.zeros((3, 3), dtype=int), parent, np.ones((3, 3, 1)), "cell")
    assert out.loc[0, "organelle_channel_0_mean_intensity_per_cell"] == 0


def test_extended_regionprops_all_nan_hits_empty_gini():
    labels = np.zeros((5, 5), dtype=int)
    labels[1:4, 1:4] = 1
    image = np.full((5, 5), np.nan)
    out = M._extended_regionprops_table(labels, image, ["label", "area"])
    assert np.isnan(out.loc[0, "gini_intensity"])


def test_extended_regionprops_legacy_intensity_image_name(monkeypatch):
    class LegacyRegion:
        image = np.ones((2, 2), dtype=bool)
        intensity_image = np.arange(4, dtype=float).reshape(2, 2)

    monkeypatch.setattr(M, "regionprops", lambda *_a, **_k: [LegacyRegion()])
    monkeypatch.setattr(
        M, "regionprops_table",
        lambda *_a, **_k: {"label": np.array([1]), "area": np.array([4.])})
    out = M._extended_regionprops_table(
        np.ones((2, 2), dtype=int), np.ones((2, 2)), ["label", "area"])
    assert out.loc[0, "integrated_intensity"] == 6


def test_periphery_and_outside_empty_samples(monkeypatch):
    labels = np.zeros((4, 4), dtype=int)
    labels[1:3, 1:3] = 1
    monkeypatch.setattr(M, "find_boundaries",
                        lambda *_a, **_k: np.zeros((4, 4), dtype=bool))
    assert np.isnan(M._periphery_intensity(labels, np.ones((4, 4)))[0][1])
    monkeypatch.setattr(M, "binary_dilation", lambda region, **_k: region.copy())
    assert np.isnan(M._outside_intensity(labels, np.ones((4, 4)), distance=1)[0][1])


def test_radial_distribution_empty_parent_and_zero_distance(monkeypatch):
    original_unique = np.unique
    sequence = iter([np.array([1]), np.array([1])])
    monkeypatch.setattr(M.np, "unique", lambda _value: next(sequence))
    empty = M._calculate_radial_distribution(
        np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int),
        np.ones((2, 2, 1)), num_bins=2)
    assert np.isnan(empty[(1, 1, 0)]).all()
    monkeypatch.setattr(M.np, "unique", original_unique)

    cell = np.ones((2, 2), dtype=int)
    obj = np.ones((2, 2), dtype=int)
    monkeypatch.setattr(M, "distance_transform_edt",
                        lambda *_a, **_k: np.zeros((2, 2)))
    zero = M._calculate_radial_distribution(cell, obj, np.ones((2, 2, 1)), 2)
    assert zero[(1, 1, 0)][0] == 1


def test_corrected_manders_signal_and_background_paths():
    mask = np.array([[0, 1, 1], [0, 1, 1]], dtype=int)
    signal = np.array([[0., 0., 0.], [0., 0., 8.]])
    out = M._calculate_correlation_object_level(
        signal, signal * 2, mask,
        {"manders_thresholds": [50], "corrected_manders": True})
    assert out.loc[0, "manders_m1"] == pytest.approx(1.)
    assert out.loc[0, "manders_overlap_coefficient"] == pytest.approx(1.)

    background = M._calculate_correlation_object_level(
        np.zeros_like(signal), np.zeros_like(signal), mask,
        {"manders_thresholds": [50], "corrected_manders": True})
    assert background.loc[0, "manders_m1"] == 0
    assert background.loc[0, "manders_m2"] == 0
    assert background.loc[0, "manders_overlap_coefficient"] == 0


def test_correlation_single_pixel_object_has_nan_pearson():
    mask = np.zeros((2, 2), dtype=int)
    mask[0, 0] = 1
    out = M._calculate_correlation_object_level(
        np.ones((2, 2)), np.ones((2, 2)), mask,
        {"manders_thresholds": [50]})
    assert np.isnan(out.loc[0, "Pearson_correlation"])


@pytest.mark.parametrize("branch", ["coords", "submask", "outside"])
def test_measure_intensity_distance_defensive_geometry(monkeypatch, branch):
    mask = np.ones((3, 3), dtype=int)
    channels = np.ones((3, 3, 1), dtype=float)
    if branch == "coords":
        monkeypatch.setattr(M.np, "argwhere", lambda _value: np.empty((0, 2), int))
    elif branch == "submask":
        real_sum = np.sum
        calls = {"n": 0}

        def zero_first(value, *args, **kwargs):
            calls["n"] += 1
            return 0 if calls["n"] == 1 else real_sum(value, *args, **kwargs)
        monkeypatch.setattr(M.np, "sum", zero_first)
    else:
        monkeypatch.setattr(M, "center_of_mass", lambda _value: (99., 99.))
    out = M._measure_intensity_distance(
        mask, np.zeros_like(mask), np.zeros_like(mask), channels,
        {"distance_gaussian_sigma": 1.})
    assert out.iloc[0, 1:].isna().all()


def test_promote_merged_rejects_invalid_explicit_factor():
    data = np.zeros((2, 2, 2), dtype=float)
    with pytest.raises(ValueError, match="finite and positive"):
        M._promote_merged_to_uint16(
            data, {"cell_mask_dim": 1}, rescale_factor=0)


def test_measure_core_optional_masks_filters_and_field_rescale(
        tmp_path, synth_masks_multi, rng, monkeypatch, capsys):
    from tests.test_measure_crop_core_synth import (
        _build_merged_stack, _settings_for, _write_stack)
    data = _build_merged_stack(synth_masks_multi, rng, with_organelle=True).astype(float)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(
        merged, nucleus_mask_dim=None, organelle_mask_dim=7,
        organelle_min_size=1, save_measurements=False, save_png=False)
    monkeypatch.setattr(
        M, "_resolve_intensity_rescale_record",
        lambda *_a, **_k: {
            "rescale_factor": 2., "rescale_scope": "field",
            "plate_intensity_max": None,
        })
    monkeypatch.setattr(M, "_write_intensity_rescale_record",
                        lambda *_a, **_k: None)
    _index, _average, cells, _figs = M._measure_crop_core(
        0, [], name, settings)
    assert isinstance(cells, np.ndarray)
    assert "NOT comparable" in capsys.readouterr().out


def test_measure_core_plot_handles_volume_and_malformed_plane(
        tmp_path, monkeypatch, capsys):
    from tests.test_measure_crop_core_synth import _settings_for, _write_stack
    monkeypatch.setattr(M, "_write_intensity_rescale_record",
                        lambda *_a, **_k: None)

    volume = np.zeros((2, 6, 6, 4), dtype=np.uint16)
    volume[..., 0] = 1
    merged, name = _write_stack(tmp_path / "volume", volume)
    volume_settings = _settings_for(
        merged, channels=[0], cell_mask_dim=1, nucleus_mask_dim=None,
        pathogen_mask_dim=None, plot=True, save_png=False,
        save_measurements=False, voxel_size_z_um=2., voxel_size_xy_um=1.)
    M._measure_crop_core(0, [], name, volume_settings)
    assert "skipping the cropped-array plots" in capsys.readouterr().out

    # A malformed plane-only array reaches the defensive plot size fallback;
    # the worker then records the expected failure instead of leaking it.
    plane = np.zeros((6, 6), dtype=np.uint16)
    merged2, name2 = _write_stack(tmp_path / "plane", plane)
    plane_settings = _settings_for(
        merged2, channels=[0], cell_mask_dim=None, nucleus_mask_dim=None,
        pathogen_mask_dim=None, plot=True, save_png=False,
        save_measurements=False)
    _i, _t, cells, _f = M._measure_crop_core(0, [], name2, plane_settings)
    assert cells == 0


def test_measure_core_empty_png_size_returns_failure_sentinel(
        tmp_path, synth_masks_multi, rng, capsys):
    from tests.test_measure_crop_core_synth import (
        _build_merged_stack, _settings_for, _write_stack)
    merged, name = _write_stack(
        tmp_path, _build_merged_stack(synth_masks_multi, rng))
    settings = _settings_for(
        merged, png_size=[], save_measurements=False, save_png=True)
    _i, _t, cells, _f = M._measure_crop_core(0, [], name, settings)
    assert cells == 0
    assert "png_size is empty" in capsys.readouterr().out


@pytest.mark.parametrize(
    "over, expected_cytoplasm",
    [
        ({"cell_mask_dim": None, "pathogen_mask_dim": None}, False),
        ({"pathogen_min_size": None, "nucleus_min_size": 1}, True),
        ({"pathogen_min_size": None, "nucleus_min_size": None}, False),
    ],
)
def test_measure_crop_derives_cytoplasm_modes(
        tmp_path, rng, monkeypatch, over, expected_cytoplasm):
    from tests.test_measure_validation import _merged, _settings
    merged = _merged(tmp_path, rng)
    seen = []

    class Plan:
        version = 1

        def filter_files(self, files):
            seen.append(("filter", list(files)))
            return []

    monkeypatch.setattr(M, "plan_measure_resume", lambda _settings: Plan())
    monkeypatch.setattr(M, "build_plate_plan",
                        lambda *_a, **_k: {"version": 1, "plates": {},
                                           "failures": {}})
    from spacr import io as sio
    monkeypatch.setattr(sio, "_save_settings_to_db",
                        lambda settings: seen.append(("settings", settings.copy())))
    settings = _settings(merged, **over)
    M.measure_crop(settings)
    saved = next(value for kind, value in seen if kind == "settings")
    assert saved["cytoplasm"] is expected_cytoplasm
    assert any(kind == "filter" for kind, _value in seen)


def test_measure_crop_dry_run_returns_preflight_result(monkeypatch):
    from spacr import validate
    marker = [SimpleNamespace(message="checked")]
    monkeypatch.setattr(validate, "run_preflight",
                        lambda settings, module: marker if module == "measure" else [])
    assert M.measure_crop({"dry_run": True}) is marker


class _Item60Result:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def get(self):
        if self.error is not None:
            raise self.error
        return self.value


class _Item60Manager:
    def list(self):
        return []


class _Item60Pool:
    def __init__(self, results):
        self.results = iter(results)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def apply_async(self, *_args, **_kwargs):
        return next(self.results)

    def close(self):
        pass

    def join(self):
        pass


class _Item60Context:
    def __init__(self, results):
        self.results = results

    def get_start_method(self):
        return "fork"

    def Pool(self, _jobs):
        return _Item60Pool(self.results)


def _orchestrator_settings(src, **over):
    from spacr.settings import get_measure_crop_settings
    settings = get_measure_crop_settings({
        "src": str(src), "save_png": False, "save_arrays": False,
        "save_measurements": False, "plot": False, "test_mode": False,
        "channels": [0], "cell_mask_dim": 1, "nucleus_mask_dim": None,
        "pathogen_mask_dim": None, "experiment": "exp", "normalize": False,
        "normalize_by": "png", "n_jobs": 1,
    })
    settings.update(over)
    return settings


def _patch_item60_orchestrator(monkeypatch, results):
    context = _Item60Context(results)
    monkeypatch.setattr(M, "_pool_context", lambda: context)

    @contextmanager
    def manager(_ctx):
        yield _Item60Manager()

    monkeypatch.setattr(M, "_start_manager", manager)
    monkeypatch.setattr(M, "build_plate_plan",
                        lambda *_a, **_k: {"version": 1, "plates": {},
                                           "failures": {}})


def test_measure_crop_retries_worker_then_succeeds(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "plate_A01_f1.npy", np.zeros((2, 2, 2), np.uint16))
    success = (0, .1, np.array([0]), {})
    # The pool takes iter(results), so counting what it PULLS is the only
    # observable -- and it is the one that matters: both results consumed
    # means the first raised and the run came back for the second. Without
    # this the test passed for a measure_crop that never retried at all.
    consumed = []

    class _Counted(list):
        def __iter__(self):
            for item in list.__iter__(self):
                consumed.append(item)
                yield item

    _patch_item60_orchestrator(
        monkeypatch,
        _Counted([_Item60Result(error=RuntimeError("transient")),
                  _Item60Result(value=success)]))
    M.measure_crop(_orchestrator_settings(
        merged, on_error="retry", on_error_attempts=2))
    assert len(consumed) == 2, consumed


def test_measure_crop_reraises_pipeline_cancellation(tmp_path, monkeypatch):
    from spacr.cancellation import PipelineCancelled
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "plate_A01_f1.npy", np.zeros((2, 2, 2), np.uint16))
    _patch_item60_orchestrator(
        monkeypatch, [_Item60Result(error=PipelineCancelled("stop"))])
    with pytest.raises(PipelineCancelled, match="stop"):
        M.measure_crop(_orchestrator_settings(merged, on_error="skip"))


def test_measure_crop_timelapse_nucleus_emits_gif(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "plate_A01_f1.npy", np.zeros((2, 2, 2), np.uint16))
    success = (0, .1, np.array([0]), {})
    _patch_item60_orchestrator(monkeypatch, [_Item60Result(value=success)])
    from spacr import timelapse
    seen = []
    monkeypatch.setattr(timelapse, "_timelapse_masks_to_gif",
                        lambda folder, channels, objects: seen.append(
                            (folder, channels, objects)))
    M.measure_crop(_orchestrator_settings(
        merged, timelapse=True, timelapse_objects="nucleus"))
    assert seen and seen[0][2] == ["nucleus", "pathogen", "cell"]
