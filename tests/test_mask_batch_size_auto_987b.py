"""Item 987b: batch_size 0 / -1 / None / 'none' / 'auto' is automatic; sizes fit the free GPU and RAM."""
import numpy as np
import pytest

from spacr import _mask_batches as MB
from spacr.io import concatenate_and_normalize
from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                            set_default_settings_preprocess_generate_masks)

GIB = 2**30
SETTINGS = {"timelapse": False, "randomize": False, "batch_size": 2,
            "lower_percentile": 2, "nucleus_channel": 0, "cell_channel": 1,
            "pathogen_channel": None, "organelle_channel": None,
            "nucleus_background": 100, "nucleus_signal_to_noise": 5,
            "remove_background_nucleus": False, "cell_background": 100,
            "cell_signal_to_noise": 5, "remove_background_cell": False,
            "background": 100, "Signal_to_noise": 5, "remove_background": False,
            "plot": False, "figuresize": 4, "cmap": "inferno", "nr": 1}


def _plate(root, count, seed=3, shape=(8, 8)):
    stack = root / "stack"
    stack.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        scale = 1.0 + 0.4 * index                       # fields differ, so percentiles differ
        field = (rng.random((*shape, 3)) * 2900.0 * scale + 60.0).astype(np.float32)
        np.save(stack / f"plate1_A01_F{index:03d}.npy", field)
    return stack


@pytest.mark.parametrize("value", [0, -1, None, "none", "None", "auto", "AUTO", "", "0", "-1"])
def test_every_automatic_spelling_means_auto(value):
    assert MB._batch_size_request(value) is None


@pytest.mark.parametrize("value,expected", [(10, 10), ("12", 12), (1, 1), (50.0, 50)])
def test_a_positive_number_is_a_request(value, expected):
    assert MB._batch_size_request(value) == expected


@pytest.mark.parametrize("value", [-2, "many", [3]])
def test_other_values_are_refused(value):
    with pytest.raises(ValueError, match="batch_size"):
        MB._batch_size_request(value)


def test_explicit_987b_mode_and_old_files_keep_fifty():
    """The earlier automatic selector remains explicit and schema 945 keeps its history."""
    assert set_default_settings_preprocess_generate_masks({"src": "/x", "batch_size": 0})["batch_size"] == 0
    assert DEFAULT_CHANGES[945] == {"batch_size": 50} and SETTINGS_SCHEMA_VERSION >= 945


def test_auto_picks_the_largest_size_that_fits_a_fake_vram_budget():
    """With 24 GiB free the batch is limited by the card, with plenty it takes every field."""
    per_gpu = int(0.5 * GIB)
    common = dict(per_field_ram=int(1.5 * GIB), free_ram=500 * GIB, margin_gb=1.5,
                  tile_batch=5, model_loaded=False)
    # spare = 20 - 1.5 - 1.3 = 17.2 GiB; working set 5 tiles = 1 GiB; 16.2 / 0.5 = 32
    assert MB._largest_fitting_size(99, per_gpu, free_vram=20 * GIB, **common) == 32
    assert MB._largest_fitting_size(99, per_gpu, free_vram=200 * GIB, **common) == 99
    assert MB._largest_fitting_size(99, per_gpu, free_vram=1 * GIB, **common) == 1
    smaller = MB._largest_fitting_size(99, per_gpu, free_vram=10 * GIB, **common)
    assert 1 <= smaller < 32


def test_auto_is_also_held_down_by_ram_and_the_fields_available():
    common = dict(per_field_gpu=int(0.1 * GIB), margin_gb=1.5, tile_batch=5, model_loaded=True,
                  free_vram=100 * GIB)
    assert MB._largest_fitting_size(99, per_field_ram=2 * GIB, free_ram=22 * GIB, **common) == 10
    assert MB._largest_fitting_size(7, per_field_ram=2 * GIB, free_ram=500 * GIB, **common) == 7
    assert MB._largest_fitting_size(99, per_field_ram=2 * GIB, free_ram=None, **common) == 99


def test_without_a_gpu_only_ram_limits_the_batch():
    assert MB._largest_fitting_size(40, int(GIB), int(GIB), free_vram=None, free_ram=12 * GIB,
                                    margin_gb=1.5, tile_batch=5, model_loaded=False) == 10


def _plan_for(tmp_path, count, request):
    stack = _plate(tmp_path, count)
    settings = dict(SETTINGS, batch_size=request)
    concatenate_and_normalize(str(stack), [0, 1], settings=settings)
    return str(tmp_path / "masks"), settings


def test_an_explicit_size_that_does_not_fit_warns_and_shrinks(tmp_path):
    masks, settings = _plan_for(tmp_path, 12, 10)
    logs = []
    plan = MB._resolve_plan(masks, settings, free_vram=int(3.0 * GIB), free_ram=64 * GIB,
                            log=logs.append)
    assert any(line.startswith("WARNING: batch_size 10") and "using" in line for line in logs)
    used = plan["batch_size_used"]
    assert used < 10 and sum(plan["sizes"]) == 12 and plan["sizes"][0] == used


def test_an_explicit_size_that_fits_is_used_quietly(tmp_path):
    masks, settings = _plan_for(tmp_path, 12, 5)
    logs = []
    plan = MB._resolve_plan(masks, settings, free_vram=200 * GIB, free_ram=500 * GIB, log=logs.append)
    assert plan["sizes"] == [5, 5, 2] and not any("WARNING" in line for line in logs)


def test_auto_records_the_resolved_size_and_a_rerun_keeps_it(tmp_path):
    masks, settings = _plan_for(tmp_path, 9, 0)
    plan = MB._resolve_plan(masks, settings, free_vram=200 * GIB, free_ram=500 * GIB, log=lambda _l: None)
    assert plan["batch_size_request"] is None and plan["batch_size_used"] == 9
    assert plan["sizes"] == [9] and plan["resolved"] is True
    # a rerun with less memory free reads the record instead of deciding again
    again = MB._resolve_plan(masks, settings, free_vram=2 * GIB, free_ram=3 * GIB, log=lambda _l: None)
    assert again["sizes"] == [9] and again["batch_size_used"] == 9


def test_the_remainder_is_last_and_only_a_tiny_one_borrows():
    assert MB._batch_sizes(25, 10) == [10, 10, 5]
    assert MB._borrowing_indices([10, 10, 5]) == []          # 5 is half of 10: not tiny
    assert MB._borrowing_indices([10, 10, 4]) == [2]
    assert MB._borrowing_indices([10, 10, 1]) == [2]
    assert MB._borrowing_indices([10]) == []                  # a single batch normalises itself
    assert MB._borrowing_indices([3]) == []
    assert MB._borrowing_indices([10, 20, 30, 4]) == [3]      # median of the others is 20
    assert MB._TINY_BATCH_FRACTION == 0.5


def _batches(masks):
    plan = MB._read_plan(masks)
    return [MB._load_batch(plan, i) for i in range(len(plan["batches"]))]


def test_a_tiny_last_batch_is_normalised_with_the_median_of_the_others(tmp_path):
    masks, settings = _plan_for(tmp_path, 7, 3)               # sizes [3, 3, 1]: the 1 is tiny
    plan = MB._resolve_plan(masks, settings, free_vram=200 * GIB, free_ram=500 * GIB, log=lambda _l: None)
    assert plan["sizes"] == [3, 3, 1] and plan["borrowed"] == [2]
    first, second, last = _batches(masks)
    plan = MB._read_plan(masks)
    recorded = plan["normalisation"]
    assert set(recorded) == {"0", "1"}                        # the tiny batch records nothing of its own
    for channel in ("0", "1"):
        lows = [recorded[i][channel][0] for i in ("0", "1")]
        highs = [recorded[i][channel][1] for i in ("0", "1")]
        assert lows[0] != lows[1]                             # the two full batches really differ
    median = MB._borrowed_values(plan, 2)
    for channel in ("0", "1"):
        assert median[channel][0] == pytest.approx(np.median([recorded[i][channel][0] for i in ("0", "1")]))
        assert median[channel][1] == pytest.approx(np.median([recorded[i][channel][1] for i in ("0", "1")]))
    # the last field, rescaled by hand with the borrowed values, is what the batch holds
    raw = np.load(tmp_path / "stack" / plan["batches"][2]["fields"][0])
    from skimage import exposure
    for column, channel in enumerate((0, 1)):
        low, high = median[str(channel)]
        expected = exposure.rescale_intensity(raw[..., channel], in_range=(low, high), out_range=(0, 1))
        np.testing.assert_allclose(last["data"][0][..., column], expected, rtol=1e-6, atol=1e-6)


def test_a_single_batch_normalises_itself(tmp_path):
    masks, settings = _plan_for(tmp_path, 3, 0)
    plan = MB._resolve_plan(masks, settings, free_vram=200 * GIB, free_ram=500 * GIB, log=lambda _l: None)
    assert plan["sizes"] == [3] and plan["borrowed"] == []
    only, = _batches(masks)
    assert set(MB._read_plan(masks)["normalisation"]) == {"0"}
    assert only["data"].shape[0] == 3


def test_a_skipped_batch_is_measured_when_the_tiny_batch_needs_it(tmp_path):
    """A resume that never built batch 1 still gives the tiny batch the same borrowed values."""
    masks, settings = _plan_for(tmp_path, 7, 3)
    MB._resolve_plan(masks, settings, free_vram=200 * GIB, free_ram=500 * GIB, log=lambda _l: None)
    _a, _b, expected = _batches(masks)
    plan = MB._read_plan(masks)
    # forget what was recorded, as a run that resumed past those batches would
    import json
    path = MB._plan_file(masks)
    record = json.load(open(path))
    record["normalisation"] = {}
    json.dump(record, open(path, "w"))
    MB._CACHE.clear()
    again = MB._load_batch(MB._read_plan(masks), 2)
    np.testing.assert_array_equal(again["data"], expected["data"])


def test_a_rerun_with_the_recorded_size_gives_identical_batches(tmp_path):
    """Auto resolves to a size; asking for that size outright reproduces every normalised batch."""
    auto_root, explicit_root = tmp_path / "auto", tmp_path / "explicit"
    masks_a, settings_a = _plan_for(auto_root, 8, 0)
    logs = []
    plan = MB._resolve_plan(masks_a, settings_a, free_vram=int(2.6 * GIB), free_ram=64 * GIB, log=logs.append)
    used = plan["batch_size_used"]
    assert 1 <= used <= 8
    masks_b, settings_b = _plan_for(explicit_root, 8, used)
    MB._resolve_plan(masks_b, settings_b, free_vram=200 * GIB, free_ram=500 * GIB, log=lambda _l: None)
    for one, two in zip(_batches(masks_a), _batches(masks_b)):
        np.testing.assert_array_equal(one["data"], two["data"])
        assert list(one["filenames"]) == list(two["filenames"])


def _cgroup_tree(tmp_path, levels):
    """A fake cgroup v2 hierarchy: ``levels`` maps a relative folder to (max, current, inactive_file)."""
    root = tmp_path / 'cg'
    for folder, (limit, current, cache) in levels.items():
        here = root / folder if folder else root
        here.mkdir(parents=True, exist_ok=True)
        (here / 'memory.max').write_text(f'{limit}\n')
        (here / 'memory.current').write_text(f'{current}\n')
        (here / 'memory.stat').write_text(f'anon 1\ninactive_file {cache}\nactive_file 5\n')
    proc = tmp_path / 'cgroup'
    proc.write_text('0::/' + max(levels, key=len) + '\n')
    return str(proc), str(root)


def test_the_cgroup_headroom_is_the_tightest_limit_plus_reclaimable_cache(tmp_path):
    """A capped job is sized by its own scope, the cache it can drop counted as free."""
    from spacr import _mask_batches as B

    gib = 2**30
    proc, root = _cgroup_tree(tmp_path, {
        '': ('max', 50 * gib, 0),
        'user.slice': (60 * gib, 40 * gib, 2 * gib),
        'user.slice/sched-1.scope': (24 * gib, 20 * gib, 3 * gib)})
    assert B._cgroup_headroom_bytes(proc, root) == 7 * gib
    proc, root = _cgroup_tree(tmp_path / 'b', {'': ('max', 1, 0), 'app.scope': ('max', 1, 0)})
    assert B._cgroup_headroom_bytes(proc, root) is None
    assert B._cgroup_headroom_bytes(str(tmp_path / 'absent'), root) is None


def test_free_ram_is_the_smaller_of_the_host_and_the_cgroup(monkeypatch):
    """The batch planners read one number: the host MemAvailable unless a cap is tighter."""
    from spacr import _mask_batches as B

    monkeypatch.setattr(B, '_host_available_ram_bytes', lambda: 100)
    monkeypatch.setattr(B, '_cgroup_headroom_bytes', lambda: 30)
    assert B._available_ram_bytes() == 30
    monkeypatch.setattr(B, '_cgroup_headroom_bytes', lambda: None)
    assert B._available_ram_bytes() == 100
    monkeypatch.setattr(B, '_host_available_ram_bytes', lambda: None)
    monkeypatch.setattr(B, '_cgroup_headroom_bytes', lambda: 30)
    assert B._available_ram_bytes() == 30
