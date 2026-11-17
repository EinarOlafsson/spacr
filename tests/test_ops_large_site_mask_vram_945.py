"""Item 945: item 921's whole-site Cellpose mask step under a VRAM cap.

The step is sized to ``seg_mask_vram_gb`` and run through the GPU governor:
the flow check runs in row bands that fit, and a step that cannot fit or runs
out of card memory runs on the CPU and is recorded. Everything but the
``gpu``-marked test runs on the CPU: a fake card stands in for CUDA, the
"card" doubles compute on the CPU, and out-of-memory errors are raised as
``torch.cuda.OutOfMemoryError``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

torch = pytest.importorskip("torch")

from spacr import _gpu_governor as G  # noqa: E402
from spacr import ops_large_sites as ols  # noqa: E402
from tests.cellpose_api_contract import configured_eval_arguments  # noqa: E402
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402

GIB = 1 << 30
CUDA = torch.device("cuda:0")
CPU = torch.device("cpu")


def _cells(shape=(400, 400), n=40, seed=3):
    """Touching discs of mixed sizes as a label image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    labels = np.zeros(shape, np.int32)
    for k in range(1, n + 1):
        cy, cx = rng.integers(10, shape[0] - 10), rng.integers(10, shape[1] - 10)
        radius = int(rng.integers(6, 22))
        labels[((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2) & (labels == 0)] = k
    return labels


def _network_outputs(labels, seed=5, noise=0.6):
    """Flows and cell probability as a network would give them: Cellpose's own
    flows of ``labels`` scaled by 5, with noise so that the flow check drops
    some masks."""
    from cellpose.dynamics import masks_to_flows_gpu

    flows = masks_to_flows_gpu(labels, device=CPU)[0].astype(np.float32) * 5
    rng = np.random.default_rng(seed)
    flows += rng.normal(0, noise, flows.shape).astype(np.float32)
    prob = np.where(labels > 0, 3.0, -3.0).astype(np.float32)
    prob += rng.normal(0, 0.5, prob.shape).astype(np.float32)
    return np.ascontiguousarray(flows), prob


@pytest.fixture
def card(monkeypatch, tmp_path):
    """A fake CUDA card the governor reads: 20 of 24 GiB free, nothing held,
    a private ledger and no real sleeping. The "card" mask parts compute on
    the CPU, so card and CPU paths can be compared exactly here; ``oom``
    names the parts that raise ``torch.cuda.OutOfMemoryError`` on the card."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(G, "_sleep", lambda seconds: None)
    G._reset()
    state = {"free": 20 * GIB, "total": 24 * GIB, "oom": set(), "calls": []}

    def index(device=None):
        """CUDA devices are card 0; anything else is not a card."""
        return 0 if getattr(device, "type", str(device).partition(":")[0]) == "cuda" else None

    monkeypatch.setattr(G, "_device_index", index)
    monkeypatch.setattr(G, "_memory", lambda i: (state["free"], state["total"]))
    monkeypatch.setattr(G, "_reserved", lambda i: (0, 0))
    monkeypatch.setattr(G, "_allocated", lambda i: (0, 0))
    for name in ("_whole_mask_step", "_mask_dynamics", "_band_flow_errors"):
        real = getattr(ols, name)

        def double(*args, _real=real, _name=name):
            """The part on the CPU, or an out-of-memory error on the card."""
            device = args[-1]
            state["calls"].append((_name, device.type))
            if device.type == "cuda" and _name in state["oom"]:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.47 GiB")
            return _real(*args[:-1], CPU)

        monkeypatch.setattr(ols, name, double)
    yield state
    G._reset()


class _CardModel:
    """A model on the card, for the mask step's device."""

    device = CUDA


def test_the_banded_flow_check_equals_cellposes_whole_step_and_drops_masks():
    """On one device the banded step gives the whole step's labels exactly."""
    from cellpose.dynamics import compute_masks

    flows, prob = _network_outputs(_cells())
    whole = ols._whole_mask_step(flows, prob, CPU)
    unchecked = compute_masks(flows, prob, flow_threshold=0.0, device=CPU)
    assert len(np.unique(unchecked)) > len(np.unique(whole)) > 20
    for budget in (1.0, 2e5, 4e5, 1e9):
        masks = ols._mask_dynamics(flows, prob, CPU)
        checked, info = ols._banded_flow_check(masks, flows, CPU, budget, {})
        from cellpose.utils import fill_holes_and_remove_small_masks

        labels = fill_holes_and_remove_small_masks(checked, min_size=15)
        assert np.array_equal(labels, whole), budget
        assert info["bands"] >= 1
    assert len(ols.flow_check_bands(ols._mask_dynamics(flows, prob, CPU), 2e5)) > 3


def test_bands_hold_every_mask_once_and_wholly():
    masks = ols._mask_dynamics(*_network_outputs(_cells()), CPU)
    bands = ols.flow_check_bands(masks, 3e5)
    found = np.concatenate([labels for _, _, labels, _ in bands])
    assert sorted(found) == list(range(1, int(masks.max()) + 1))
    touching = ols._touching(masks)
    assert touching
    for y0, y1, labels, context in bands:
        rows = np.nonzero(np.isin(masks, np.concatenate([labels, context])).any(axis=1))[0]
        assert y0 <= rows.min() and rows.max() < y1
        assert set(context) == set().union(*(touching.get(int(k), set()) for k in labels)) - set(labels)
        assert ols._band_bytes(masks, (y0, y1, labels, context)) <= 3e5 or len(labels) == 1


def test_a_band_of_a_float32_site_is_padded_past_cellposes_float32_switch():
    assert ols._float32_rows((5500, 5500)) == 0
    rows = ols._float32_rows((7000, 7000))
    assert (rows + 2) * 7002 > ols.CELLPOSE_FLOAT32_PIXELS >= (rows + 1) * 7002
    assert (7002 * 7002) > ols.CELLPOSE_FLOAT32_PIXELS


def test_a_site_that_fits_runs_item_921s_call_whole_on_the_card(card):
    flows, prob = _network_outputs(_cells())
    notes = []
    labels = ols._cellpose_masks(_CardModel(), flows, prob, settings={}, record=notes,
                                 label="cells")
    assert card["calls"] == [("_whole_mask_step", "cuda")]
    assert notes[0]["device"] == "gpu" and notes[0]["reason"] == "fits"
    assert notes[0]["objects"] == "cells" and notes[0]["budget_gb"] > 4
    assert np.array_equal(labels, ols._whole_mask_step(flows, prob, CPU))


def test_under_a_binding_cap_the_flow_check_runs_in_bands_with_the_same_labels(card):
    flows, prob = _network_outputs(_cells())
    need = ols.mask_step_bytes(prob.shape, int((prob > 0).sum()))
    assert need["dynamics"] < need["flow_check"]
    budget = (need["dynamics"] + need["flow_check"]) / 2
    settings = {"seg_mask_vram_gb": (G._CONTEXT_BYTES + budget) / GIB}
    notes = []
    labels = ols._cellpose_masks(_CardModel(), flows, prob, settings=settings, record=notes)
    assert notes[0]["device"] == "gpu_banded" and notes[0]["reason"] == "fits"
    assert notes[0]["bands"] >= 2
    names = {name for name, device in card["calls"]}
    assert names == {"_mask_dynamics", "_band_flow_errors"}
    assert all(device == "cuda" for _, device in card["calls"])
    assert np.array_equal(labels, ols._whole_mask_step(flows, prob, CPU))


def test_a_cap_that_cannot_be_met_runs_on_the_cpu_and_says_why(card, caplog):
    flows, prob = _network_outputs(_cells())
    notes = []
    with caplog.at_level("WARNING", logger="spacr.ops_large_sites"):
        labels = ols._cellpose_masks(_CardModel(), flows, prob,
                                     settings={"seg_mask_vram_gb": 0.1}, record=notes,
                                     label="cells")
    assert notes[0]["device"] == "cpu" and notes[0]["reason"] == "cap"
    assert notes[0]["budget_gb"] == 0.0
    assert all(device == "cpu" for _, device in card["calls"])
    assert "cells mask step ran on cpu (cap" in caplog.text
    assert np.array_equal(labels, ols._whole_mask_step(flows, prob, CPU))


@pytest.mark.parametrize("parts, settings, device", [
    ({"_whole_mask_step"}, {}, "cpu"),
    ({"_whole_mask_step"}, {"seg_mask_device": "gpu"}, "cpu"),
    ({"_band_flow_errors"}, "binding", "mixed"),
    ({"_mask_dynamics", "_band_flow_errors"}, "binding", "cpu"),
])
def test_an_out_of_memory_error_falls_back_to_the_cpu_with_the_same_labels(card, parts, settings,
                                                                           device):
    flows, prob = _network_outputs(_cells())
    if settings == "binding":
        need = ols.mask_step_bytes(prob.shape, int((prob > 0).sum()))
        settings = {"seg_mask_vram_gb": (G._CONTEXT_BYTES + need["dynamics"] * 1.5) / GIB}
    card["oom"] = parts
    notes = []
    labels = ols._cellpose_masks(_CardModel(), flows, prob, settings=settings, record=notes)
    assert notes[0]["device"] == device and "oom" in notes[0]["reason"]
    assert np.array_equal(labels, ols._whole_mask_step(flows, prob, CPU))
    cpu_after_oom = [d for name, d in card["calls"] if name == "_band_flow_errors"]
    if "_band_flow_errors" in parts:
        assert cpu_after_oom.count("cuda") == 1 + G._RETRIES_AT_ONE


def test_seg_mask_device_cpu_never_touches_the_card(card):
    flows, prob = _network_outputs(_cells())
    notes = []
    labels = ols._cellpose_masks(_CardModel(), flows, prob,
                                 settings={"seg_mask_device": "cpu"}, record=notes)
    assert card["calls"] == [("_whole_mask_step", "cpu")]
    assert notes[0]["device"] == "cpu" and notes[0]["reason"] == "setting"
    assert np.array_equal(labels, ols._whole_mask_step(flows, prob, CPU))


def test_a_model_off_cuda_keeps_item_921s_call_without_the_governor(monkeypatch):
    flows, prob = _network_outputs(_cells())
    monkeypatch.setattr(G, "_guarded", lambda *a, **k: pytest.fail("governed"))

    class _CpuModel:
        device = CPU

    notes = []
    labels = ols._cellpose_masks(_CpuModel(), flows, prob, record=notes)
    assert notes[0]["reason"] == "model_on_cpu"
    assert np.array_equal(labels, ols._whole_mask_step(flows, prob, CPU))


def test_the_settings_have_defaults_tooltips_a_category_and_are_checked():
    from spacr.settings import ALPHA_FEATURES

    defaults = ols.large_site_settings()
    assert defaults["seg_mask_vram_gb"] == 6.0 and defaults["seg_mask_device"] == "auto"
    keys = {"seg_mask_vram_gb", "seg_mask_device"}
    assert keys <= set(ALPHA_FEATURES[945]["module_settings"]["ops_large_sites"])
    assert keys <= set(ols.LARGE_SITE_TOOLTIPS)
    assert set(ols.LARGE_SITE_CATEGORIES["Mask step GPU memory"]) == keys
    with pytest.raises(ValueError, match="seg_mask_device"):
        ols.large_site_settings({"seg_mask_device": "tpu"})
    with pytest.raises(ValueError, match="seg_mask_vram_gb"):
        ols.large_site_settings({"seg_mask_vram_gb": -1})


def test_older_settings_files_keep_the_uncapped_step():
    from spacr.settings import DEFAULT_CHANGES, _fold_older_defaults

    assert DEFAULT_CHANGES[938] == {"seg_mask_vram_gb": 0}
    settings = {}
    _fold_older_defaults(settings, 936)
    assert settings["seg_mask_vram_gb"] == 0


def test_summary_columns_name_one_device_or_mixed():
    assert ols.summarise_mask_steps([])["mask_step_device"] == "in_eval"
    one = ols.summarise_mask_steps([{"device": "gpu", "reason": "fits", "bands": 0},
                                    {"device": "gpu", "reason": "fits", "bands": 0}])
    assert one == {"mask_step_device": "gpu", "mask_step_reason": "fits", "mask_step_bands": 0}
    mixed = ols.summarise_mask_steps([{"device": "gpu_banded", "reason": "fits", "bands": 3},
                                      {"device": "cpu", "reason": "cap;oom", "bands": 2}])
    assert mixed == {"mask_step_device": "mixed", "mask_step_reason": "cap;fits;oom",
                     "mask_step_bands": 5}


class _DiscCellpose:
    """Cellpose's eval as a stand-in network: the flows and probability of the
    thresholded first channel's components; labels when masks are asked for."""

    device = CUDA

    def __init__(self, threshold):
        self.threshold = threshold

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True, progress=None):
        """Check the call as Cellpose would, then return its outputs."""
        from cellpose.dynamics import masks_to_flows_gpu

        check_cellpose_eval_call(x, channel_axis, require_channel_axis=False)
        configured_eval_arguments(locals())
        x = np.asarray(x)
        x = x[..., 0] if x.ndim == 3 else x
        labels = ndimage.label(x > self.threshold)[0].astype(np.int32)
        if compute_masks:
            return labels, None, None
        flows = masks_to_flows_gpu(labels, device=CPU)[0].astype(np.float32) * 5
        prob = np.where(labels > 0, 4.0, -4.0).astype(np.float32)
        return np.zeros(0), [None, flows, prob], None


def _well(tmp_path):
    """One 900 px site of 16 cells as a stacked ``.npy`` well."""
    yy, xx = np.ogrid[:900, :900]
    dna = np.zeros((900, 900), np.uint16)
    actin = np.zeros((900, 900), np.uint16)
    for cy in (150, 350, 550, 750):
        for cx in (150, 350, 550, 750):
            dna[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16 ** 2] = 1000
            actin[(yy - cy) ** 2 + (xx - cx) ** 2 <= 30 ** 2] = 1000
    other = np.zeros_like(dna)
    folder = tmp_path / "npy"
    folder.mkdir()
    np.save(folder / "1.npy", np.stack([dna, actin, other, other, other], axis=-1))
    return folder


def _run(tmp_path, name, **extra):
    """The settings run on the stand-in well, with the cell model's own cells
    (seeded cells, item 965's default, run no cell mask step)."""
    (tmp_path / name).mkdir()
    settings = {"seg_preset": "cell_painting_phalloidin", "seg_sites_src": str(_well(tmp_path / name)),
                "seg_dst": str(tmp_path / name / "out"), "seg_tile_px": 800,
                "seg_tile_overlap_px": 64, "nucleus_diameter": 20.0,
                "seg_measure_band_px": 300, "seg_cells_from_nucleus_seeds": False,
                "seg_merge_small_nuclei_frac": 0.0, **extra}
    return ols.run_large_site_segmentation(settings, models=(_DiscCellpose(0.5), _DiscCellpose(0.5)),
                                           available_bytes=2 ** 40)


def test_the_run_records_the_fallback_and_keeps_the_labels(card, tmp_path):
    card["oom"] = {"_whole_mask_step"}
    oom = _run(tmp_path, "oom")
    card["oom"] = set()
    cpu = _run(tmp_path, "cpu", seg_mask_device="cpu")
    padded = ols.tiling_layout((900, 900), 800, 2 * ols.FLOW_MARGIN_PX, True)["padded"]
    need = ols.mask_step_bytes(padded, 16 * int(np.pi * 30 ** 2))
    budget = (need["dynamics"] + need["whole"]) / 2
    capped = _run(tmp_path, "capped", seg_mask_vram_gb=(G._CONTEXT_BYTES + budget) / GIB)
    sites = {name: pd.read_csv(tmp_path / name / "out" / "sites.csv")
             for name in ("oom", "cpu", "capped")}
    assert sites["oom"].loc[0, "mask_step_device"] == "cpu"
    assert sites["oom"].loc[0, "mask_step_reason"] == "oom"
    assert sites["cpu"].loc[0, "mask_step_reason"] == "setting"
    assert sites["capped"].loc[0, "mask_step_device"] == "gpu_banded"
    assert sites["capped"].loc[0, "mask_step_bands"] >= 1
    assert sites["oom"].loc[0, "cells"] == 16
    reference = np.load(tmp_path / "cpu" / "out" / "labels" / "1_cells.npy")
    for name in ("oom", "capped"):
        assert np.array_equal(np.load(tmp_path / name / "out" / "labels" / "1_cells.npy"),
                              reference)
    saved = json.loads((tmp_path / "oom" / "out" / "mask_step.json").read_text())
    assert saved["sites"]["1"][0]["reason"] == "oom" and saved["seg_mask_vram_gb"] == 6.0
    assert oom["mask_steps"]["sites"]["1"][0]["objects"] == "cells"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")
def test_on_a_real_card_the_capped_step_matches_the_uncapped_and_the_cpu():
    """Capped (banded) and uncapped on the card are identical; the CPU agrees
    within item 921's tolerance (every object at IoU >= 0.9)."""
    G._reset()
    flows, prob = _network_outputs(_cells((600, 600), n=80))

    class _Model:
        device = torch.device("cuda")

    uncapped = ols._whole_mask_step(flows, prob, torch.device("cuda"))
    need = ols.mask_step_bytes(prob.shape, int((prob > 0).sum()))
    G._release()
    held = G._held(G._device_index(torch.device("cuda")))
    notes = []
    capped = ols._cellpose_masks(_Model(), flows, prob, record=notes,
                                 settings={"seg_mask_vram_gb": (held + need["dynamics"] * 1.5) / GIB})
    assert notes[0]["device"] == "gpu_banded"
    assert np.array_equal(capped, np.asarray(uncapped, np.int32))
    on_cpu = np.asarray(ols._whole_mask_step(flows, prob, CPU), np.int32)
    for k in range(1, int(capped.max()) + 1):
        mine = capped == k
        best = np.bincount(on_cpu[mine]).argmax()
        other = on_cpu == best
        assert best > 0 and (mine & other).sum() / (mine | other).sum() >= 0.9
    G._reset()
