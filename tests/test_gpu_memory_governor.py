"""The GPU memory governor keeps spaCR GPU work alive on a shared card.

Everything here runs on the CPU. A fake card stands in for CUDA:
``torch.cuda.is_available``, ``mem_get_info``, ``memory_allocated``,
``max_memory_allocated``, ``empty_cache`` and
``set_per_process_memory_fraction`` are monkeypatched, and out-of-memory
errors are raised by the fakes as ``torch.cuda.OutOfMemoryError``.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacr import _gpu_governor as G  # noqa: E402
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS  # noqa: E402

GIB = 1 << 30
ROOT = Path(__file__).resolve().parent.parent


class _Card:
    """A CUDA device whose free memory and allocation the test sets."""

    def __init__(self, free=20 * GIB, total=24 * GIB):
        self.free, self.total = free, total
        self.allocated = 0
        self.peak = 0
        self.emptied = 0
        self.fractions = []

    def install(self, monkeypatch):
        cuda = torch.cuda
        monkeypatch.setattr(cuda, "is_available", lambda: True)
        monkeypatch.setattr(cuda, "current_device", lambda: 0)
        monkeypatch.setattr(cuda, "mem_get_info",
                            lambda index=0: (self.free, self.total))
        monkeypatch.setattr(cuda, "memory_allocated",
                            lambda index=0: self.allocated)
        monkeypatch.setattr(cuda, "max_memory_allocated",
                            lambda index=0: self.peak)
        monkeypatch.setattr(cuda, "empty_cache", self._empty)
        monkeypatch.setattr(cuda, "set_per_process_memory_fraction",
                            lambda fraction, index=0:
                            self.fractions.append((fraction, index)))
        return self

    def _empty(self):
        self.emptied += 1

    def use(self, nbytes):
        """Record a call that peaked at ``nbytes`` above the allocation."""
        self.peak = max(self.peak, self.allocated + int(nbytes))


def _oom():
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


@pytest.fixture
def card(monkeypatch, tmp_path):
    """A fake 24 GiB card, a private ledger and no real sleeping."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    sleeps = []
    monkeypatch.setattr(G, "_sleep", sleeps.append)
    G._reset()
    fake = _Card().install(monkeypatch)
    fake.sleeps = sleeps
    yield fake
    G._reset()


def test_an_out_of_memory_error_halves_the_batch_and_retries_the_same_item(card):
    seen = []

    def work(size):
        seen.append(size)
        if size > 2:
            raise _oom()
        return ("item-7", size)

    result = G._controller("seg").run(work, 8)

    assert result == ("item-7", 2)
    assert seen == [8, 4, 2]
    counts = G._counters()
    assert counts["oom_retries"] == 2 and counts["halvings"] == 2
    assert counts["cpu_fallbacks"] == 0 and counts["items"] == 1
    assert card.emptied >= 2


def test_a_cupy_out_of_memory_error_is_recognised_by_name():
    class OutOfMemoryError(MemoryError):
        pass

    assert G._is_oom(OutOfMemoryError("Out of memory allocating 1,024 bytes"))
    assert G._is_oom(RuntimeError("CUDA error: out of memory"))
    assert not G._is_oom(RuntimeError("the distance transform ran out of memory"))
    assert not G._is_oom(ValueError("bad shape"))


def test_after_the_retries_at_batch_one_the_item_runs_on_the_cpu(card):
    seen = []

    def work(size):
        seen.append(size)
        raise _oom()

    result = G._controller("seg").run(work, 4, cpu_fn=lambda size: ("cpu", size))

    assert result == ("cpu", 1)
    assert seen == [4, 2, 1] + [1] * G._RETRIES_AT_ONE
    counts = G._counters()
    assert counts["cpu_fallbacks"] == 1
    assert counts["oom_retries"] == 3 + G._RETRIES_AT_ONE


def test_without_a_cpu_path_the_run_waits_rather_than_exits(card, monkeypatch):
    monkeypatch.setattr(G, "_WAIT_SECONDS", 3 * G._POLL_SECONDS)
    calls = []

    def work(size):
        calls.append(size)
        if len(calls) < 6:
            raise _oom()
        return "done"

    assert G._controller("seg").run(work, 1) == "done"
    assert G._counters()["waits"] >= 1


def test_the_batch_grows_back_after_a_run_of_clean_items(card):
    fail = {"on": True}
    seen = []

    def work(size):
        seen.append(size)
        if fail["on"] and size > 4:
            raise _oom()
        return size

    controller = G._controller("seg")
    assert controller.run(work, 8) == 4
    fail["on"] = False
    sizes = [controller.run(work, 8) for _ in range(G._RAMP_AFTER + 1)]

    assert sizes[:G._RAMP_AFTER - 1] == [4] * (G._RAMP_AFTER - 1)
    assert sizes[-1] == 8
    assert controller.cap is None
    assert G._counters()["ramp_ups"] == 1


def test_a_batch_that_would_not_fit_the_free_memory_is_shrunk(card):
    unit = 512 << 20

    def work(size):
        card.use(size * unit)
        return size

    controller = G._controller("seg")
    assert controller.run(work, 8) == 8
    assert controller.per_unit == pytest.approx(unit)

    card.free = int(1.5 * GIB) + 3 * unit
    assert controller.run(work, 8) == 3
    assert G._counters()["shrinks"] == 1


def test_with_less_free_than_the_margin_the_run_waits_for_memory(card):
    card.free = GIB

    def free_up(seconds):
        card.sleeps.append(seconds)
        card.free = 10 * GIB

    G._sleep = free_up
    assert G._controller("seg").run(lambda size: size, 4) == 4
    assert G._counters()["waits"] == 1
    assert card.sleeps


def test_the_margin_comes_from_the_settings(card):
    card.free = 3 * GIB
    controller = G._controller("seg")
    controller.per_unit = float(GIB)

    narrow = G._config({"gpu_memory_margin_gb": 0.5})
    wide = G._config({"gpu_memory_margin_gb": 1.5})
    assert controller.run(lambda size: size, 8, config=narrow) == 2
    controller.cap = None
    assert controller.run(lambda size: size, 8, config=wide) == 1


def test_the_fraction_caps_the_process_once(card):
    config = G._config({"gpu_memory_fraction": 0.25})
    controller = G._controller("seg")
    controller.run(lambda size: size, 2, config=config)
    controller.run(lambda size: size, 2, config=config)
    assert card.fractions == [(0.25, 0)]


def test_adaptive_batch_off_runs_the_requested_size(card):
    seen = []
    config = G._config({"gpu_adaptive_batch": False})
    card.free = 0
    G._controller("seg").run(lambda size: seen.append(size), 8, config=config)
    assert seen == [8]


def test_without_cuda_every_call_runs_as_asked(monkeypatch):
    G._reset()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seen = []
    G._controller("seg").run(lambda size: seen.append(size), 8)
    assert seen == [8]
    assert G._counters()["items"] == 0


def _dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_the_ledger_ignores_entries_of_dead_processes(card, tmp_path):
    path = G._ledger_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dead = _dead_pid()
    live = os.getppid()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"entries": {
            f"{dead}:0": {"pid": dead, "device": 0, "used": 0,
                          "want": 20 * GIB, "ceiling": 0},
            f"{live}:0": {"pid": live, "device": 0, "used": GIB,
                          "want": 3 * GIB, "ceiling": 0},
        }}, handle)

    others = G._ledger_update(0, used=0, want=GIB)

    assert set(others) == {f"{live}:0"}
    assert G._pending_elsewhere(others) == 2 * GIB
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)["entries"]
    assert f"{dead}:0" not in stored
    assert stored[f"{os.getpid()}:0"]["want"] == GIB
    assert not [name for name in os.listdir(os.path.dirname(path))
                if name.startswith(".gpu_ledger.")]


def test_another_jobs_pending_claim_shrinks_this_batch(card):
    path = G._ledger_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    live = os.getppid()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"entries": {f"{live}:0": {
            "pid": live, "device": 0, "used": 0, "want": 4 * GIB,
            "ceiling": 0}}}, handle)
    card.free = int(1.5 * GIB) + 6 * GIB
    controller = G._controller("seg")
    controller.per_unit = float(GIB)
    assert controller.run(lambda size: size, 8) == 2


def test_the_counts_reach_the_run_report(card, caplog):
    from spacr.runctx import run_context

    def work(size):
        if size > 1:
            raise _oom()
        return size

    with caplog.at_level(logging.INFO, logger="spacr"):
        with run_context("mask", {"gpu_memory_margin_gb": 0.5},
                         log=False) as run:
            assert G._config()["gpu_memory_margin_gb"] == 0.5
            G._controller("seg").run(work, 2)
    assert run.gpu_memory["oom_retries"] == 1
    assert run.gpu_memory["halvings"] == 1
    assert run.to_dict()["gpu_memory"]["items"] == 1
    assert any("GPU memory:" in record.getMessage() for record in caplog.records)
    assert G._config()["gpu_memory_margin_gb"] == 1.5


def test_the_summary_is_empty_when_nothing_adapted():
    assert G._summary({"items": 5}) == ""
    assert "1 CPU fallbacks" in G._summary({"items": 1, "cpu_fallbacks": 1})


def test_the_settings_are_declared_and_alpha_gated():
    from spacr import settings as S

    keys = ("gpu_memory_fraction", "gpu_memory_margin_gb", "gpu_adaptive_batch")
    assert S.ALPHA_FEATURES[803] == {"settings": keys}
    for key in keys:
        assert S._is_alpha("settings", key)
        assert key in S.expected_types
        assert S.tooltips[key].rstrip(".").split("Default ")[-1] in (
            "None", "1.5", "True")
    for build in (S.set_default_settings_preprocess_generate_masks,
                  S.get_measure_crop_settings, S.set_default_train_test_model,
                  S.deep_spacr_defaults):
        resolved = build({})
        assert resolved["gpu_memory_fraction"] is None
        assert resolved["gpu_memory_margin_gb"] == 1.5
        assert resolved["gpu_adaptive_batch"] is True


class _Variadic:
    """A Cellpose-like model whose ``eval`` fails above a tile batch."""

    def __init__(self, limit=2):
        self.limit = limit
        self.sizes = []
        self.net = torch.nn.Linear(1, 1)
        self.device = torch.device("cuda")
        self.gpu = True

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=256,
             compute_masks=True, progress=None):
        self.sizes.append((batch_size, self.device.type))
        if self.device.type == "cuda" and batch_size > self.limit:
            raise _oom()
        return np.ones((4, 4), np.int32), None, None


def test_governed_eval_halves_the_tile_batch(card):
    model = _Variadic(limit=4)
    masks, _flows, _styles = G._governed_eval(model, np.zeros((4, 4)),
                                              site="seg", batch_size=16)
    assert masks.shape == (4, 4)
    assert [size for size, _ in model.sizes] == [16, 8, 4]


def test_governed_eval_falls_back_to_the_cpu_and_moves_the_model_back(card, monkeypatch):
    model = _Variadic(limit=0)
    moved = []
    monkeypatch.setattr(model.net, "to", lambda device: moved.append(str(device)))
    G._governed_eval(model, np.zeros((4, 4)), site="seg", batch_size=2)
    assert model.sizes[-1] == (1, "cpu")
    assert model.device.type == "cuda" and model.gpu is True
    assert moved == ["cpu", "cuda"]


def test_the_ops_phenotype_segmentation_calls_the_governor(card, monkeypatch):
    from spacr import ops_engine

    calls = []
    real = ops_engine._governed_eval

    def spy(model, x, **kwargs):
        calls.append(kwargs["site"])
        return real(model, x, **kwargs)

    monkeypatch.setattr(ops_engine, "_governed_eval", spy)

    class Model(_Variadic):
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            super().eval(x, batch_size=batch_size)
            shape = np.asarray(x).shape[:2]
            masks = np.zeros(shape, np.int32)
            masks[2:6, 2:6] = 1
            return masks, None, None

    stack = np.random.default_rng(0).random((16, 16, 2)).astype(np.float32)
    nucleus, cell = Model(limit=1), Model(limit=1)
    ops_engine._ph_segment_field(
        stack, ["DAPI", "AF750"], (nucleus, cell),
        {"ops_ph_downsample": 1, "ops_ph_batch_size": 4})
    assert calls == ["ops_phenotype", "ops_phenotype"]
    assert [size for size, _ in nucleus.sizes] == [4, 2, 1]


def test_the_ops_backends_retry_an_out_of_memory_call(card):
    from spacr.ops_accel import _try

    attempts = []

    def run(name):
        attempts.append(name)
        if name == "torch" and attempts.count("torch") == 1:
            raise _oom()
        return name

    assert _try(("torch", "numpy"), run) == ("torch", "torch")
    assert attempts == ["torch", "torch"]


def test_the_ops_backends_fall_back_to_numpy_on_a_full_card(card):
    from spacr.ops_accel import _try

    def run(name):
        if name != "numpy":
            raise _oom()
        return "numpy-result"

    assert _try(("torch", "numpy"), run) == ("numpy-result", "torch")
    assert G._counters()["cpu_fallbacks"] == 1


class _Picky(torch.nn.Module):
    """A linear classifier that runs out of memory on more than two rows."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layer = torch.nn.Linear(3, 2)
        self.fail = True

    def forward(self, x):
        if self.fail and x.shape[0] > 2:
            raise _oom()
        return self.layer(x)


def test_cnn_logits_are_computed_in_parts_without_dropping_rows(card):
    from spacr.deep_spacr import _governed_logits

    model = _Picky()
    data = torch.randn(5, 3)
    with torch.no_grad():
        logits = _governed_logits(model, data, None)
        model.fail = False
        expected = model(data)
    assert torch.allclose(logits, expected)


def test_cnn_predictions_are_computed_in_parts(card):
    from spacr.deep_spacr import _governed_predictions

    model = _Picky()
    data = torch.randn(5, 3)
    with torch.no_grad():
        scores, labels, _extra = _governed_predictions(model, data, None, None)
    assert scores.shape[0] == 5 and labels.shape[0] == 5


def test_a_training_batch_split_by_the_governor_gives_the_whole_batch_gradient(card, monkeypatch):
    from spacr.deep_spacr import _gradient_scaler, _train_batch

    monkeypatch.setattr(G, "_device_index", lambda device=None: 0)
    data = torch.randn(6, 3)
    target = torch.tensor([0, 1, 0, 1, 1, 0])
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    model = _Picky()
    scaler = _gradient_scaler(device, enabled=False)
    _train_batch(model, data, target, loss_fn=loss_fn, scaler=scaler,
                 mixed_precision=False, device=device, head_dim=2)
    split = [p.grad.clone() for p in model.parameters()]

    reference = _Picky()
    reference.fail = False
    loss_fn(reference(data), target).backward()
    for got, want in zip(split, (p.grad for p in reference.parameters())):
        assert torch.allclose(got, want, atol=1e-6)
    assert G._counters()["halvings"] >= 1


def test_a_failed_training_attempt_leaves_no_partial_gradient(card, monkeypatch):
    from spacr.deep_spacr import _gradient_scaler, _train_batch

    monkeypatch.setattr(G, "_device_index", lambda device=None: 0)

    class FailsMidway(_Picky):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls == 2:
                raise _oom()
            return self.layer(x)

    data = torch.randn(4, 3)
    target = torch.tensor([0, 1, 0, 1])
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model = FailsMidway()
    G._controller("cnn_training", device).cap = 2
    _train_batch(model, data, target, loss_fn=loss_fn,
                 scaler=_gradient_scaler(device, enabled=False),
                 mixed_precision=False, device=device, head_dim=2)
    reference = _Picky()
    reference.fail = False
    loss_fn(reference(data), target).backward()
    for got, want in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(got.grad, want.grad, atol=1e-6)


def _calls_in(path, function):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            names = set()
            raw_eval = False
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    target = call.func
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        names.add(target.attr)
                        if (target.attr == "eval" and call.args == []
                                and any(k.arg == "x" for k in call.keywords)):
                            raw_eval = True
            return names, raw_eval
    raise AssertionError(f"{function} not found in {path}")


@pytest.mark.parametrize("path, function, helper", [
    ("spacr/object.py", "generate_cellpose_masks", "_governed_eval"),
    ("spacr/spacr_cellpose.py", "identify_masks_finetune", "_governed_eval"),
    ("spacr/spacr_cellpose.py", "generate_masks_from_imgs", "_governed_eval"),
    ("spacr/ops_engine.py", "_ph_segment_field", "_governed_eval"),
    ("spacr/ops_engine.py", "_objects", "_governed_eval"),
    ("spacr/ops_engine.py", "run_spatial_masks", "_governed_eval"),
    ("spacr/ops_accel.py", "_try", "_guarded"),
    ("spacr/measure.py", "_intensity_measurements", "_guarded"),
    ("spacr/measure.py", "_cucim_morphology_table", "_guarded"),
    ("spacr/measure.py", "_calculate_zernike", "_guarded"),
    ("spacr/deep_spacr.py", "apply_model", "_governed_predictions"),
    ("spacr/deep_spacr.py", "apply_model_to_tar", "_governed_predictions"),
    ("spacr/deep_spacr.py", "evaluate_model_performance", "_governed_logits"),
    ("spacr/deep_spacr.py", "test_model_core", "_governed_logits"),
    ("spacr/deep_spacr.py", "train_model", "_train_batch"),
])
def test_each_gpu_path_goes_through_the_governor(path, function, helper):
    names, raw_eval = _calls_in(path, function)
    assert helper in names
    assert not raw_eval
