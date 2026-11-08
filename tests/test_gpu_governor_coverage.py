"""The GPU memory governor's edge paths behave as documented on a fake card.

Everything runs on the CPU with the fake CUDA device of
``tests/test_gpu_memory_governor.py``: bad settings fall back to defaults,
device and memory readers degrade to safe values when PyTorch cannot answer,
the ledger survives write failures, models move to the CPU and back, and
batches split by the governor merge back whole.
"""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacr import _gpu_governor as G  # noqa: E402
from tests.test_gpu_memory_governor import GIB, _oom, card  # noqa: E402,F401


def _no_torch(monkeypatch):
    monkeypatch.setattr(G, "_torch", lambda: None)


def test_unreadable_or_out_of_range_settings_fall_back_to_the_defaults():
    """A fraction that is not a number or lies outside (0, 1] is dropped, and
    a margin that is not a number reads as the default 1.5 GB."""
    assert G._normalised({"gpu_memory_fraction": "half"})["gpu_memory_fraction"] is None
    assert G._normalised({"gpu_memory_fraction": 1.5})["gpu_memory_fraction"] is None
    assert G._normalised({"gpu_memory_fraction": -0.2})["gpu_memory_fraction"] is None
    assert G._normalised({"gpu_memory_fraction": "0.25"})["gpu_memory_fraction"] == 0.25
    assert G._normalised({"gpu_memory_margin_gb": "lots"})["gpu_memory_margin_gb"] == 1.5
    assert G._normalised({"gpu_memory_margin_gb": None})["gpu_memory_margin_gb"] == 1.5
    assert G._normalised({"gpu_memory_margin_gb": -3})["gpu_memory_margin_gb"] == 0.0


def test_configuring_without_settings_restores_the_defaults(monkeypatch):
    """``_configure(None)`` puts the defaults in force and hands back the old
    values, which ``_restore`` puts back."""
    monkeypatch.setenv(G._ALLOC_ENV, "kept")
    G._reset()
    G._configure({"gpu_memory_margin_gb": 4, "gpu_fallback_policy": "pause"})
    previous = G._configure(None)
    try:
        assert previous["gpu_memory_margin_gb"] == 4.0
        assert previous["gpu_fallback_policy"] == "pause"
        assert G._config()["gpu_memory_margin_gb"] == 1.5
        assert G._config()["gpu_fallback_policy"] == "per_item"
        G._restore(previous)
        assert G._config()["gpu_fallback_policy"] == "pause"
    finally:
        G._reset()
    assert os.environ[G._ALLOC_ENV] == "kept"


@pytest.mark.parametrize("value", ["", "-1"])
def test_hidden_devices_mean_no_nvidia_driver(monkeypatch, value):
    """``CUDA_VISIBLE_DEVICES`` set to empty or -1 hides every GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    with monkeypatch.context() as patch:
        patch.setattr(G.os.path, "exists", lambda path: True)
        assert G._nvidia_present() is False


def test_the_nvidia_driver_is_found_by_its_file_on_linux(monkeypatch):
    """On Linux the driver is present when its proc file exists."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    asked = []

    def exists(path):
        asked.append(path)
        return path == "/proc/driver/nvidia/version"

    with monkeypatch.context() as patch:
        patch.setattr(G.sys, "platform", "linux")
        patch.setattr(G.os.path, "exists", exists)
        found = G._nvidia_present()
    assert found is True
    assert asked == ["/proc/driver/nvidia/version"]


def test_the_nvidia_driver_is_found_by_its_dll_on_windows(monkeypatch, tmp_path):
    """On Windows the driver is present when ``nvcuda.dll`` is in System32."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    with monkeypatch.context() as patch:
        patch.setattr(G.sys, "platform", "win32")
        missing = G._nvidia_present()
    (tmp_path / "System32").mkdir()
    (tmp_path / "System32" / "nvcuda.dll").write_bytes(b"")
    with monkeypatch.context() as patch:
        patch.setattr(G.sys, "platform", "win32")
        present = G._nvidia_present()
    assert missing is False and present is True


def test_the_allocator_is_left_alone_when_torch_cannot_be_asked(monkeypatch):
    """A PyTorch whose CUDA query raises leaves the allocator setting unset."""
    monkeypatch.delenv(G._ALLOC_ENV, raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    def broken():
        raise RuntimeError("driver gone")

    monkeypatch.setattr(torch.cuda, "is_available", broken)
    assert G._prefer_expandable_segments() is False
    assert G._ALLOC_ENV not in os.environ


def test_without_torch_the_allocator_follows_the_nvidia_driver(monkeypatch):
    """Before PyTorch is imported the setting is made only when a driver is
    visible."""
    monkeypatch.delenv(G._ALLOC_ENV, raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    _no_torch(monkeypatch)
    monkeypatch.setattr(G, "_nvidia_present", lambda: False)
    assert G._prefer_expandable_segments() is False
    assert G._ALLOC_ENV not in os.environ
    monkeypatch.setattr(G, "_nvidia_present", lambda: True)
    assert G._prefer_expandable_segments() is True
    assert os.environ[G._ALLOC_ENV] == G._ALLOC_VALUE


def test_a_run_only_on_the_cpu_reads_as_cpu():
    """Items that all ran on the CPU give the compute path ``cpu``."""
    assert G._compute_path({"cpu_items": 3}) == "cpu"
    assert G._compute_path({}) == ""


def test_a_cupy_runtime_error_counts_as_out_of_memory_only_by_its_message():
    """CuPy's ``CUDARuntimeError`` is out-of-memory only when it says so."""
    class CUDARuntimeError(RuntimeError):
        pass

    assert G._is_oom(CUDARuntimeError("cudaErrorMemoryAllocation: out of memory"))
    assert not G._is_oom(CUDARuntimeError("cudaErrorIllegalAddress"))


def test_the_device_index_without_torch_or_cuda_is_none(monkeypatch):
    """No PyTorch, or a CUDA query that raises, means no CUDA device."""
    _no_torch(monkeypatch)
    assert G._device_index("cuda:0") is None
    monkeypatch.undo()

    def broken():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", broken)
    assert G._device_index("cuda:0") is None


def test_the_device_index_is_read_from_every_device_form(card, monkeypatch):
    """Ints, strings and ``torch.device`` values give their CUDA index, CPU
    devices give None, and an unreadable current device reads as 0."""
    assert G._device_index(3) == 3
    assert G._device_index("cuda:2") == 2
    assert G._device_index("cuda") == 0
    assert G._device_index("cpu") is None
    assert G._device_index(torch.device("cpu")) is None
    assert G._device_index(torch.device("cuda", 1)) == 1
    assert G._device_index(torch.device("cuda")) == 0
    assert G._device_index(None) == 0

    def broken():
        raise RuntimeError("no context")

    monkeypatch.setattr(torch.cuda, "current_device", broken)
    assert G._device_index(None) == 0


def test_unreadable_allocator_counters_read_as_zero(card, monkeypatch):
    """When PyTorch cannot report allocation or reservation both read 0."""
    def broken(index=0):
        raise RuntimeError("no context")

    for name in ("memory_allocated", "max_memory_allocated",
                 "memory_reserved", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, broken)
    assert G._allocated(0) == (0, 0)
    assert G._reserved(0) == (0, 0)
    assert G._held(0) == G._CONTEXT_BYTES


def test_releasing_frees_cupy_pools_even_without_torch(monkeypatch):
    """CuPy's device and pinned pools are freed when CuPy is loaded, with or
    without PyTorch."""
    freed = []

    class Pool:
        def __init__(self, name):
            self.name = name

        def free_all_blocks(self):
            freed.append(self.name)

    cupy = types.SimpleNamespace(
        get_default_memory_pool=lambda: Pool("device"),
        get_default_pinned_memory_pool=lambda: Pool("pinned"))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    _no_torch(monkeypatch)
    G._release()
    assert freed == ["device", "pinned"]


def test_releasing_survives_failing_torch_and_cupy(monkeypatch):
    """Errors from ``empty_cache`` or the CuPy pools never escape a release."""
    calls = []

    def broken_empty():
        calls.append("empty")
        raise RuntimeError("busy")

    def broken_pool():
        calls.append("pool")
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", broken_empty)
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(
        get_default_memory_pool=broken_pool,
        get_default_pinned_memory_pool=broken_pool))
    G._release()
    assert calls == ["empty", "pool"]


def test_releasing_without_cuda_leaves_the_cache_alone(monkeypatch):
    """A PyTorch without CUDA is not asked to empty its cache."""
    emptied = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    monkeypatch.setitem(sys.modules, "cupy", None)
    G._release()
    assert emptied == []


def test_a_fraction_that_cannot_be_set_is_reported_once(card, monkeypatch, capsys):
    """A failing per-process cap prints the reason and is not retried."""
    calls = []

    def refuse(fraction, index=0):
        calls.append(fraction)
        raise RuntimeError("not supported")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", refuse)
    config = G._normalised({"gpu_memory_fraction": 0.5})
    G._apply_fraction(0, config)
    G._apply_fraction(0, config)
    out = capsys.readouterr().out
    assert "could not cap this process at 0.50 of cuda:0: not supported" in out
    assert calls == [0.5]
    assert G._STATE["fractions"][0] == 0.5


def test_the_headroom_check_skips_an_unreadable_card(card, monkeypatch):
    """Without the card's size the fraction is not raised."""
    def broken(index=0):
        raise RuntimeError("no context")

    monkeypatch.setattr(torch.cuda, "mem_get_info", broken)
    card.peak_reserved = 30 * GIB
    G._keep_headroom(0, G._normalised({"gpu_memory_fraction": 0.1}))
    assert G._STATE["floors"] == {}
    assert card.fractions == []


def test_a_full_fraction_is_not_raised_past_the_whole_card(card, capsys):
    """A need above the whole card cannot raise a fraction already at 1.0."""
    card.peak_reserved = 30 * GIB
    G._keep_headroom(0, G._normalised({"gpu_memory_fraction": 1.0}))
    assert G._STATE["floors"] == {}
    assert "raised to" not in capsys.readouterr().out


@pytest.mark.parametrize("platform, local, expected", [
    ("win32", "LOCALDATA", ("LOCALDATA", "spacr")),
    ("win32", None, ("HOME", "AppData", "Local", "spacr")),
    ("darwin", None, ("HOME", "Library", "Caches", "spacr")),
    ("linux", None, ("HOME", ".cache", "spacr")),
])
def test_the_cache_folder_follows_the_platform(monkeypatch, tmp_path,
                                               platform, local, expected):
    """Without ``XDG_CACHE_HOME`` the cache sits where each platform keeps it."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", "  ")
    if local is None:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
    else:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / local))
    with monkeypatch.context() as patch:
        patch.setattr(G.sys, "platform", platform)
        found = G._cache_dir()
    roots = {"HOME": str(home), "LOCALDATA": str(tmp_path / "LOCALDATA")}
    assert found == os.path.join(roots.get(expected[0], expected[0]), *expected[1:])


def test_an_unknown_process_has_no_start_time():
    """A process id that does not exist has no readable start time."""
    assert G._process_start(-5) is None


def test_bad_process_ids_are_never_alive():
    """Unparseable, zero and negative process ids belong to no process."""
    assert G._alive("not-a-pid") is False
    assert G._alive(None) is False
    assert G._alive(0) is False
    assert G._alive(-4) is False


def _kill_raising(error):
    def kill(pid, signal):
        raise error
    return kill


@pytest.mark.parametrize("error, alive", [
    (ProcessLookupError(), False),
    (PermissionError(), True),
    (OSError("bad"), False),
])
def test_without_psutil_liveness_comes_from_a_null_signal(monkeypatch, error, alive):
    """Without psutil a null signal decides: no such process is dead, one
    owned by another user is alive, any other failure is dead."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    with monkeypatch.context() as patch:
        patch.setattr(G.sys, "platform", "linux")
        patch.setattr(G.os, "kill", _kill_raising(error))
        found = G._alive(12345)
    assert found is alive


def test_without_psutil_a_signalled_process_is_alive(monkeypatch):
    """Without psutil this very process answers the null signal."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert G._alive(os.getpid()) is True


def test_without_psutil_on_windows_every_entry_counts_as_alive(monkeypatch):
    """On Windows without psutil there is no safe check, so entries are kept."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    with monkeypatch.context() as patch:
        patch.setattr(G.sys, "platform", "win32")
        found = G._alive(12345)
    assert found is True


def test_a_failed_ledger_write_leaves_no_temporary_file(tmp_path):
    """When the rename fails the temporary file is removed and the error
    reaches the caller."""
    target = tmp_path / "gpu_ledger.json"
    target.mkdir()
    (target / "keep").write_text("x")
    with pytest.raises(OSError):
        G._write_ledger(str(target), {"a": {"pid": 1}})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gpu_ledger.json"]
    assert (target / "keep").read_text() == "x"


def test_the_ledger_runs_unlocked_where_flock_is_missing(monkeypatch, tmp_path):
    """Without ``fcntl`` the block still runs, with no lock file made."""
    monkeypatch.setitem(sys.modules, "fcntl", None)
    path = str(tmp_path / "gpu_ledger.json")
    ran = []
    with G._ledger_lock(path):
        ran.append(True)
    assert ran == [True]
    assert not os.path.exists(path + ".lock")


def test_removing_this_process_drops_its_ledger_entry(tmp_path):
    """``remove=True`` takes this process's line out of the ledger."""
    path = str(tmp_path / "cache" / "gpu_ledger.json")
    G._ledger_update(0, used=5, want=7, path=path)
    key = f"{os.getpid()}:0"
    assert key in G._read_ledger(path)
    assert G._ledger_update(0, remove=True, path=path) == {}
    assert key not in G._read_ledger(path)
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == {"entries": {}}


def test_an_unwritable_ledger_folder_reads_as_no_other_jobs(tmp_path):
    """A ledger that cannot be created costs only the shared view."""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert G._ledger_update(0, used=1, path=str(blocker / "gpu_ledger.json")) == {}


def test_a_broken_ledger_entry_is_skipped_in_the_pending_claims():
    """An entry with an unreadable size is ignored; the rest still count."""
    others = {
        "bad": {"used": "lots"},
        "good": {"used": G._CONTEXT_BYTES, "want": G._CONTEXT_BYTES + 5},
    }
    assert G._pending_elsewhere(others) == 5


def test_the_model_device_comes_from_its_parameters():
    """A model without a ``device`` attribute is placed by its first
    parameter, and one with no parameters has no device."""
    class Params:
        def __init__(self, items):
            self.items = items

        def parameters(self):
            return iter(self.items)

    assert G._model_device(Params([types.SimpleNamespace(device="cuda:3")])) == "cuda:3"
    assert G._model_device(Params([])) is None


def test_the_wrapper_chain_skips_failing_and_unrelated_attributes():
    """A wrapper attribute that raises, or holds something that is not a
    model, does not extend the chain."""
    class Wrapper:
        _model = object()

        @property
        def model(self):
            raise RuntimeError("lazy load failed")

    outer = Wrapper()
    assert G._model_chain(outer) == [outer]


def test_without_torch_nothing_moves_to_the_cpu(monkeypatch):
    """Without PyTorch there is no model to move."""
    _no_torch(monkeypatch)
    assert G._cpu_target(torch.nn.Linear(2, 2)) is None


def test_a_plain_object_runs_unmoved_on_the_cpu_path():
    """``_on_cpu`` hands back an object it cannot move, unchanged."""
    thing = object()
    with G._on_cpu(thing) as inside:
        assert inside is thing


def test_a_module_moves_to_the_cpu_and_back():
    """A PyTorch module is moved to the CPU for the block and back after."""
    moves = []

    class Tracked(torch.nn.Linear):
        def to(self, *args, **kwargs):
            moves.append(str(args[0]))
            return super().to(*args, **kwargs)

    module = Tracked(2, 2)
    assert G._cpu_target(module) == ("module", module)
    with G._on_cpu(module) as inside:
        assert inside is module
    assert moves == ["cpu", "cpu"]


def test_a_module_without_parameters_is_not_moved_back():
    """A module with no parameters has no device to return to."""
    moves = []

    class Bare(torch.nn.Identity):
        def to(self, *args, **kwargs):
            moves.append(str(args[0]))
            return self

    with G._on_cpu(Bare()):
        pass
    assert moves == ["cpu"]


def test_a_cellpose_model_without_a_gpu_flag_keeps_none():
    """A Cellpose-like model with ``net`` and ``device`` but no ``gpu`` flag
    is moved and restored without gaining one."""
    class Net:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    model = types.SimpleNamespace(net=Net(), device="cuda:0")
    with G._on_cpu(model):
        assert str(model.device) == "cpu"
        assert not hasattr(model, "gpu")
    assert model.device == "cuda:0"
    assert model.net.moves == ["cpu", "cuda:0"]
    assert not hasattr(model, "gpu")


def test_an_error_that_is_not_out_of_memory_reaches_the_caller(card):
    """Only out-of-memory errors are retried; any other is raised at once."""
    seen = []

    def work(size):
        seen.append(size)
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        G._controller("seg").run(work, 8)
    assert seen == [8]
    assert G._counters()["oom_retries"] == 0


def test_without_a_cpu_path_the_error_is_raised_after_the_wait(card, monkeypatch):
    """An item that never fits and has no CPU path is raised once the wait
    for memory runs out."""
    monkeypatch.setattr(G, "_WAIT_SECONDS", 2 * G._POLL_SECONDS)
    seen = []

    def work(size):
        seen.append(size)
        raise _oom()

    with pytest.raises(torch.cuda.OutOfMemoryError):
        G._controller("seg").run(work, 1)
    assert seen == [1] * (1 + G._RETRIES_AT_ONE + 2)
    assert G._counters()["waits"] == 2


def test_the_batch_grows_back_in_steps_below_the_request(card):
    """A ramp-up that stays below the requested size keeps the new cap."""
    fail = {"on": True}

    def work(size):
        if fail["on"] and size > 2:
            raise _oom()
        return size

    controller = G._controller("seg")
    assert controller.run(work, 16) == 2
    fail["on"] = False
    sizes = [controller.run(work, 16) for _ in range(G._RAMP_AFTER - 1)]
    assert sizes[-1] == 2
    assert controller.cap == 4
    assert controller.run(work, 16) == 4


def test_a_batch_that_fits_is_kept(card):
    """With a learned unit size and room for the whole batch nothing shrinks."""
    controller = G._controller("seg")
    controller.per_unit = float(GIB)
    assert controller._fit(4, 0, G._config()) == 4
    assert G._counters()["shrinks"] == 0 and G._counters()["waits"] == 0


def test_a_wait_that_never_frees_memory_runs_at_batch_size_one(card, monkeypatch):
    """When memory stays full through the whole wait the batch drops to 1."""
    monkeypatch.setattr(G, "_WAIT_SECONDS", 3 * G._POLL_SECONDS)
    card.free = GIB
    controller = G._controller("seg")
    controller.per_unit = 2.0 * GIB
    assert controller._fit(4, 0, G._config()) == 1
    assert controller.cap == 1
    assert len(card.sleeps) == 3
    assert G._counters()["waits"] == 1


def test_memory_freed_during_the_wait_keeps_the_batch(card):
    """When the wait frees enough for the whole batch it runs at full size."""
    card.free = GIB
    controller = G._controller("seg")
    controller.per_unit = 2.0 * GIB

    def free_up(seconds):
        card.sleeps.append(seconds)
        card.free = 20 * GIB

    G._sleep = free_up
    assert controller._fit(4, 0, G._config()) == 4
    assert controller.cap is None
    assert card.sleeps == [G._POLL_SECONDS]


def test_a_fraction_on_an_unreadable_card_posts_no_ceiling(card, monkeypatch):
    """The ledger line carries no ceiling when the card's size is unknown."""
    def broken(index=0):
        raise RuntimeError("no context")

    monkeypatch.setattr(torch.cuda, "mem_get_info", broken)
    controller = G._controller("seg")
    controller._post(0, want=3, used=2,
                     config=G._normalised({"gpu_memory_fraction": 0.5}))
    entry = G._read_ledger(G._ledger_path())[f"{os.getpid()}:0"]
    assert entry["ceiling"] == 0
    assert entry["want"] == 3 and entry["used"] == 2


def test_parts_merge_back_in_order_for_every_result_kind():
    """Single parts pass through; lists chain; scalar tensors stack; arrays
    concatenate; anything else comes back as the list of parts."""
    only = {"x": 1}
    assert G._merge([only]) is only
    assert G._merge([[1, 2], [3]]) == [1, 2, 3]
    stacked = G._merge([torch.tensor(1.0), torch.tensor(2.0)])
    assert stacked.tolist() == [1.0, 2.0]
    joined = G._merge([np.arange(2), np.arange(2, 5)])
    assert joined.tolist() == [0, 1, 2, 3, 4]
    assert G._merge([1, 2]) == [1, 2]
    assert G._merge([([1], np.arange(1)), ([2], np.arange(1, 2))])[0] == [1, 2]


def test_an_unsized_batch_runs_whole(card):
    """A batch without a length is one item and runs in one call."""
    batch = object()
    seen = []
    assert G._chunked(lambda part: seen.append(part) or "ok", batch,
                      site="seg") == "ok"
    assert seen == [batch]


def test_an_empty_batch_runs_once_without_the_governor(card):
    """An empty batch is passed straight to the work function."""
    seen = []
    assert G._chunked(lambda part: seen.append(part) or "empty", [],
                      site="seg") == "empty"
    assert seen == [[]]
    assert G._counters()["items"] == 0


def test_a_chunked_batch_falls_back_to_the_cpu_part_by_part(card):
    """When no part fits the GPU every row is computed once on the CPU."""
    seen = []

    def gpu(part):
        seen.append(len(part))
        raise _oom()

    result = G._chunked(gpu, [10, 20, 30, 40], site="seg",
                        cpu_fn=lambda part: [value + 1 for value in part])
    assert result == [11, 21, 31, 41]
    assert G._counters()["cpu_fallbacks"] == 1
    assert G._counters()["cpu_items"] == 1


def test_a_chunked_batch_without_a_cpu_path_still_halves(card):
    """Without a CPU path the parts shrink until they fit, rows intact."""
    def gpu(part):
        if len(part) > 2:
            raise _oom()
        return [value * 2 for value in part]

    assert G._chunked(gpu, [1, 2, 3, 4], site="seg") == [2, 4, 6, 8]
    assert G._counters()["halvings"] == 1
