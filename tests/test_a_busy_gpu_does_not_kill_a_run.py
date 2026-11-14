"""The card is shared, and a run started while it is full must not die on it.

Instruction 236's standing rule -- "The GPU is shared. Nothing here kills
another session's process" -- has a consequence nobody had drawn: spaCR has
to cope with a card it cannot have.

Every training and inference entry point read

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

which asks whether a GPU EXISTS, not whether it can be used. With another
session holding 21 GiB of 24, a run built its dataset, built its model, and
died inside `optim/adam.py` with "torch.OutOfMemoryError: Tried to allocate
2.00 MiB". Nine tests across four files were red for the same reason.

`pick_device` asks for ROOM. A slow run that finishes beats a fast one that
does not, so it falls back to the CPU -- announced, never quietly, because
the difference between ten minutes and ten hours is not something to
discover afterwards.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import spacr.accelerator as accelerator  # noqa: E402
from spacr.accelerator import Accelerator  # noqa: E402
from spacr.deep_spacr import GPU_ROOM_MB, pick_device  # noqa: E402


class _Card:
    """A CUDA device with a given amount free, in MiB."""

    def __init__(self, free_mb, total_mb=24000, available=True,
                 readable=True):
        self.free_mb, self.total_mb = free_mb, total_mb
        self.available, self.readable = available, readable

    def install(self, monkeypatch):
        # ``pick_device`` now uses the shared cached resolver. Each card in
        # this test is a different machine, so it must start with a fresh
        # hardware answer rather than inheriting the previous test's GPU.
        monkeypatch.setattr(accelerator, "_CACHED", None)
        # A forced device (CPU-only runners export SPACR_DEVICE=cpu) would
        # answer before any of the probes this card fakes.
        monkeypatch.delenv(accelerator.ENV_DEVICE, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available",
                            lambda: self.available)
        # This fixture simulates a CUDA-only machine, including on a host
        # whose real Metal device would otherwise answer the next probe.
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        def mem_get_info(*_args, **_kwargs):
            if not self.readable:
                raise RuntimeError("this driver has no mem_get_info")
            return (int(self.free_mb * 1024 * 1024),
                    int(self.total_mb * 1024 * 1024))

        monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)


class TestWhichDeviceItPicks:
    def test_an_empty_card_is_used(self, monkeypatch):
        _Card(free_mb=20000).install(monkeypatch)
        device, note = pick_device()
        assert device.type == "cuda"
        assert note == ""

    def test_a_full_card_falls_back_to_the_cpu(self, monkeypatch):
        """THE DEFECT. 62 MiB free is what this machine actually reports
        while somebody else is training."""
        _Card(free_mb=62).install(monkeypatch)
        device, note = pick_device()
        assert device.type == "cpu"
        assert note

    def test_no_card_at_all_is_the_cpu_and_says_nothing(self, monkeypatch):
        """A machine with no GPU is not a machine with a problem."""
        _Card(free_mb=0, available=False).install(monkeypatch)
        device, note = pick_device()
        assert device.type == "cpu"
        assert note == ""

    def test_a_non_cuda_accelerator_is_used_without_cuda_memory_probe(
            self, monkeypatch):
        """Metal and XPU memory are not reported by CUDA's driver API."""
        found = Accelerator(
            kind="mps",
            device="mps",
            label="Apple Metal",
        )
        monkeypatch.setattr(accelerator, "resolve", lambda: found)

        def unexpected_cuda_probe():
            raise AssertionError("a non-CUDA device reached mem_get_info")

        monkeypatch.setattr(torch.cuda, "mem_get_info", unexpected_cuda_probe)

        device, note = pick_device()
        assert device.type == "mps"
        assert note == ""

    def test_exactly_enough_room_is_enough(self, monkeypatch):
        _Card(free_mb=GPU_ROOM_MB).install(monkeypatch)
        assert pick_device()[0].type == "cuda"

    def test_one_mib_short_is_not(self, monkeypatch):
        _Card(free_mb=GPU_ROOM_MB - 1).install(monkeypatch)
        assert pick_device()[0].type == "cpu"

    def test_a_driver_that_cannot_be_asked_is_trusted(self, monkeypatch):
        """An older driver with no `mem_get_info`. Using the card and
        letting a real OOM be a real failure beats guessing -- a wrong
        guess here sends every run on that machine to the CPU forever."""
        _Card(free_mb=0, readable=False).install(monkeypatch)
        assert pick_device()[0].type == "cuda"

    def test_the_caller_can_say_how_much_it_needs(self, monkeypatch):
        """A montage and a training run do not need the same room."""
        _Card(free_mb=2000).install(monkeypatch)
        assert pick_device(room_mb=1000)[0].type == "cuda"
        assert pick_device(room_mb=8000)[0].type == "cpu"


class TestWhatItSays:
    def test_the_note_carries_the_numbers(self, monkeypatch):
        """"Running on CPU" without the memory is a sentence a user cannot
        act on."""
        _Card(free_mb=62, total_mb=24000).install(monkeypatch)
        _device, note = pick_device()
        assert "62" in note and "24000" in note

    @pytest.mark.parametrize("words", [
        "slower",
        # The fix is not spaCR's to make -- something else is using the GPU,
        # and the user decides whether to wait or to free it, so the note
        # names how to find out what is holding the card.
        "nvidia-smi",
    ], ids=["the_run_will_be_slower", "how_to_find_what_holds_the_card"])
    def test_it_says_the_run_will_be_slower(self, monkeypatch, words):
        _Card(free_mb=62).install(monkeypatch)
        assert words in pick_device()[1]

    def test_the_stage_is_named(self, monkeypatch):
        """A log with several stages has to say which one fell back."""
        _Card(free_mb=62).install(monkeypatch)
        assert "training" in pick_device(what="training")[1]


class TestCapabilityIsNotRoom:
    """Device selection asks for room; RNG snapshots ask only for capability."""

    def test_the_attribution_block_never_asks_cuda_when_there_is_no_card(
            self, monkeypatch):
        from spacr import deep_spacr

        def never(*_args, **_kwargs):
            raise AssertionError("asked CUDA about a card that is not there")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", never)
        before = torch.backends.cudnn.benchmark
        with deep_spacr._deterministic_attribution(seed=3):
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.benchmark == before

    def test_the_cache_is_released_only_where_there_is_a_card(self, monkeypatch):
        from spacr import deep_spacr

        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        deep_spacr._empty_device_cache()
        assert calls == []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        deep_spacr._empty_device_cache()
        assert calls == [1]


class TestTheTestSuiteSkipsGpuTestsOnABusyCard:
    """One conftest hook rather than a guard per file -- every test that needs
    the card already carries `@pytest.mark.gpu`."""

    @staticmethod
    def _item(marked):
        import types

        return types.SimpleNamespace(
            stash={},
            get_closest_marker=lambda name: object() if (
                marked and name == "gpu") else None)

    def _conftest(self):
        import sys

        conftest = sys.modules.get("conftest") or sys.modules.get(
            "tests.conftest")
        assert conftest is not None
        return conftest

    def test_a_gpu_test_is_skipped_with_the_reason_when_the_card_is_full(
            self, monkeypatch):
        conftest = self._conftest()
        monkeypatch.setattr(conftest, "_no_room_on_the_gpu",
                            lambda: "the card is held by another session")
        with pytest.raises(pytest.skip.Exception,
                           match="held by another session"):
            conftest.pytest_runtest_setup(self._item(marked=True))

    def test_a_test_that_does_not_need_the_card_is_not_asked_about_it(
            self, monkeypatch):
        conftest = self._conftest()

        def never():
            raise AssertionError("an unmarked test asked about the GPU")

        monkeypatch.setattr(conftest, "_no_room_on_the_gpu", never)
        conftest.pytest_runtest_setup(self._item(marked=False))
