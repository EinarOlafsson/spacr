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

from spacr.deep_spacr import GPU_ROOM_MB, pick_device        # noqa: E402


class _Card:
    """A CUDA device with a given amount free, in MiB."""

    def __init__(self, free_mb, total_mb=24000, available=True,
                 readable=True):
        self.free_mb, self.total_mb = free_mb, total_mb
        self.available, self.readable = available, readable

    def install(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available",
                            lambda: self.available)

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

    def test_it_says_the_run_will_be_slower(self, monkeypatch):
        _Card(free_mb=62).install(monkeypatch)
        assert "slower" in pick_device()[1]

    def test_it_names_how_to_find_out_what_is_holding_the_card(self,
                                                               monkeypatch):
        """The fix is not spaCR's to make -- something else is using the
        GPU, and the user decides whether to wait or to free it."""
        _Card(free_mb=62).install(monkeypatch)
        assert "nvidia-smi" in pick_device()[1]

    def test_the_stage_is_named(self, monkeypatch):
        """A log with several stages has to say which one fell back."""
        _Card(free_mb=62).install(monkeypatch)
        assert "training" in pick_device(what="training")[1]


class TestNoEntryPointAsksTheOldQuestion:
    def test_the_training_and_inference_paths_ask_for_room(self):
        """`torch.cuda.is_available()` alone is the question that produced
        the OOM; a call site still asking it would still die."""
        import ast
        import inspect

        from spacr import deep_spacr

        # PARSED, not grepped: the sentence describing the defect appears in
        # `pick_device`'s own docstring, and a text search finds it there.
        tree = ast.parse(inspect.getsource(deep_spacr))
        allowed = {"pick_device", "_empty_device_cache"}
        stray = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in allowed:
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                target = inner.func
                if (isinstance(target, ast.Attribute)
                        and target.attr == "is_available"
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "cuda"):
                    stray.append(f"{node.name}:{inner.lineno}")
        assert not stray, (
            f"{len(stray)} call site(s) still ask only whether a GPU "
            f"exists: {stray}")

    def test_the_test_suite_skips_gpu_tests_on_a_busy_card(self):
        """One conftest hook rather than a guard per file -- every test
        that needs the card already carries `@pytest.mark.gpu`, and a guard
        written into each file is a guard the next file will not have."""
        import inspect
        import sys

        conftest = sys.modules.get("conftest") or sys.modules.get(
            "tests.conftest")
        assert conftest is not None
        assert hasattr(conftest, "pytest_runtest_setup")
        source = inspect.getsource(conftest.pytest_runtest_setup)
        assert '"gpu"' in source
