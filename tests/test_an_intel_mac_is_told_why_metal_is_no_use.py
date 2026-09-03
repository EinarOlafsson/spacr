"""An Intel Mac with Intel graphics is told the truth, which is that nothing
can be done.

Reported 2026-09-02 by the maintainer, on a 2020 Intel Mac with Intel Iris
Plus Graphics 1536: "apple chose not to support this in metal and intel dosnt
support these cards with IPEX. if such a card is encountered spacr-doctr
should say why it dosnt work. now it just says that apple metal is detected
but spaCR cant use it."

The old answer was one sentence for every cause -- "torch has Metal support
but this system does not offer a Metal device" -- and on that machine it is
FALSE. The Mac has a Metal device. It draws the display with it. What it does
not have is any PyTorch backend that will open it:

* Apple's MPS backend covers Apple Silicon and the AMD discrete cards Apple
  shipped in Intel Macs. Intel's integrated GPUs were never added.
* Intel's IPEX / ``torch.xpu`` targets Arc and Xe DISCRETE cards on Linux and
  Windows. No macOS build, and not these parts.

So the user is between two vendors, and the useful thing to tell them is that
there is nothing to install -- otherwise they go looking for the driver they
must have got wrong, and there isn't one.
"""
from __future__ import annotations

import types

import pytest

from spacr import accelerator as A


@pytest.fixture(autouse=True)
def _forget_the_cached_gpu_name():
    """`_metal_gpu_name` is `lru_cache`d over a `system_profiler` call."""
    A._metal_gpu_name.cache_clear()
    yield
    A._metal_gpu_name.cache_clear()


def _mac(monkeypatch, *, gpu, release="13.5", machine="x86_64"):
    monkeypatch.setattr(A.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(A.platform, "machine", lambda: machine)
    monkeypatch.setattr(A.platform, "mac_ver", lambda: (release, ("", "", ""), ""))
    monkeypatch.setattr(A, "_metal_gpu_name", lambda: gpu)


def _torch_with_metal_built_but_unavailable():
    """The exact shape torch reports on the reported machine."""
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False,
                                      is_built=lambda: True)))


def test_the_card_is_named_rather_than_the_backend(monkeypatch):
    """The doctor prints "<label> was detected but spaCR cannot use it".

    "Apple Metal (MPS) was detected" is what the maintainer read, and it
    sent them looking at Metal. The card's own name is what they can check
    against About This Mac.
    """
    _mac(monkeypatch, gpu="Intel Iris Plus Graphics 1536")

    found = A._mps(_torch_with_metal_built_but_unavailable())

    assert found is not None
    assert found.detected is True and found.usable is False
    assert found.label == "Intel Iris Plus Graphics 1536"
    assert found.name == "Intel Iris Plus Graphics 1536"


def test_both_vendors_are_named_and_so_is_the_absence_of_a_fix(monkeypatch):
    _mac(monkeypatch, gpu="Intel Iris Plus Graphics 1536")

    note = A._mps(_torch_with_metal_built_but_unavailable()).note

    assert "Apple Silicon" in note and "AMD" in note, (
        "the note must say what Apple's Metal backend DOES cover, or the "
        "reader cannot tell whether their machine is the exception")
    assert "IPEX" in note and "torch.xpu" in note, (
        "Intel's own path is the second thing a user would go looking for")
    assert "NOTHING TO INSTALL" in note.upper(), (
        "the whole point: a user who is not told this goes hunting for a "
        "driver that does not exist")
    assert "does not offer a Metal device" not in note, (
        "this machine HAS a Metal device; that sentence is what made the "
        "old message wrong")
    assert "spacr-remote" in note, (
        "'get a CUDA machine' is true and unhelpful when spaCR ships a "
        "client for exactly that; `spacr-remote` runs the batch on a Linux "
        "box and shares its profiles with the Distributed Jobs screen")


@pytest.mark.parametrize("gpu", [
    "Intel Iris Plus Graphics 1536",
    "Intel Iris Plus Graphics 645",
    "Intel UHD Graphics 630",
    "Intel HD Graphics 4000",
])
def test_every_intel_integrated_family_is_recognised(monkeypatch, gpu):
    _mac(monkeypatch, gpu=gpu)
    assert "NOTHING TO INSTALL" in A._mps(
        _torch_with_metal_built_but_unavailable()).note.upper()


def test_an_old_macos_is_told_to_upgrade_instead(monkeypatch):
    """A different cause with a different answer, and this one IS fixable.

    Asked before the Intel check on purpose: an Apple Silicon Mac on 12.2
    reaches the same branch, and "upgrade macOS" is the true answer there
    too.
    """
    _mac(monkeypatch, gpu="Apple M1", release="12.2", machine="arm64")

    note = A._mps(_torch_with_metal_built_but_unavailable()).note

    assert "12.3" in note and "Upgrading macOS" in note
    assert "NOTHING TO INSTALL" not in note.upper()


def test_an_amd_card_still_gets_the_honest_generic(monkeypatch):
    """The AMD discrete cards ARE supported, so a machine that reaches this
    branch with one has some other problem and must not be told a story
    about Intel."""
    _mac(monkeypatch, gpu="AMD Radeon Pro 5300")

    note = A._mps(_torch_with_metal_built_but_unavailable()).note

    assert "does not offer a Metal device" in note
    assert "Intel" not in note


def test_a_working_metal_machine_is_untouched(monkeypatch):
    """The change must not reach the path that matters most."""
    _mac(monkeypatch, gpu="Apple M2 Pro", machine="arm64")
    torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True,
                                      is_built=lambda: True)))

    found = A._mps(torch)

    assert found.usable is True and found.device == "mps"
    assert found.note == ""
