"""A mixed fit says what it needs before asking the machine for it.

REPORTED 2026-08-18: "i ran one ols then a mixed regression model and this hung
my computer twice so i had to restart it."

`mixed_gpu` allocates a DENSE n x q design with no size check. An allocation
that asks the operating system for more than it has does not fail politely; it
takes the session and everything else the user had open with it. The shape is
known exactly before the matrix exists, so the bytes are too -- and a refusal a
user can read beats a machine they have to power-cycle.
"""

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

import pytest


def test_the_size_is_exact_not_estimated():
    from spacr.mixed_gpu import design_bytes

    # float64: 8 bytes a cell, and nothing else in the arithmetic.
    assert design_bytes(1000, 200) == 1000 * 200 * 8
    assert design_bytes(1000, 200, itemsize=4) == 1000 * 200 * 4


def test_a_design_that_fits_is_not_refused():
    from spacr.mixed_gpu import _refuse_if_too_large

    # The maintainer's own merge: 226,467 cells x 1,212 random effects is
    # 2.2 GB, which is large and is NOT what took the machine down. A guard
    # that refused this would have broken a fit that works.
    _refuse_if_too_large(226_467, 1_212, device="cpu")


def test_an_impossible_design_is_refused_with_both_numbers():
    from spacr.mixed_gpu import _refuse_if_too_large

    with pytest.raises(MemoryError) as caught:
        _refuse_if_too_large(5_000_000, 50_000, device="cpu")
    message = str(caught.value)
    assert "TB" in message, "it must say how much it wanted"
    assert "free" in message, "and how much there was"
    # And what to do, because a refusal with no alternative is a dead end.
    assert "ols" in message


def test_it_refuses_rather_than_allocating(monkeypatch):
    """The guard must fire BEFORE the allocation, not catch its failure.

    Catching MemoryError afterwards is not the same thing: by then the request
    has already been made, and on Linux an over-large request is answered by
    the OOM killer rather than by an exception.
    """
    from spacr import mixed_gpu

    monkeypatch.setattr(mixed_gpu, "available_memory", lambda device="cpu": 1024)
    with pytest.raises(MemoryError):
        mixed_gpu._refuse_if_too_large(10_000, 10_000, device="cpu")


def test_headroom_is_left_for_everything_else(monkeypatch):
    """A fit is not the only thing in the process.

    The merged frame it came from, both results tables and every figure drawn
    are live beside it -- so taking all of what is free is how the refusal
    arrives too late.
    """
    from spacr import mixed_gpu

    assert 0 < mixed_gpu.MEMORY_HEADROOM < 1
    monkeypatch.setattr(mixed_gpu, "available_memory", lambda device="cpu": 1000)
    # 600 bytes is under the free total and over the headroom share.
    with pytest.raises(MemoryError):
        mixed_gpu._refuse_if_too_large(600, 1, itemsize=1, device="cpu") \
            if False else mixed_gpu._refuse_if_too_large(75, 1, device="cpu")
