"""The three array operations OPS runs on, checked against each other.

Instruction 372, PHASE E: "The CPU path is not an error path; both
backends run the same fixture and must agree." That is the contract, and
the reason it is written as a test rather than as a comment is that a
fallback nobody compares is not a fallback -- it is a second
implementation that will drift, and it drifts in the direction nobody
looks, because the accelerated path is the one that runs on the
maintainer's machine and the CPU path is the one that runs everywhere
else.

TORCH IS ASKED FOR BY NAME so this runs where there is no card at all:
`torch.nn.functional.max_pool2d` and `torch.cdist` on the CPU are the
same code with a different device, and CI has no GPU. CuPy cannot do
that -- it is CUDA or nothing -- so it is skipped rather than passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_accel import (CHUNK, accelerated_backends, matmul,
                             maximum_filter, nearest_neighbours)

ACCELERATED = ("torch", "cupy")


def _skip_unless(name):
    if name == "cupy" and "cupy" not in accelerated_backends(gpu=True):
        pytest.skip("this machine has no usable cupy backend")
    if name == "torch":
        pytest.importorskip("torch")


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(4)
    return rng.random((97, 61)).astype(np.float32)


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(9)
    return (rng.random((500, 2)).astype(np.float32) * 100.0,
            rng.random((300, 2)).astype(np.float32) * 100.0)


# -- the windowed maximum --------------------------------------------------

def test_the_windowed_maximum_is_scipys_answer(field):
    """The CPU path IS scipy, so this pins the contract the others meet."""
    from scipy.ndimage import maximum_filter as scipy_filter

    assert np.array_equal(maximum_filter(field, 7, backend="numpy"),
                          scipy_filter(field, size=7, mode="nearest"))


@pytest.mark.parametrize("name", ACCELERATED)
def test_every_backends_windowed_maximum_agrees(field, name):
    _skip_unless(name)
    reference = maximum_filter(field, 7, backend="numpy")
    assert np.allclose(maximum_filter(field, 7, backend=name), reference,
                       atol=1e-6)


def test_an_even_window_is_raised_rather_than_centred_wrongly(field):
    """A window with no centre cannot be centred on a pixel."""
    assert np.array_equal(maximum_filter(field, 6), maximum_filter(field, 7))
    # And a window of one is the field itself, not an error.
    assert np.array_equal(maximum_filter(field, 1), field)
    with pytest.raises(ValueError, match="2-D"):
        maximum_filter(np.zeros((3, 3, 3)), 3)


def test_the_edges_repeat_rather_than_inventing_a_rim(field):
    """Zero-padding would put a rim of false maxima round every field."""
    found = maximum_filter(field, 5)
    assert found[0, 0] >= field[0, 0]
    assert found.shape == field.shape


# -- the matrix multiply ---------------------------------------------------

@pytest.mark.parametrize("name", ACCELERATED)
def test_every_backends_matmul_agrees(name):
    _skip_unless(name)
    rng = np.random.default_rng(1)
    left = rng.random((40, 6)).astype(np.float32)
    right = rng.random((6, 9)).astype(np.float32)
    assert np.allclose(matmul(left, right, backend=name), left @ right,
                       atol=1e-5)


# -- the nearest neighbour -------------------------------------------------

@pytest.mark.parametrize("name", ACCELERATED)
def test_every_backends_nearest_neighbour_agrees(clouds, name):
    _skip_unless(name)
    source, target = clouds
    want_index, want_gap = nearest_neighbours(source, target, backend="numpy")
    got_index, got_gap = nearest_neighbours(source, target, backend=name)
    assert np.array_equal(got_index, want_index), (
        "two backends disagree about which cell is nearest")
    assert np.allclose(got_gap, want_gap, atol=1e-2)


def test_chunking_changes_nothing_but_the_working_set(clouds):
    """The ceiling must not depend on how many cells the well holds."""
    source, target = clouds
    whole = nearest_neighbours(source, target, chunk=len(source))
    in_pieces = nearest_neighbours(source, target, chunk=7)
    assert np.array_equal(whole[0], in_pieces[0])
    assert np.allclose(whole[1], in_pieces[1])
    assert CHUNK >= 1024, "a chunk this small would be all overhead"


def test_an_empty_set_is_an_empty_answer_not_a_crash(clouds):
    source, _target = clouds
    index, gap = nearest_neighbours(source, np.empty((0, 2)))
    assert index.shape == (len(source),) and gap.shape == (len(source),)
    index, gap = nearest_neighbours(np.empty((0, 2)), source)
    assert index.size == 0 and gap.size == 0
    with pytest.raises(ValueError, match="same width"):
        nearest_neighbours(np.zeros((4, 2)), np.zeros((4, 3)))


# -- the fallback ----------------------------------------------------------

def test_gpu_false_refuses_a_card_that_exists():
    """"There is a GPU" and "this run may use it" are different questions."""
    assert accelerated_backends(gpu=False) == ("numpy",)
    assert accelerated_backends(gpu=True)[-1] == "numpy"


def test_a_backend_that_raises_falls_through_but_a_named_one_does_not(field,
                                                                     monkeypatch):
    """An out-of-memory on a shared card is not a failed plate."""
    import spacr.ops_accel as accel

    monkeypatch.setattr(accel, "accelerated_backends",
                        lambda gpu=True: ("torch", "numpy"))
    monkeypatch.setattr(accel, "_torch",
                        lambda gpu=True: (_ for _ in ()).throw(
                            RuntimeError("the card went away")))
    assert np.array_equal(accel.maximum_filter(field, 5),
                          maximum_filter(field, 5, backend="numpy"))
    with pytest.raises(RuntimeError, match="went away"):
        accel.maximum_filter(field, 5, backend="torch")


# -- and the callers -------------------------------------------------------

def test_the_decode_chain_gives_the_same_answer_either_way():
    """`find_peaks` and the unmixing, CPU against accelerated."""
    pytest.importorskip("torch")
    from spacr.ops_sbs import compensate_crosstalk, find_peaks

    rng = np.random.default_rng(12)
    score = rng.random((120, 90)).astype(np.float32)
    assert np.array_equal(find_peaks(score, gpu=False),
                          find_peaks(score, gpu=True))

    values = rng.random((200, 4, 4)).astype(np.float32) * 1000
    assert np.allclose(compensate_crosstalk(values, gpu=False),
                       compensate_crosstalk(values, gpu=True), atol=1e-2)


def test_matching_cells_gives_the_same_pairs_either_way():
    """The pairing decides which phenotype carries which barcode."""
    pytest.importorskip("torch")
    from spacr.ops_merge import match_cells

    rng = np.random.default_rng(21)
    source = rng.random((300, 2)).astype(np.float64) * 500
    target = source[:200] + rng.normal(0, 0.5, (200, 2))
    assert np.array_equal(match_cells(source, target, gpu=False),
                          match_cells(source, target, gpu=True))
