"""What a run says about the hardware it ended up on, and the half-precision
block it runs the forward pass in.

The card is shared. :func:`spacr.deep_spacr.pick_device` answers "can this
run use the GPU", not "does a GPU exist", so a run started while somebody
else holds most of the card silently continues on the CPU -- ten to a hundred
times slower. The note it returns is the ONLY notice a user gets of that, so
every entry point that picks a device has to print it, and print it before it
starts working rather than after the run is already slow.

Two smaller pieces of the same block are covered here for the same reason:
``autocasting`` has to be a context manager whether or not half precision was
asked for, since the training loop has one shape rather than a branch around
every forward pass; and releasing the accelerator cache must not touch a
backend that is not there.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt             # noqa: E402

from spacr import deep_spacr                # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# autocasting / cache release
# ---------------------------------------------------------------------------

def _cpu_autocast_enabled():
    """Read CPU autocast state across the supported Torch 2.x range."""
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled()


def _cpu_autocast_dtype():
    """Read the CPU autocast dtype before Torch exposed the generic getter."""
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is not None:
        try:
            return getter("cpu")
        except TypeError:
            pass
    return torch.get_autocast_cpu_dtype()

def test_autocasting_on_actually_enables_half_precision():
    """The training loop's ``with`` block has to be the real autocast block."""
    device = torch.device("cpu")

    with deep_spacr.autocasting(True, device):
        inside = _cpu_autocast_enabled()
        dtype = _cpu_autocast_dtype()

    assert inside is True
    assert dtype is torch.bfloat16
    assert _cpu_autocast_enabled() is False


def test_autocasting_off_is_still_a_context_manager():
    """One shape for the loop: off must not mean "no ``with`` block"."""
    with deep_spacr.autocasting(False, torch.device("cpu")):
        assert _cpu_autocast_enabled() is False


def test_the_cache_release_reaches_cuda_when_cuda_is_there(monkeypatch):
    """A CUDA run has to get its cache back between stages."""
    released = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda: released.append("cuda"))

    deep_spacr._empty_device_cache()

    assert released == ["cuda"]


def test_the_cache_release_leaves_an_absent_backend_alone(monkeypatch):
    """Asking a backend that is not there is how a CPU run crashes."""
    def must_not_run():
        raise AssertionError("empty_cache was called with no CUDA present")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", must_not_run)

    assert deep_spacr._empty_device_cache() is None
