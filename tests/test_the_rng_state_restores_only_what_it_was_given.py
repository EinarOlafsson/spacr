"""Restoring random-generator state from a partial or absent capture.

``restore_rng_state`` is what makes a resumed run continue the same sequence
rather than starting a new one, and every guard in it is about a capture that
does not carry all three generators. A checkpoint written by an older spaCR, or
one written with ``include_rng=False``, is exactly that -- and restoring a
missing key would raise inside resume, turning a recoverable run into a lost one.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _draws():
    """One draw from each generator, as a fingerprint of their state."""
    return (random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()))


def test_a_full_capture_restores_all_three_generators():
    """The baseline: the same sequence continues after a restore."""
    from spacr.torch_artifacts import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    expected = _draws()

    restore_rng_state(state)

    assert _draws() == expected


@pytest.mark.parametrize("keep", ["python", "numpy", "torch"])
def test_a_capture_carrying_one_generator_restores_only_that_one(keep):
    """Arcs 69 -> 71, 71 -> 73 and 73 -> 75, one absent key at a time.

    A partial capture must restore what it has and leave the rest running. The
    alternative is a KeyError or a None handed to ``setstate`` deep inside a
    resume, which loses a run that was otherwise recoverable.
    """
    from spacr.torch_artifacts import capture_rng_state, restore_rng_state

    full = capture_rng_state()
    partial = {keep: full[keep]}

    restore_rng_state(partial)                 # must not raise

    # The kept generator really was rewound.
    if keep == "python":
        assert random.random() == pytest.approx(
            _restore_and_draw(full, "python"))


def _restore_and_draw(full, which):
    from spacr.torch_artifacts import restore_rng_state

    restore_rng_state({which: full[which]})
    return random.random()


def _live_state():
    """Every generator's state, as comparable values."""
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return (random.getstate(),
            (numpy_state[0], tuple(numpy_state[1]), *numpy_state[2:]),
            torch.get_rng_state().clone().tolist())


def test_a_capture_with_none_for_a_generator_leaves_it_running():
    """The guards test ``is not None``, not mere presence.

    A payload round-tripped through JSON or a partially written checkpoint can
    carry the key with a null value, and ``random.setstate(None)`` raises. But
    not raising is only half the contract: skipping has to leave the live
    generator exactly where it was, because a run resuming from such a
    checkpoint keeps drawing from it.
    """
    from spacr.torch_artifacts import restore_rng_state

    before = _live_state()

    restore_rng_state({"python": None, "numpy": None, "torch": None})

    assert _live_state() == before


@pytest.mark.parametrize("state", [None, {}])
def test_no_capture_at_all_leaves_every_generator_untouched(state):
    """The early return: include_rng=False writes no rng_state.

    Resuming such a checkpoint is legitimate -- the user asked not to carry
    generator state -- so this must be silence rather than a complaint, and
    the generators must keep the state the process already had.
    """
    from spacr.torch_artifacts import restore_rng_state

    before = _live_state()

    restore_rng_state(state)

    assert _live_state() == before


# ---------------------------------------------------------------------------
# restore_training_state — the pieces a payload may not carry
# ---------------------------------------------------------------------------

def test_a_payload_with_no_optimizer_or_scheduler_still_returns_its_state():
    """Arcs 331 -> 333 and 334 -> 336: nothing to load into, nothing loaded.

    A checkpoint saved for inference carries neither, and resuming it for
    evaluation is an ordinary thing to do.
    """
    from spacr.torch_artifacts import restore_training_state

    out = restore_training_state({"training_state": {"epoch": 7}})

    assert out == {"epoch": 7}


def test_a_legacy_payload_with_no_training_state_returns_an_empty_dict():
    """The documented answer for a legacy full-module checkpoint.

    ``{}`` rather than None, so the caller can read ``out.get('epoch')``
    without checking first.
    """
    from spacr.torch_artifacts import restore_training_state

    assert restore_training_state({}) == {}
    assert restore_training_state({"training_state": None}) == {}


def test_generators_are_left_alone_when_the_caller_says_so():
    """Arc 336 -> 338: ``restore_random_generators=False``.

    Evaluating a checkpoint must not rewind the caller's generators -- a
    metric computed with a rewound RNG would differ from the same metric
    computed a moment earlier.
    """
    from spacr.torch_artifacts import capture_rng_state, restore_training_state

    payload = {"training_state": {"epoch": 1},
               "rng_state": capture_rng_state()}

    before = random.random()
    restore_training_state(payload, restore_random_generators=False)
    after = random.random()

    assert before != after, "the generator was rewound when it should not be"
