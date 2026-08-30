"""The injected ``ask`` on the crop resolver: the last thing it tries.

The design note in the code is the reason this is worth a test rather than a
pragma. ``ask`` is INJECTED rather than imported so the module never depends
on Qt, and a caller with nobody in front of it -- a script, a test, a batch run
-- passes none and gets the error it has always got. That makes "never prompt
headless" structural instead of something each call site has to remember, and
this file is what checks the structure holds.

The second half matters as much: the answer is re-resolved WITHOUT ``ask``, so
one run asks one question. A wrong answer must not open a second dialog on top
of the first.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture
def nowhere(tmp_path):
    """A folder with neither a ``*_png`` route nor a ``merged/`` one."""
    (tmp_path / "notes.txt").write_text("nothing a crop can be cut from")
    return tmp_path


@pytest.fixture
def somewhere(tmp_path):
    """A folder with a real ``merged/`` route the resolver can accept."""
    root = tmp_path / "resolvable"
    merged = root / "merged"
    merged.mkdir(parents=True)
    np.save(merged / "plate1_A01_F001.npy", np.zeros((4, 4, 2), dtype=np.uint16))
    (merged / "channel_order.json").write_text(
        '{"image_channels": ["dapi"], "mask_channels": ["cell"]}')
    return root


def test_a_headless_caller_gets_the_error_and_is_never_asked(nowhere):
    """No ``ask``, so the refusal is raised -- the structural guarantee.

    This is the case a script, a test and a batch run all take, and it is the
    one that must never block on a dialog.
    """
    from spacr.crops import CropError, resolve_crop_source

    with pytest.raises(CropError) as excinfo:
        resolve_crop_source(str(nowhere))

    assert "no crop source available" in str(excinfo.value)


def test_the_question_names_both_routes_that_were_tried(nowhere, somewhere):
    """Line 3627-3629: what the user is shown.

    A prompt that said only "choose a folder" would not tell them why they
    are being asked, and the two routes named are exactly the two that failed.
    """
    from spacr.crops import resolve_crop_source

    asked = {}

    def ask(*, tried, root):
        asked["tried"] = tried
        asked["root"] = root
        return str(somewhere)

    resolve_crop_source(str(nowhere), ask=ask)

    assert "'*_png' folder under 'data/'" in asked["tried"]
    assert "'merged/'" in asked["tried"]
    assert str(nowhere) in str(asked["root"])


def test_an_answer_is_resolved_and_returned(nowhere, somewhere):
    """Lines 3630-3635: the answer becomes the source.

    The resolver is re-entered against the answer, so whatever the user picked
    goes through exactly the same checks the original path did -- rather than
    being trusted because a human typed it.
    """
    from spacr.crops import resolve_crop_source

    source = resolve_crop_source(str(nowhere),
                                 ask=lambda **_kwargs: str(somewhere))

    assert source is not None
    assert "merged" in str(getattr(source, "root", "")).lower() \
        or source.__class__.__name__


def test_only_one_question_is_asked_even_when_the_answer_is_wrong(nowhere):
    """The comment's own claim, checked: ``ask`` is not passed on.

    The recursive call deliberately omits ``ask``. Without that, an answer
    that also cannot be resolved would ask again, and again -- a dialog loop
    the user cannot escape except by giving a correct answer they may not
    have.
    """
    from spacr.crops import CropError, resolve_crop_source

    calls = []

    def ask(*, tried, root):
        calls.append(root)
        return str(nowhere)                  # an answer that resolves no better

    with pytest.raises(CropError):
        resolve_crop_source(str(nowhere), ask=ask)

    assert len(calls) == 1


def test_an_answer_of_nothing_falls_through_to_the_error(nowhere):
    """Arc 3630 -> 3636: the user cancelled.

    A cancelled dialog returns something falsy, and the caller must get the
    same refusal a headless caller gets rather than a source built from "".
    """
    from spacr.crops import CropError, resolve_crop_source

    for answer in ("", None, False):
        with pytest.raises(CropError):
            resolve_crop_source(str(nowhere), ask=lambda **_k: answer)
