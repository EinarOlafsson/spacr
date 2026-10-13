"""What the animation measurements do when the imaging stack is not there.

Every measurement in :mod:`spacr.setting_animations` decodes a GIF, and
spaCR is installed in places where NumPy or Pillow is not -- a docs build, a
minimal wheel check, a headless packaging job. Each measurement therefore
answers with the value that keeps its own validator quiet rather than with a
traceback, and the direction differs per function: "shows something" assumes
a full frame changed, the border checks assume none of it did. Both mean
"do not report a failure you could not measure", and neither is observable
unless the import is actually made to fail.

The decoder is exercised here for the same reason. It was shadowed by a
second definition of the same name until this file's fix, so its guard had
never run in any test, in any release.
"""
from __future__ import annotations

import builtins

import numpy as np
import pytest
from PIL import Image

from spacr import setting_animations as SA


def _gif(path, frames):
    """Write ``frames`` (H, W, 3 arrays) as an animated GIF, returning its path."""
    images = [Image.fromarray(frame.astype(np.uint8), "RGB") for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], loop=0)
    return str(path)


def _blocked(monkeypatch, *names):
    """Make ``import <name>`` raise ImportError for each of ``names``."""
    real_import = builtins.__import__
    blocked = set(names)

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"No module named {name.split('.')[0]!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _still(tmp_path, name="still.gif", size=24):
    """A GIF whose two frames are identical: it really does show nothing."""
    frame = np.zeros((size, size, 3))
    return _gif(tmp_path / name, [frame, frame.copy()])


def _flashing_border(tmp_path, name="ring.gif", size=24):
    """A GIF whose entire frame border turns white on the second frame."""
    first = np.zeros((size, size, 3))
    second = np.full((size, size, 3), 255.0)
    second[SA._BORDER_BAND:-SA._BORDER_BAND,
           SA._BORDER_BAND:-SA._BORDER_BAND] = 0.0
    return _gif(tmp_path / name, [first, second])


# ---------------------------------------------------------------------------
# measure_visible_change without an imaging stack
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["numpy", "PIL"])
def test_an_unmeasurable_animation_is_credited_with_a_full_frame_of_change(
        tmp_path, monkeypatch, missing):
    """Without imaging the measurement claims 1.0, not 0.0.

    The number is chosen so :func:`validate_animations_show_something`
    reports nothing on an installation that cannot decode a GIF. Returning
    0.0 there would fail every one of the 94 packaged animations on a
    machine whose only fault is a missing dependency.
    """
    still = _still(tmp_path)
    assert SA.measure_visible_change(still) == 0.0

    _blocked(monkeypatch, missing)
    assert SA.measure_visible_change(still) == 1.0


def test_the_change_measurement_gives_up_before_it_touches_the_file(
        tmp_path, monkeypatch):
    """The imports are checked first, so a path that does not exist is 1.0.

    An unreadable file measures 0.0; a file that was never opened because
    there was nothing to open it with measures 1.0. The two answers are the
    ones that distinguish the branches.
    """
    absent = tmp_path / "never_generated.gif"
    assert SA.measure_visible_change(absent) == 0.0

    _blocked(monkeypatch, "PIL")
    assert SA.measure_visible_change(absent) == 1.0


def test_no_animation_is_reported_as_blank_when_imaging_is_missing(
        tmp_path, monkeypatch):
    """The whole point of the 1.0: the shipped validator stays silent."""
    _blocked(monkeypatch, "numpy", "PIL")
    assert SA.validate_animations_show_something() == {}


# ---------------------------------------------------------------------------
# the decoder itself
# ---------------------------------------------------------------------------

def test_the_decoder_returns_one_int16_rgb_array_per_stored_frame(tmp_path):
    """Frames come back in order, as signed arrays a difference can use.

    ``int16`` is load-bearing rather than incidental: the callers subtract
    two frames, and on ``uint8`` a black pixel minus a white one wraps to
    241 instead of -255.
    """
    black = np.zeros((12, 10, 3))
    grey = np.full((12, 10, 3), 128.0)
    white = np.full((12, 10, 3), 255.0)
    frames = SA._animation_frames(_gif(tmp_path / "three.gif",
                                       [black, grey, white]))

    assert len(frames) == 3
    assert [frame.shape for frame in frames] == [(12, 10, 3)] * 3
    assert all(frame.dtype == np.int16 for frame in frames)
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 128, 255]
    assert int((frames[0] - frames[2])[0, 0, 0]) == -255


def test_the_decoder_returns_an_empty_list_for_a_file_that_is_not_a_gif(
        tmp_path):
    """A file Pillow refuses is "no frames", which callers read as 0.0."""
    broken = tmp_path / "text.gif"
    broken.write_bytes(b"this is not a GIF")
    assert SA._animation_frames(broken) == []


def test_the_decoder_returns_none_rather_than_an_empty_list_without_pillow(
        tmp_path, monkeypatch):
    """``None`` means "not measured"; ``[]`` means "measured, no frames".

    Both are falsy, so both callers behave identically, but only the
    distinction tells a maintainer whether the GIF or the environment is at
    fault. This contract was unreachable while a second definition of
    ``_animation_frames`` shadowed this one.
    """
    readable = _still(tmp_path)
    assert SA._animation_frames(readable) != []

    _blocked(monkeypatch, "PIL")
    assert SA._animation_frames(readable) is None


# ---------------------------------------------------------------------------
# measure_border_artifact without an imaging stack
# ---------------------------------------------------------------------------

def test_a_ring_that_flashes_is_measured_as_the_whole_border(tmp_path):
    """The baseline the missing-imports cases are contrasted against."""
    assert SA.measure_border_artifact(_flashing_border(tmp_path)) == 1.0


def test_an_unmeasurable_border_reports_no_artifact_rather_than_raising(
        tmp_path, monkeypatch):
    """Without NumPy the border check answers 0.0 for a 1.0 animation.

    The direction is the opposite of the change measurement on purpose:
    this validator fails animations ABOVE its threshold, so the quiet
    answer is the floor rather than the ceiling.
    """
    ring = _flashing_border(tmp_path)
    _blocked(monkeypatch, "numpy")
    assert SA.measure_border_artifact(ring) == 0.0


def test_a_border_measured_without_pillow_reports_no_artifact(
        tmp_path, monkeypatch):
    """NumPy present, Pillow absent: the undecodable frames are not a crash.

    This is the path that reaches the border check through the decoder's
    ``None`` rather than through its own import guard.
    """
    ring = _flashing_border(tmp_path)
    _blocked(monkeypatch, "PIL")
    assert SA.measure_border_artifact(ring) == 0.0


def test_no_animation_is_reported_as_flashing_when_imaging_is_missing(
        monkeypatch):
    """The validator stays silent on an installation that cannot decode."""
    _blocked(monkeypatch, "numpy")
    assert SA.validate_animations_have_no_border_artifact() == {}


# ---------------------------------------------------------------------------
# measure_border_object_removal, which errors instead of answering
# ---------------------------------------------------------------------------

def test_a_single_frame_border_animation_is_refused_as_incomparable(tmp_path):
    """One frame is a picture; the measurement needs a before and an after."""
    only_one = _gif(tmp_path / "one.gif", [np.zeros((24, 24, 3))])
    with pytest.raises(SA.SettingAnimationError,
                       match="not enough frames to compare"):
        SA.measure_border_object_removal(only_one)


def test_a_border_animation_that_cannot_be_decoded_raises_the_module_error(
        tmp_path, monkeypatch):
    """Without Pillow the measurement raises its own error, not ``TypeError``.

    The decoder answers ``None`` here, and ``len(None)`` is a ``TypeError``
    that names neither the file nor the reason. A caller validating the
    shipped border animations is told which animation it could not measure.
    """
    ring = _flashing_border(tmp_path)
    _blocked(monkeypatch, "PIL")
    with pytest.raises(SA.SettingAnimationError,
                       match="not enough frames to compare"):
        SA.measure_border_object_removal(ring)
