"""The v2 pipeline does not score masks, and says so.

`seg_qc` defaults to 'report', so a v1 run scores every mask through
`object._run_seg_qc`. v2 cannot reuse that: v1 writes a
`<object_type>_mask_stack` FOLDER, which the scorecard globs, while v2
appends the mask as extra CHANNELS of `merged/stack_<field>.npy`. The
setting was accepted and silently did nothing, which is the failure this
guards -- a user who believes a bad plate was checked finds out in the
measurements instead.
"""

import pytest

from spacr.core import _say_v2_does_not_score_masks as announce


def test_off_says_nothing(capsys):
    """QC turned off is not a surprise, so it is not worth a line."""
    assert announce({"seg_qc": "off"}) is False
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("mode", ["report", "flag"])
def test_a_requested_mode_is_announced(capsys, mode):
    assert announce({"seg_qc": mode}) is True
    said = capsys.readouterr().out
    assert "NOT scored on the v2 pipeline" in said


def test_the_default_is_announced_too(capsys):
    """`seg_qc` defaults to 'report', so a user who set nothing still
    expects a scorecard and still does not get one."""
    assert announce({}) is True
    assert "NOT scored" in capsys.readouterr().out


def test_it_says_why_rather_than_only_that(capsys):
    """The reason is the mask LAYOUT, and a reader who knows that can tell
    this is not a bug they should report."""
    announce({"seg_qc": "report"})
    said = capsys.readouterr().out
    assert "merged/stack_<field>.npy" in said
    assert "mask_stack" in said


def test_it_says_the_masks_are_fine(capsys):
    """Unscored is not damaged, and the difference matters to someone
    deciding whether to re-run a plate."""
    announce({"seg_qc": "report"})
    assert "masks are unaffected" in capsys.readouterr().out.lower()


def test_it_names_the_way_to_get_the_scorecard(capsys):
    announce({"seg_qc": "report"})
    assert "pipeline_style='v1'" in capsys.readouterr().out


def test_the_v2_branch_calls_it():
    """Wired, not merely defined -- the check that would have caught the
    original bug, which was a function nothing called."""
    import inspect

    from spacr import core

    body = inspect.getsource(core.preprocess_generate_masks)
    assert "_say_v2_does_not_score_masks(settings)" in body


def test_v2_still_does_not_score_masks():
    """If someone wires seg_qc into v2, this fails and the announcement
    above should be deleted rather than left lying."""
    import spacr.pipeline_v2 as v2
    import inspect

    assert "seg_qc" not in inspect.getsource(v2)
