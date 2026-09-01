"""A join that never starts must not leave the button saying Cancel.

Instruction 288. ``join_the_tables`` flips ``_joining`` and turns the
join button into a Cancel BEFORE it submits, because the submit is the
thing that takes minutes. If the submit is refused, both have to be put
back or the panel is stuck: the button offers to cancel a job that is
not running, and ``_joining`` being True makes every later press return
early without doing anything.

The arm was marked ``# pragma: no cover - JobRunner always returns True
today``, and that reason was WRONG. JobRunner returns False whenever it
is unthreaded and either the work or the completion callback raises --
see ``JobRunner.submit``. It is only this panel's runner, built threaded,
that cannot reach it. So the guard is correct code guarding a real
contract, not defensive noise, and it is driven here rather than
deleted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.measurement_compare_dialog import (
    MeasurementComparePanel,
)


def _objects(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "cell_area": rng.normal(10.0, 2.0, n),
        "plateID": np.repeat(["p1", "p2"], n // 2),
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2"], n),
        "fieldID": rng.choice(["f1", "f2"], n),
        "object_label": np.arange(n),
    })


class _RefusingRunner:
    """A runner that declines the job, the way an unthreaded one does.

    Not a fiction: ``JobRunner.submit`` returns False on exactly this
    path. It stands in for the runner rather than for the panel, so the
    panel's own code runs unmodified.
    """

    def __init__(self):
        self.asked = 0

    def submit(self, _fn, _on_done=None) -> bool:
        self.asked += 1
        return False

    def cancel(self):
        pass


@pytest.fixture
def panel(qtbot, tmp_path):
    objects = _objects()
    widget = MeasurementComparePanel(objects, {"g": list(objects.index)},
                                     databases=[str(tmp_path / "a.db")])
    qtbot.addWidget(widget)
    return widget


def test_a_refused_join_restores_the_button_and_the_flag(panel):
    """The arm itself."""
    runner = _RefusingRunner()
    panel._jobs = runner

    assert panel.join_the_tables() == ""

    assert runner.asked == 1, "the panel never tried to submit"
    assert panel._joining is False, (
        "a refused join left the panel believing one is running; every "
        "later press returns early and the join can never be started")
    assert panel.join_button.text() != panel.CANCEL_LABEL, (
        "the button still offers to cancel a job that is not running")


def test_the_panel_can_still_join_after_a_refusal(panel):
    """WHY the flag matters, not just that it is False.

    Asserting `_joining is False` alone would pass against a version that
    reset the flag and left the panel broken some other way. This drives
    the consequence: a second press must reach the runner again.
    """
    refusing = _RefusingRunner()
    panel._jobs = refusing
    panel.join_the_tables()

    accepting = _RefusingRunner()
    accepting.submit = lambda _fn, _on_done=None: (  # accepted this time
        setattr(accepting, "asked", accepting.asked + 1) or True)
    panel._jobs = accepting
    panel.join_the_tables()
    assert accepting.asked == 1, (
        "the second join never reached the runner -- the panel was still "
        "stuck on the first")


def test_a_started_join_does_keep_the_cancel_label(panel):
    """The OTHER side, so the test above is not passing vacuously.

    If the button were simply never set to Cancel, the refusal test would
    pass for the wrong reason. This pins that a job which IS accepted
    leaves the panel in the running state.
    """
    class _Accepting(_RefusingRunner):
        def submit(self, _fn, _on_done=None) -> bool:
            self.asked += 1
            return True

    panel._jobs = _Accepting()
    panel.join_the_tables()
    assert panel._joining is True
    assert panel.join_button.text() == panel.CANCEL_LABEL


def test_job_runner_really_can_refuse_which_is_what_this_rests_on():
    """THE PREMISE. The pragma claimed submit always returns True.

    It does not: unthreaded, a callback that raises makes it return
    False. If that ever stopped being so, the guard would be genuinely
    unreachable and this file should go with it -- so the contract is
    pinned here rather than assumed.
    """
    from spacr.qt.job_runner import JobRunner

    runner = JobRunner(threaded=False, app_key="test", user_visible=False)

    def boom(_result):
        raise RuntimeError("the completion callback failed")

    assert runner.submit(lambda: {"ok": True}, boom) is False
    assert runner.submit(lambda: {"ok": True}, lambda _r: None) is True
