"""A shading thread must not outlive the test that started it.

Instruction 288. A 360-file prefix run wedged on a 120 s timeout with NINE
threads named ``spacr-ambient-shade`` alive at once, most inside numpy work in
``_paint_field``. A CPU load that grows with the number of tests already run is
the shape 288 was looking for: no single test file segfaults on its own, and
the position of the failure moves between runs.

Production does not leak, and that claim is measured here rather than assumed
-- see :func:`test_destroying_a_shown_widget_stops_its_thread`. What leaks is a
test-shaped lifetime, and :func:`test_the_leak_this_guards_against_is_real`
builds one on purpose so the guard is never asserting against nothing.
"""

import gc
import threading

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt.widgets.ambient import AmbientWidget


def _shading_threads():
    return [t for t in threading.enumerate() if t.name == "spacr-ambient-shade"]


def _settle(qapp, turns=25):
    for _ in range(turns):
        qapp.processEvents()


@pytest.fixture
def running_backdrop(qapp):
    """A shown widget with its shading thread actually running.

    Yields ``(host, widget)``. The caller is expected to leave the widget in
    whatever state it is testing; the fixture stops the thread on the way out
    so a failure here cannot itself leak into the rest of the session.
    """
    host = QWidget()
    host.resize(320, 240)
    widget = AmbientWidget(host)
    host.show()
    _settle(qapp)
    if not _shading_threads():
        pytest.skip("this engine does not put its shading on a thread here")
    yield host, widget
    try:
        widget.stop()
    except RuntimeError:
        pass
    host.hide()


def test_the_leak_this_guards_against_is_real(qapp, running_backdrop):
    """Dropping the last Python reference does NOT stop the thread.

    Without this the guard below could pass in a world where nothing ever
    leaked, which is the failure mode 288 keeps running into: an absence
    asserted is worth nothing until it has been watched to fail.
    """
    host, widget = running_backdrop
    assert len(_shading_threads()) == 1

    del widget
    gc.collect()
    _settle(qapp)

    # Still shading: the C++ half is alive as a child of `host`, so `destroyed`
    # never fires and `hideEvent` never runs.
    assert len(_shading_threads()) == 1, (
        "the leak this module guards against no longer reproduces -- if "
        "ambient.py learned to stop on Python collection, delete the guard "
        "rather than leaving a test that proves nothing"
    )


def test_stopping_every_live_ambient_widget_clears_the_leak(qapp,
                                                            running_backdrop):
    """The conftest fixture's mechanism, exercised directly.

    ``_no_backdrop_threads_survive_a_test`` walks ``QApplication.allWidgets``
    and calls the public :meth:`AmbientWidget.stop` on each ambient widget it
    finds. This is that walk, against a leak built on purpose.
    """
    host, widget = running_backdrop
    del widget
    gc.collect()
    _settle(qapp)
    assert len(_shading_threads()) == 1

    for candidate in QApplication.allWidgets():
        if isinstance(candidate, AmbientWidget):
            candidate.stop()
    _settle(qapp)

    assert _shading_threads() == []


def test_destroying_a_shown_widget_stops_its_thread(qapp, running_backdrop):
    """The production path, which is why the guard lives in the test tree.

    ``AmbientWidget.__init__`` connects ``destroyed`` to ``_retire_producer``
    through a closure holding only the one-element producer box, so closing a
    window stops and joins the thread. Nothing in spacr/ needs fixing.
    """
    from shiboken6 import delete

    host, widget = running_backdrop
    assert len(_shading_threads()) == 1

    delete(widget)
    _settle(qapp)

    assert _shading_threads() == []
