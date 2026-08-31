"""The macOS libpyside slot warning is dropped, and nothing else is.

Opening any module on macOS printed a dozen lines of

    libpyside: addMetaMethod: Cannot add dynamic method "_on_tick()" (2)
    to QWidget/0x7feab2c18060: No Wrapper found.

once per screen built. What it reports is that PySide could not register
a dynamic slot on the receiver -- the optimisation that makes a
connection die with its receiver. The connection is still made and still
fires, which is why every affected module works.

It is filtered rather than fixed, and that distinction is the subject of
these tests. Four hypotheses were killed on the reporting Mac before the
filter went in (a PySide version change, connect-in-__init__, widget
parenting, and a dangling connection), and
``tools/diagnose_pyside_slot_warning.py`` is kept so the question can be
re-opened in one command.

The risk of filtering is that the pattern is too greedy and swallows a
warning that matters. That is what most of this file is about.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import _QT_NOISE

#: Verbatim from the macOS report, both the anonymous widget and the
#: named ones. Kept exact rather than reconstructed: the filter has to
#: match what PySide actually prints.
REPORTED = (
    'libpyside: addMetaMethod: Cannot add dynamic method "_on_tick()" (2) '
    'to QWidget/0x7feab2c18060: No Wrapper found.',
    'libpyside: addMetaMethod: Cannot add dynamic method '
    '"_on_use_offered()" (2) to QWidget/"ChainingBar": No Wrapper found.',
    'libpyside: addMetaMethod: Cannot add dynamic method "refresh()" (2) '
    'to QWidget/"MeasureQCBanner": No Wrapper found.',
    'libpyside: addMetaMethod: Cannot add dynamic method '
    '"_on_measure_clicked()" (2) to QWidget/"DiameterPanel": '
    'No Wrapper found.',
)

#: Warnings this project has actually needed to read. Every one must
#: survive the filter. The first two are not hypothetical -- the thread
#: affinity line arrives immediately before a crash and is singled out
#: for a Python stack in the handler itself.
MUST_SURVIVE = (
    "QBasicTimer::start: Timers cannot be started from another thread",
    "QObject: Cannot create children for a parent that is in a different "
    "thread.",
    "QPixmap::scaled: Pixmap is a null pixmap",
    "Could not parse stylesheet of object QWidget",
    "QOpenGLShaderProgram: could not create shader program",
    "libpyside: Invalid return value in function Foo, expected bar",
)


@pytest.mark.parametrize("message", REPORTED)
def test_every_reported_line_is_filtered(message):
    """All four shapes it was seen in, not just the one in the docstring."""
    assert _QT_NOISE.search(message), f"still printed: {message}"


@pytest.mark.parametrize("message", MUST_SURVIVE)
def test_a_warning_that_matters_still_gets_through(message):
    """The filter is a scalpel, not a mute.

    A pattern broad enough to catch "No Wrapper found" anywhere, or any
    line beginning "libpyside:", would take the thread-affinity warning
    with it -- and that one arrives immediately before a crash.
    """
    assert not _QT_NOISE.search(message), f"wrongly swallowed: {message}"


def test_the_filter_is_specific_to_this_one_pyside_message():
    """Both halves of the signature are required, not either.

    Written because the obvious pattern -- "No Wrapper found" on its own
    -- reads as sufficient and is not: it is generic PySide vocabulary
    that other, unrelated failures also use.
    """
    assert not _QT_NOISE.search("No Wrapper found."), (
        "the filter keys on a phrase generic enough to hide other faults")
    assert not _QT_NOISE.search("libpyside: addMetaMethod: something else")


def test_the_diagnostic_that_justified_the_filter_is_still_here():
    """The evidence stays with the decision.

    A filter for an unexplained warning is only defensible while the
    means of re-opening it survives. If this tool is ever deleted, the
    comment in ``spacr/qt/__init__.py`` becomes an unfalsifiable claim.
    """
    from pathlib import Path

    tool = (Path(__file__).resolve().parents[2]
            / "tools" / "diagnose_pyside_slot_warning.py")
    assert tool.is_file(), (
        "the diagnostic behind the filter is gone; either restore it or "
        "remove the filter, because the reasoning no longer stands up")
    body = tool.read_text()
    assert "addMetaMethod" in body
