"""Instruction 320 (1) and (2): one console line per Qt message, at its level.

Two application-wide console sinks exist and every ConsolePanel subscribes to
both -- ``verbose_logger``'s forwarder on the ``spacr`` logger, and
``qt.logging_util``'s ``QtLogHandler`` on the ROOT. A record from ``spacr.qt``
walks spacr -> root, so it was rendered twice in two different formats. That is
what macOS reported, a dozen PySide warnings each appearing twice.

Neither copy is the raw stderr ``print`` in ``_install_quiet_qt_logging``, and
that is the part worth pinning: both are FORMATTED records, so anyone who reads
the print statement looking for the duplicate finds nothing.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def sinks(qapp):
    from spacr.qt import logging_util as qt_log, verbose_logger

    qt_log.setup_logging()
    forwarder = verbose_logger._ensure_handler()
    yield qt_log, verbose_logger, forwarder


def _renders(qt_log, forwarder, level, message):
    """Every console rendering of one record, from both sinks."""
    out = []
    qt_log.get_signal_handler().record_ready.connect(
        lambda text, lvl: out.append(("root", text)))
    original = forwarder.emit
    forwarder.emit = lambda rec: out.append(("verbose", forwarder.format(rec)))
    try:
        logging.getLogger("spacr.qt").log(level, "%s", message)
    finally:
        forwarder.emit = original
    return out


def test_a_qt_warning_is_rendered_once_not_twice(sinks):
    """The duplicate itself."""
    qt_log, _verbose, forwarder = sinks
    out = _renders(qt_log, forwarder, logging.WARNING,
                   'Qt warning: libpyside: addMetaMethod: Cannot add dynamic '
                   'method "_on_tick()" (2) to QWidget: No Wrapper found.')
    assert len(out) == 1, (
        "one Qt warning must produce one console line, not one per sink:\n  "
        + "\n  ".join(f"[{who}] {text}" for who, text in out))


def test_verbose_only_detail_below_the_root_sink_still_arrives(sinks):
    """The de-duplication must not cost verbose mode its reason to exist.

    Verbose logging is for the records the ordinary sink filters out. Dropping
    those too would make this a mute rather than a de-duplication.
    """
    qt_log, _verbose, forwarder = sinks
    qt_log.get_signal_handler().setLevel(logging.INFO)
    logging.getLogger("spacr.qt").setLevel(logging.DEBUG)
    out = _renders(qt_log, forwarder, logging.DEBUG, "a trace-level detail")
    assert [who for who, _ in out] == ["verbose"], (
        f"below the root sink's level only verbose should render: {out}")


def test_a_warning_is_not_classified_as_an_error(qapp):
    """320 (2): a QtWarningMsg must not land in the ERROR pane.

    The Qt message handler maps QtWarningMsg to logging.WARNING correctly, so
    the mislabel is downstream. It is NOT the "anything on stderr is an error"
    heuristic the report guessed at -- ``ConsolePanel._on_log_record`` routes
    every record at or above WARNING to ``append_error``, which draws the red
    "spaCR ERROR" banner. A pane that cries error over routine noise is one
    people stop reading, and the next line in it might be the one that matters.
    """
    from spacr.qt.widgets.console_panel import ConsolePanel

    panel = ConsolePanel()
    routed = []
    panel.append_error = lambda text: routed.append(("error", text))
    panel.append_warning = lambda text: routed.append(("warning", text))
    panel.append_stdout = lambda text: routed.append(("stdout", text))

    panel._on_log_record("chatter", logging.INFO)
    panel._on_log_record("Qt warning: something routine", logging.WARNING)
    panel._on_log_record("a real failure", logging.ERROR)
    panel._on_log_record("a fatal one", logging.CRITICAL)

    assert [kind for kind, _ in routed] == [
        "stdout", "warning", "error", "error"], (
        f"each level must reach its own band, not two: {routed}")
