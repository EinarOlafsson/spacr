"""A busy fit's warnings reach the console without consuming one whole GUI turn."""
import logging
import threading

from PySide6.QtCore import QTimer

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_queued_burst_yields_and_preserves_warning_error_and_report_order(
        qtbot, private_ai_profile, monkeypatch):
    from spacr import crashreport
    from spacr.logging_util import _AutomaticErrorHandler
    from spacr.qt.logging_util import QtLogHandler
    from spacr.qt.widgets.console_panel import ConsolePanel

    preferences, _, _ = private_ai_profile
    preferences.set_log_levels(
        (logging.INFO, logging.WARNING, logging.ERROR),
        (logging.INFO, logging.WARNING, logging.ERROR))
    panel = ConsolePanel(follow_log=False, chat=False)
    qtbot.addWidget(panel)
    panel.show()
    handler = QtLogHandler()
    handler.records_ready.connect(panel._on_log_records)
    received, turns, dispatched = [], [], []

    def interactive_turn():
        turns.append((len(received), panel.as_text()))
        logger.warning('GUI_FOLLOWUP_AFTER_BACKLOG')

    def delivered(batch):
        received.extend(batch)
        if len(received) == len(batch):
            QTimer.singleShot(0, interactive_turn)

    handler.records_ready.connect(delivered)
    monkeypatch.setattr(crashreport, '_dispatch_error',
                        lambda text, **kwargs: dispatched.append(text))
    logger = logging.Logger('spacr.fit_warning_burst', logging.INFO)
    logger.addHandler(_AutomaticErrorHandler(logging.ERROR))
    logger.addHandler(handler)
    records = [(logging.WARNING, f'FEATURE_NAME_WARNING_{i:03d}') for i in range(96)]
    records.insert(35, (logging.ERROR, 'FIT_ERROR_A'))
    records.insert(72, (logging.ERROR, 'FIT_ERROR_B'))

    def produce():
        for level, text in records:
            logger.log(level, text)

    worker = threading.Thread(target=produce)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    try:
        qtbot.waitUntil(lambda: len(received) == len(records) + 1, timeout=10000)
        qtbot.waitUntil(lambda: bool(turns), timeout=1000)
        assert 0 < turns[0][0] < len(records), 'the first GUI callback ran only after the entire burst'
        assert 'FEATURE_NAME_WARNING_000' in turns[0][1]
        assert 'FEATURE_NAME_WARNING_095' not in turns[0][1]
        assert [(level, text.rsplit(': ', 1)[-1].strip()) for text, level in received] == records + [(logging.WARNING, 'GUI_FOLLOWUP_AFTER_BACKLOG')]
        visible = panel.as_text()
        positions = [visible.index(text) for _, text in records]
        assert positions == sorted(positions)
        assert visible.index('GUI_FOLLOWUP_AFTER_BACKLOG') > positions[-1]
        assert len(dispatched) == 2
        assert 'FIT_ERROR_A' in dispatched[0] and 'FIT_ERROR_B' in dispatched[1]
    finally:
        logger.handlers.clear()
        handler.close()
        panel.close()
