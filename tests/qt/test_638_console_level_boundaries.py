"""Saved console levels gate every visible sink without changing INFO on disk."""
import logging
import threading

import pytest

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.fixture
def console(qtbot, private_ai_profile):
    from spacr.qt import preferences
    from spacr.qt.widgets.console_panel import ConsolePanel
    preferences._settings().setValue(preferences._KEY_LOG_FILE_LEVELS,
                                     'DEBUG,INFO,WARNING,ERROR,CRITICAL')
    preferences._settings().setValue(preferences._KEY_LOG_CONSOLE_LEVELS,
                                     'WARNING,ERROR,CRITICAL')
    panel = ConsolePanel(follow_log=False, chat=False)
    qtbot.addWidget(panel)
    yield panel, preferences
    panel.close()


def test_real_qt_records_from_worker_hide_info_but_keep_warning_error_and_file(console, qtbot, tmp_path):
    from spacr.qt.logging_util import QtLogHandler
    panel, preferences = console
    handler = QtLogHandler(logging.DEBUG)
    handler.records_ready.connect(panel._on_log_records)
    path = tmp_path / 'full.log'
    disk = logging.FileHandler(path)
    logger = logging.Logger('spacr.test_console_level_boundary', logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(disk)
    def produce():
        logger.info('HIDDEN_STRUCTURED_INFO')
        logger.warning('VISIBLE_STRUCTURED_WARNING')
        logger.error('VISIBLE_STRUCTURED_ERROR')
    worker = threading.Thread(target=produce)
    worker.start()
    worker.join(1)
    assert not worker.is_alive()
    qtbot.waitUntil(lambda: 'VISIBLE_STRUCTURED_ERROR' in panel.as_text())
    visible = panel.as_text()
    assert 'HIDDEN_STRUCTURED_INFO' not in visible
    assert 'VISIBLE_STRUCTURED_WARNING' in visible
    disk.flush()
    assert 'HIDDEN_STRUCTURED_INFO' in path.read_text()
    assert logging.INFO in preferences.get_log_file_levels()
    assert logging.INFO not in preferences.get_log_console_levels()
    disk.close()
    handler.close()


def test_pending_metadata_is_filtered_again_after_console_switch_changes(console):
    panel, preferences = console
    store = preferences._settings()
    store.setValue(preferences._KEY_LOG_CONSOLE_LEVELS, 'DEBUG,INFO,WARNING,ERROR,CRITICAL')
    panel._on_log_records([('queued info\n', logging.INFO), ('queued debug\n', logging.DEBUG)])
    assert panel._log_pending == [('queued info\n', logging.INFO), ('queued debug\n', logging.DEBUG)]
    store.setValue(preferences._KEY_LOG_CONSOLE_LEVELS, 'DEBUG,WARNING,ERROR,CRITICAL')
    panel._flush_log_records()
    assert 'queued info' not in panel.as_text() and 'queued debug' in panel.as_text()


def test_mixed_stdout_keeps_warning_and_multichunk_traceback(console):
    panel, _ = console
    panel.append_stdout('12:00:01 [INFO] HIDDEN_TAGGED\n'
                        '12:00:02 [WARNING] VISIBLE_WARNING\nwarning detail\n'
                        'Traceback (most recent call last):\n')
    panel.append_stdout('  File "worker.py", line 3, in measure\n')
    panel.append_stdout('ValueError: VISIBLE_TRACEBACK\n')
    panel.append_stdout('warning: VISIBLE_LOWERCASE_WARNING\n')
    panel.append_stdout('  VISIBLE_WARNING_CONTINUATION\n')
    panel.append_stdout('HIDDEN_UNCLASSIFIED_INFO\n')
    panel.append_notice('HIDDEN_AUTHORED_NOTICE')
    text = panel.as_text()
    assert 'HIDDEN_' not in text
    assert 'VISIBLE_WARNING' in text and 'warning detail' in text
    assert 'worker.py' in text and 'VISIBLE_TRACEBACK' in text
    assert 'VISIBLE_LOWERCASE_WARNING' in text and 'VISIBLE_WARNING_CONTINUATION' in text


def test_stdout_relay_reads_preference_on_gui_delivery(console, qtbot):
    panel, preferences = console
    store = preferences._settings()
    store.setValue(preferences._KEY_LOG_CONSOLE_LEVELS, 'INFO,WARNING,ERROR,CRITICAL')
    worker = threading.Thread(target=lambda: panel.append_stdout('QUEUED_RAW_INFO\n'))
    worker.start()
    worker.join(1)
    store.setValue(preferences._KEY_LOG_CONSOLE_LEVELS, 'WARNING,ERROR,CRITICAL')
    qtbot.wait(60)
    assert 'QUEUED_RAW_INFO' not in panel.as_text()
    worker = threading.Thread(target=lambda: panel.append_warning('VISIBLE_THREAD_WARNING\n'))
    worker.start()
    worker.join(1)
    qtbot.waitUntil(lambda: 'VISIBLE_THREAD_WARNING' in panel.as_text())


def test_verbose_forwarder_style_and_progress_use_info_policy(console):
    panel, preferences = console
    panel._display_stdout('[12:04:05] spacr.measure INFO HIDDEN_VERBOSE\n')
    panel.append_stdout('Progress: 1/10, operation_type: measure\n')
    assert 'HIDDEN_VERBOSE' not in panel.as_text() and 'Progress:' not in panel.as_text()
    preferences._settings().setValue(preferences._KEY_LOG_CONSOLE_LEVELS, 'INFO,WARNING,ERROR,CRITICAL')
    panel.append_stdout('Progress: 2/10, operation_type: measure\n')
    assert 'Progress: 2/10' in panel.as_text()


def test_fragmented_worker_headers_keep_warning_error_and_traceback(console, qtbot):
    panel, preferences = console
    chunks = ('[WARN', 'ING] SPLIT_WARNING\n', '[ERR', 'OR] SPLIT_ERROR\n',
              'Traceb', 'ack (most recent call last):\n',
              '  File "fragment.py", line 2\n', 'Exception: SPLIT_EXCEPTION\n',
              '  ERROR_DETAIL_AFTER_EXCEPTION\n', 'HIDDEN_PLAIN_AFTER_TRACEBACK\n')
    def produce():
        for chunk in chunks:
            panel.append_stdout(chunk)
    worker = threading.Thread(target=produce)
    worker.start()
    worker.join(1)
    qtbot.waitUntil(lambda: 'ERROR_DETAIL_AFTER_EXCEPTION' in panel.as_text())
    text = panel.as_text()
    assert all(value in text for value in ('[WARNING] SPLIT_WARNING', '[ERROR] SPLIT_ERROR',
                                          'Traceback (most recent call last):', 'fragment.py',
                                          'Exception: SPLIT_EXCEPTION'))
    assert 'HIDDEN_PLAIN_AFTER_TRACEBACK' not in text
    preferences._settings().setValue(preferences._KEY_LOG_CONSOLE_LEVELS, 'INFO,WARNING,ERROR,CRITICAL')
    panel.append_stdout('ordinary output without newline')
    assert 'ordinary output without newline' in panel.as_text()
    panel.append_stdout('[INF')
    qtbot.waitUntil(lambda: '[INF' in panel.as_text())
    assert not panel._stdout_prefix_fragment
