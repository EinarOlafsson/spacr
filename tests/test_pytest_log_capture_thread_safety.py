"""Log capture and background imports share the standard logger-registry lock."""
import logging
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from _pytest.logging import LogCaptureHandler, catching_logs

from tests.conftest import _guard_pytest_logging_capture


pytestmark = pytest.mark.skipif(
    "attached_loggers" not in getattr(catching_logs, "__slots__", ()),
    reason="requires pytest 9.2 non-propagating logger capture capability",
)


def test_capture_serializes_real_logger_creation_and_restores_handlers_and_levels():
    root = logging.getLogger()
    name = 'spacr_capture_' + uuid4().hex
    anchor = logging.getLogger(name)
    anchor.propagate = False
    handler = LogCaptureHandler()
    prior_root_handlers, prior_anchor_handlers = list(root.handlers), list(anchor.handlers)
    prior_level = root.level
    start, attempting, completed = threading.Event(), threading.Event(), threading.Event()
    original_add = anchor.addHandler
    observed, registry_available = [], []

    def create_logger():
        if start.wait(2):
            acquired = logging._lock.acquire(blocking=False)
            registry_available.append(acquired)
            if acquired:
                logging._lock.release()
            attempting.set()
            logging.getLogger(name + '.background')
            completed.set()

    def checkpoint(value):
        start.set()
        assert attempting.wait(2)
        assert registry_available == [False]
        observed.append(completed.is_set())
        return original_add(value)

    patch = pytest.MonkeyPatch()
    patch.setattr(anchor, 'addHandler', checkpoint)
    worker = threading.Thread(target=create_logger, name='capture-registry-creation')
    worker.start()
    capture = catching_logs(handler, logging.DEBUG)
    try:
        with capture:
            assert completed.wait(2), 'registry lock must be released before the body'
            assert observed == [False]
            assert root.level == min(prior_level, logging.DEBUG)
            anchor.warning('non-propagating record')
            root.warning('root record')
        records = [(record.name, record.getMessage()) for record in handler.records
                   if record.name in (name, root.name)]
        assert records == [(name, 'non-propagating record'), (root.name, 'root record')]
        assert root.level == prior_level
        assert root.handlers == prior_root_handlers and anchor.handlers == prior_anchor_handlers
        assert name + '.background' in root.manager.loggerDict
    finally:
        patch.undo()
        start.set()
        worker.join(2)
        for logger in list(capture.attached_loggers):
            logger.removeHandler(handler)
        capture.attached_loggers.clear()
        root.setLevel(prior_level)
    assert not worker.is_alive()


def test_capture_entry_exception_releases_registry_lock_and_configuration_cleanup_restores_entry():
    import _pytest.logging as pytest_logging

    legacy_enter = object()
    legacy_capture = SimpleNamespace(__slots__=(), __enter__=legacy_enter)
    legacy_callbacks = []
    with pytest.MonkeyPatch.context() as legacy_patch:
        legacy_patch.setattr(pytest_logging, 'catching_logs', legacy_capture)
        _guard_pytest_logging_capture(SimpleNamespace(add_cleanup=legacy_callbacks.append))
        assert legacy_capture.__enter__ is legacy_enter and legacy_callbacks == []
    callbacks = []
    original_enter = catching_logs.__enter__
    _guard_pytest_logging_capture(SimpleNamespace(add_cleanup=callbacks.append))
    assert len(callbacks) == 1 and catching_logs.__enter__ is not original_enter
    root = logging.getLogger()
    name = 'spacr_capture_failure_' + uuid4().hex
    anchor = logging.getLogger(name)
    anchor.propagate = False
    handler = LogCaptureHandler()
    failure = RuntimeError('the existing handler refused attachment')
    completed = threading.Event()
    capture = catching_logs(handler, level=None)
    worker = None

    def refuse(value):
        raise failure

    def create_after_failure():
        logging.getLogger(name + '.after_failure')
        completed.set()

    patch = pytest.MonkeyPatch()
    patch.setattr(anchor, 'addHandler', refuse)
    try:
        with pytest.raises(RuntimeError) as caught:
            capture.__enter__()
        assert caught.value is failure
        worker = threading.Thread(target=create_after_failure, name='capture-registry-after-error')
        worker.start()
        assert completed.wait(2), 'entry failure must release the registry lock'
    finally:
        patch.undo()
        capture.__exit__(None, None, None)
        if worker is not None:
            worker.join(2)
        callbacks.pop()()
    assert catching_logs.__enter__ is original_enter
    assert handler not in root.handlers and handler not in anchor.handlers
    assert worker is not None and not worker.is_alive()
