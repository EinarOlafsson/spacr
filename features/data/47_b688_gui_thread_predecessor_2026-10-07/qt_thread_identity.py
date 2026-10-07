"""Passive Qt GUI thread wrapper identity at pytest phase boundaries."""
import json
import os
import sys
import time

import pytest


def _record(phase, nodeid):
    widgets = sys.modules.get('PySide6.QtWidgets')
    sample = {'time_ns': time.time_ns(), 'phase': phase, 'nodeid': nodeid,
              'pid': os.getpid(), 'app': None, 'thread': None, 'valid': None}
    if widgets is not None:
        app = widgets.QApplication.instance()
        if app is not None:
            from shiboken6 import getCppPointer, isValid
            thread = app.thread()
            sample['app'] = getCppPointer(app)[0]
            sample['valid'] = bool(isValid(thread))
            sample['thread'] = getCppPointer(thread)[0] if sample['valid'] else None
    with open(os.environ['QT_THREAD_IDENTITY_LOG'], 'ab', buffering=0) as output:
        output.write((json.dumps(sample, sort_keys=True) + '\n').encode())


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _record('setup', item.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if report.when in ('setup', 'call', 'teardown'):
        _record('report_' + report.when, report.nodeid)
