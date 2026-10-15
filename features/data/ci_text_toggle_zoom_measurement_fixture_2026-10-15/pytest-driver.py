import atexit
import json
import os
from pathlib import Path
import socket
import sys

folder = Path(__file__).parent
events = []
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_sendto = socket.socket.sendto
original_getaddrinfo = socket.getaddrinfo
def refuse(kind, value):
    events.append({'kind': kind, 'address': repr(value)})
    raise RuntimeError('Network prohibited in zoom-only CI baseline')
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('connect', address)
    return original_connect(self, address)
def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('connect_ex', address)
    return original_connect_ex(self, address)
def sendto(self, data, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('sendto', args)
    return original_sendto(self, data, *args)
def getaddrinfo(host, *args, **kwargs):
    refuse('getaddrinfo', host)
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.socket.sendto = sendto
socket.getaddrinfo = getaddrinfo
atexit.register(lambda: (folder / 'baseline-network.json').write_text(json.dumps(events, indent=2) + '\n'))
(folder / 'baseline-startup.json').write_text(json.dumps({'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'qt_platform': os.environ.get('QT_QPA_PLATFORM')}, indent=2) + '\n')
import pytest
raise SystemExit(pytest.main(['-q', '--junitxml=' + str(folder / 'baseline.xml'),
    'tests/qt/test_zoom_reaches_text.py::test_zoom_reaches_the_text_toggles[Live]',
    'tests/qt/test_zoom_reaches_text.py::test_zoom_reaches_the_text_toggles[AI]']))
