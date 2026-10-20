import json
import pathlib
import sys
blocked=[]
def offline(event,args):
    if event=='socket.getaddrinfo' or event=='socket.connect' and isinstance(args[1],tuple):
        blocked.append(event)
        raise RuntimeError('Offline assertion-only AST gate: network refused')
sys.addaudithook(offline)
import pytest
try:
    status=pytest.main(['-q','tests/test_test_suite_hygiene.py::test_no_new_assertion_free_tests'])
finally:
    pathlib.Path('/mnt/wd4tb/scratch/ci-current-assertion-free-disposition-20261015/network.json').write_text(json.dumps(blocked)+'\n')
raise SystemExit(status)
