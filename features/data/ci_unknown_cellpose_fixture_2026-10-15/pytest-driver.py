import json, pathlib, sys
import pytest
blocked = []
def deny_network(event, args):
    if event == "socket.getaddrinfo" or (event == "socket.connect" and isinstance(args[1], tuple)):
        blocked.append(event)
        raise RuntimeError("External network forbidden in fixture verification")
sys.addaudithook(deny_network)
try:
    status = pytest.main(sys.argv[2:])
finally:
    pathlib.Path(sys.argv[1]).write_text(json.dumps({"blocked_network_events": blocked}) + "\n")
raise SystemExit(status)
