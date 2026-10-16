import sys
import pytest
def guard(event,args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network forbidden in targeted fixture check")
sys.addaudithook(guard)
raise SystemExit(pytest.main(sys.argv[1:]))
