import sys
import pytest

def guard(event, args):
    if event == 'socket.getaddrinfo' and str(args[0]) not in {'127.0.0.1', '::1'}:
        raise RuntimeError('Only owned loopback HTTP fixtures are permitted')
    if event == 'socket.connect':
        address = args[1]
        if not isinstance(address, tuple) or str(address[0]) not in {'127.0.0.1', '::1'}:
            raise RuntimeError('Only owned loopback HTTP fixtures are permitted')

sys.addaudithook(guard)
raise SystemExit(pytest.main(sys.argv[1:]))
