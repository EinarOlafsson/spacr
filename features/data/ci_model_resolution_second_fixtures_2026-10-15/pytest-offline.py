import sys
import pytest
def guard(event,args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network forbidden in targeted fixture check")
sys.addaudithook(guard)
from PySide6.QtWidgets import QApplication
application = QApplication.instance() or QApplication([])
raise SystemExit(pytest.main(sys.argv[1:]))
