import pytest

@pytest.fixture(scope="session", autouse=True)
def existing_gui_application():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
