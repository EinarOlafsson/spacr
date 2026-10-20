import pytest

@pytest.fixture(autouse=True)
def actual_session_theme_context(qt_theme_applied):
    yield qt_theme_applied
