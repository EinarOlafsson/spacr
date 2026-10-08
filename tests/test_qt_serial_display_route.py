"""The protected serial runner must use a measurable X display."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_qt_serial_acceptance.sh"


def _bootstrap_source() -> str:
    script = RUNNER.read_text(encoding="utf-8")
    return script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def _fake_qt_and_pytest(tmp_path: Path) -> None:
    widgets = tmp_path / "PySide6" / "QtWidgets.py"
    widgets.parent.mkdir()
    (widgets.parent / "__init__.py").write_text("", encoding="utf-8")
    widgets.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def record(value):\n"
        "    with Path(os.environ['SPACR_BOOTSTRAP_ORDER']).open('a') as out:\n"
        "        out.write(value + '\\n')\n"
        "class QApplication:\n"
        "    def __init__(self, argv): record('application')\n"
        "    def platformName(self): return os.environ['SPACR_FAKE_QT_PLATFORM']\n"
        "    def setApplicationName(self, name): record('name=' + name)\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.py").write_text(
        "from PySide6.QtWidgets import record\n"
        "record('pytest import')\n"
        "def main(args):\n"
        "    record('pytest args=' + '|'.join(args))\n"
        "    return 0\n",
        encoding="utf-8",
    )


def test_serial_bootstraps_x_before_pytest_and_keeps_the_original_selection(tmp_path):
    """Qt 6.12's pytest-qt plugin can crash if it creates the first QApp."""
    _fake_qt_and_pytest(tmp_path)
    order = tmp_path / "order.txt"
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "SPACR_BOOTSTRAP_ORDER": str(order),
        "SPACR_FAKE_QT_PLATFORM": "xcb",
    }
    args = [
        "tests/qt", "-v", "--tb=short", "-p", "no:randomly",
        "-p", "tools.pytest_plugins.qt_serial_rss_journal",
        "-o", "faulthandler_timeout=900", "--timeout=1200",
        "--timeout-method=thread",
    ]
    result = subprocess.run(
        [sys.executable, "-c", _bootstrap_source(), *args],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert order.read_text(encoding="utf-8").splitlines() == [
        "application", "name=pytest-qt-qapp", "pytest import",
        "pytest args=" + "|".join(args),
    ]


def test_serial_rejects_an_offscreen_app_before_pytest_collection(tmp_path):
    """An offscreen grab cannot validate the original Home pixel assertions."""
    _fake_qt_and_pytest(tmp_path)
    order = tmp_path / "order.txt"
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "SPACR_BOOTSTRAP_ORDER": str(order),
        "SPACR_FAKE_QT_PLATFORM": "offscreen",
    }
    result = subprocess.run(
        [sys.executable, "-c", _bootstrap_source(), "tests/qt"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "did not construct a real X-backed QApplication" in result.stderr
    assert order.read_text(encoding="utf-8").splitlines() == ["application"]


def test_serial_shell_guards_the_hard_limits_and_display_preflight():
    script = RUNNER.read_text(encoding="utf-8")
    assert '"$memory_max" != "12884901888"' in script
    assert '"$swap_max" != "0"' in script
    assert '"${SPACR_TEST_MEMORY_GB:-}" != "10.8"' in script
    assert '"${CUDA_VISIBLE_DEVICES-unset}" != ""' in script
    assert '"${QT_QPA_PLATFORM:-}" != "xcb"' in script
    assert 'python tools/can_this_display_be_measured.py' in script
    assert script.index('python tools/can_this_display_be_measured.py') < script.index("exec python - tests/qt")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
