"""Starting spaCR must not import the analysis stack to draw a Home page.

Three module-level imports were paid at every launch for code the window
does not run:

  * `retranslate_widget_tree` imported the settings model to refresh API
    tooltips -- which that model itself attaches, so a tree can only hold one
    if the model has already been imported. On the Home page it never has.
    That import reached external_mask_inputs, then external_masks, then
    convert, which imports pandas.
  * `classify_classes` imported pandas at module scope for two lines inside
    one function; every other mention is an annotation, and this file already
    defers those.
  * `class_editor` imported pandas for two annotations and nothing else.

Measured: main window 1.141 s to 0.806 s, 856 modules to 434, and pandas no
longer loaded at launch.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from tests.child_env import child_env

pytest.importorskip("PySide6")


def _in_a_cold_process(body: str) -> str:
    """Run `body` in a fresh interpreter and return its stdout.

    A COLD ONE, because this is a question about what gets imported, and the
    test process has already imported most of spaCR by the time it runs.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, timeout=300,
        env=child_env(home="/tmp/spacr-launch-probe", qt=True,
                      CUDA_VISIBLE_DEVICES=""),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_a_cold_launch_does_not_import_pandas():
    out = _in_a_cold_process("""
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        import sys
        import spacr.qt.app as A
        A.MainWindow()
        print("pandas:", "pandas" in sys.modules)
        print("modules:", len(sys.modules))
    """)
    assert "pandas: False" in out, out


def test_a_cold_launch_stays_under_a_module_budget():
    """A number, so a regression is visible rather than merely slower.

    Set with headroom over the 434 measured, because an unrelated feature may
    legitimately add a few -- but not four hundred, which is what pulling the
    analysis stack back in would cost.
    """
    out = _in_a_cold_process("""
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        import sys
        import spacr.qt.app as A
        A.MainWindow()
        print("modules:", len(sys.modules))
    """)
    count = int(out.split("modules:")[1].split()[0])
    assert count < 600, f"{count} modules imported to open the window"


def test_classify_classes_does_not_import_pandas():
    out = _in_a_cold_process("""
        import sys
        import spacr.classify_classes
        print("pandas:", "pandas" in sys.modules)
    """)
    assert "pandas: False" in out, out


def test_assign_classes_still_works():
    """The deferral must not have broken the two lines it deferred for."""
    import pandas as pd

    from spacr.classify_classes import CLASSES, assign_classes

    frame = pd.DataFrame({"col": ["a", "b", "a", "c"]})
    labels = assign_classes(frame, {CLASSES: {
        "A": {"column": "col", "value": "a"},
        "B": {"column": "col", "value": "b"}}})
    assert list(labels)[:3] == ["A", "B", "A"]
    assert pd.isna(list(labels)[3])


def test_retranslating_a_tree_with_no_settings_imports_nothing():
    out = _in_a_cold_process("""
        from PySide6.QtWidgets import QApplication, QLabel, QWidget
        app = QApplication([])
        import sys
        from spacr.qt.i18n import retranslate_widget_tree
        plain = QWidget(); QLabel("hello", plain)
        retranslate_widget_tree(plain, "sv")
        print("settings_model:",
              "spacr.qt.screens.settings_model" in sys.modules)
    """)
    assert "settings_model: False" in out, out


def test_retranslating_a_real_settings_tree_still_refreshes(qtbot):
    """The other half: the guard must not skip when there IS something."""
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.screens.app_screen import AppScreen
    import spacr.qt.screens.settings_model as settings_model

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    seen = []
    original = settings_model.refresh_api_tooltips
    settings_model.refresh_api_tooltips = (
        lambda root, code: seen.append(code) or original(root, code))
    try:
        retranslate_widget_tree(screen, "sv")
    finally:
        settings_model.refresh_api_tooltips = original
    assert seen == ["sv"], "the API tooltips were not refreshed"
