"""Qt recovery messages retain error details and stop failed operations."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QMessageBox

pytestmark = pytest.mark.qt


@pytest.fixture
def warning(monkeypatch):
    """Capture English warnings without creating windows or changing preferences."""
    from spacr.qt import i18n

    captured = Mock()
    monkeypatch.setattr(QMessageBox, "warning", captured)
    monkeypatch.setattr(i18n, "current_language", Mock(return_value="en"))
    return captured


def test_unreachable_update_server_explains_connection_settings(warning):
    """A failed update check retains the server error and returns before offering an update."""
    from spacr.qt.app import MainWindow

    window = SimpleNamespace(_closing=False)
    info = SimpleNamespace(error="connection refused", latest_release=None)
    assert MainWindow._on_update_check_done(window, info) is None
    message = warning.call_args.args[2]
    assert "connection refused" in message
    assert "internet connection and proxy settings" in message
    assert "try again later" in message


def test_failed_package_upgrade_keeps_output_without_restarting(warning):
    """A nonzero package-manager result shows the captured details and never restarts."""
    from spacr.qt.app import MainWindow

    window = SimpleNamespace(_closing=False, _restart_after_package_upgrade=Mock())
    assert MainWindow._on_upgrade_done(window, (17, "permission denied")) is None
    message = warning.call_args.args[2]
    assert "exit code 17" in message and "permission denied" in message
    assert "before retrying the update" in message
    window._restart_after_package_upgrade.assert_not_called()


def test_update_worker_failure_explains_what_to_report(warning):
    """A worker failure preserves its final diagnostic line and explains the report contents."""
    from spacr.qt.app import MainWindow

    window = SimpleNamespace(_closing=False)
    assert MainWindow._on_update_worker_failed(window, "check", "trace\nnetwork failed\n") is None
    message = warning.call_args.args[2]
    assert "Update check failed:" in message and "network failed" in message
    assert "spaCR version and installation method" in message


@pytest.mark.parametrize("condition, advice", [
    ("cell_area > 10;", "Remove the semicolon"),
    ("cell_area > 10 -- note", "Remove comment text"),
    ("DELETE FROM cell", "turn off raw SQL and use the column/operator controls"),
])
def test_rejected_raw_filter_explains_valid_condition(condition, advice, warning):
    """Invalid raw filters remain ValueErrors and explain how to enter a read condition."""
    from spacr.qt.screens.db_browser import validate_raw_predicate

    with pytest.raises(ValueError) as caught:
        validate_raw_predicate(condition)
    assert advice in str(caught.value)
    warning.assert_not_called()


@pytest.mark.parametrize("name, advice", [
    (" ", "Enter a nonempty name"),
    ("existing", "Enter a different name"),
])
def test_template_rename_refusal_preserves_files(tmp_path, name, advice, warning):
    """Blank and duplicate template names leave both saved files untouched."""
    from spacr.qt import recipes

    recipe = recipes.Recipe(name="original", app_key="mask", settings={})
    original = Path(recipes.save_recipe(recipe, directory=str(tmp_path)))
    other = tmp_path / "existing.json"
    other.write_text("keep", encoding="utf-8")
    before = original.read_bytes()
    with pytest.raises(ValueError) as caught:
        recipes._rename_recipe(recipe, name)
    assert advice in str(caught.value)
    assert recipe.name == "original" and recipe.path == str(original)
    assert original.read_bytes() == before
    assert other.read_text(encoding="utf-8") == "keep"
    warning.assert_not_called()


def test_cluster_failure_names_inputs_to_review(warning):
    """A rejected cluster result keeps the cause and asks for measurement and parameter review."""
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    assert GateEditorPanel._finish_cluster(SimpleNamespace(), ("error", ValueError("too few columns"))) is None
    message = warning.call_args.args[2]
    assert "too few columns" in message
    assert "selected measurement columns and clustering parameters" in message


def test_annotation_threshold_refusal_explains_numeric_format(warning):
    """A nonnumeric annotation threshold remains an error and includes the offending rule."""
    from spacr.qt.screens.annotate import _AutoAnnotateDialog

    dialog = SimpleNamespace(_rules=SimpleNamespace(toPlainText=Mock(return_value="cell_area > many")))
    with pytest.raises(ValueError) as caught:
        _AutoAnnotateDialog.parsed_rules(dialog)
    message = str(caught.value)
    assert "cell_area > many" in message
    assert "Replace the threshold with a number" in message
    assert "dot as the decimal separator" in message
    warning.assert_not_called()
