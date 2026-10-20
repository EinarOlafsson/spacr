"""Optional installer cleanup is an explicit, operation-bound GUI choice."""
import copy
import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QTreeWidget

from spacr.qt import install_consent as consent


@pytest.fixture
def preview(tmp_path):
    rows = []
    for key, name, body, selectable in [
        ("news", "releases.json", json.dumps({"fetched": 1, "releases": []}).encode(), True),
        ("icons", "button-0123456789abcdef0123.png", b"cached icon bytes", True),
        ("models", "my-trained-checkpoint.pt", b"irreplaceable user weights", False),
    ]:
        folder = tmp_path / ".spacr" / key
        folder.mkdir(parents=True)
        (folder / name).write_bytes(body)
        rows.append({"key": key, "path": str(folder), "size_bytes": len(body),
                     "contents": [{"name": name, "size_bytes": len(body), "kind": "file"}],
                     "selectable": selectable, "reason": "" if selectable else "protected"})
    return {"operation_id": "install-operation-20261015-opaque", "folders": rows}


def test_cleanup_dialog_shows_actual_paths_contents_sizes_with_every_choice_off(qtbot, preview):
    dialog = consent._InstallerCleanupDialog(preview)
    qtbot.addWidget(dialog)
    tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
    assert tree is not None and tree.topLevelItemCount() == 3
    assert dialog.choices() == {"operation_id": preview["operation_id"], "snapshot_id": None, "keys": []}
    for index, row in enumerate(preview["folders"]):
        item = tree.topLevelItem(index)
        assert item.checkState(0) == Qt.Unchecked
        assert item.text(1) == row["path"]
        assert str(row["size_bytes"]) in item.text(2)
        assert item.child(0).text(0) == row["contents"][0]["name"]
    assert not (tree.topLevelItem(2).flags() & Qt.ItemIsEnabled)


def test_cleanup_choice_returns_only_explicit_eligible_keys_without_touching_files(qtbot, preview):
    dialog = consent._InstallerCleanupDialog(preview)
    qtbot.addWidget(dialog)
    tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
    for index in range(3):
        tree.topLevelItem(index).setCheckState(0, Qt.Checked)
    assert dialog.choices() == {"operation_id": preview["operation_id"], "snapshot_id": None, "keys": ["news", "icons"]}
    from pathlib import Path

    for row in preview["folders"]:
        assert sum(path.stat().st_size for path in Path(row["path"]).iterdir()) == row["size_bytes"]


def test_dialog_freezes_the_displayed_operation_and_rows(qtbot, preview):
    expected = copy.deepcopy(preview)
    dialog = consent._InstallerCleanupDialog(preview)
    qtbot.addWidget(dialog)
    preview["operation_id"] = "another-operation"
    preview["folders"][0]["key"] = "models"
    preview["folders"][0]["path"] = "/another/user/folder"
    tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
    tree.topLevelItem(0).setCheckState(0, Qt.Checked)
    assert tree.topLevelItem(0).text(1) == expected["folders"][0]["path"]
    assert dialog.choices() == {"operation_id": expected["operation_id"], "snapshot_id": None, "keys": ["news"]}


def test_protected_category_cannot_become_selectable_from_a_preview_flag(qtbot, preview):
    preview["folders"][2]["selectable"] = True
    dialog = consent._InstallerCleanupDialog(preview)
    qtbot.addWidget(dialog)
    item = dialog.findChild(QTreeWidget, "InstallerCleanupFolders").topLevelItem(2)
    item.setCheckState(0, Qt.Checked)
    assert not (item.flags() & Qt.ItemIsEnabled)
    assert dialog.choices()["keys"] == []


def test_declined_cleanup_never_reads_or_retains_an_authorization(monkeypatch, preview):
    class Declined:
        def __init__(self, plan, parent):
            assert plan == preview

        def exec(self):
            return QDialog.Rejected

        def choices(self):
            raise AssertionError("cancelled cleanup choices must not be read")

    monkeypatch.setattr(consent, "_InstallerCleanupDialog", Declined)
    assert consent._choose_optional_cleanup(preview) is None


def test_accepting_no_selected_folder_is_a_keep_everything_choice(monkeypatch, preview):
    class Kept:
        def __init__(self, plan, parent):
            pass

        def exec(self):
            return QDialog.Accepted

        def choices(self):
            return {"operation_id": preview["operation_id"], "snapshot_id": None, "keys": []}

    monkeypatch.setattr(consent, "_InstallerCleanupDialog", Kept)
    assert consent._choose_optional_cleanup(preview) is None


def test_keep_is_default_even_after_a_candidate_is_selected(qtbot, preview):
    from PySide6.QtWidgets import QPushButton
    dialog = consent._InstallerCleanupDialog(preview)
    qtbot.addWidget(dialog)
    tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
    tree.topLevelItem(0).setCheckState(0, Qt.Checked)
    keep = dialog.findChild(QPushButton, "InstallerCleanupKeep")
    remove = dialog.findChild(QPushButton, "InstallerCleanupRemove")
    assert keep.isDefault() and not remove.autoDefault()
    dialog.show()
    keep.setFocus()
    qtbot.keyClick(keep, Qt.Key_Return)
    assert dialog.result() == QDialog.Rejected


def test_production_preview_disables_deletion_and_explains_why(qtbot, tmp_path):
    from spacr import install_cleanup as ic
    folder = tmp_path / ".spacr" / "news"
    folder.mkdir(parents=True)
    (folder / "releases.json").write_text("{}")
    machine = ic._Machine(environ={"HOME": str(tmp_path)}, platform="linux")
    actual = ic.preview_optional_cleanup("actual-operation", system=machine)
    dialog = consent._InstallerCleanupDialog(actual)
    qtbot.addWidget(dialog)
    tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
    assert not any(tree.topLevelItem(i).flags() & Qt.ItemIsEnabled
                   for i in range(tree.topLevelItemCount()))
    assert "cannot confirm" in tree.topLevelItem(0).text(3)
    assert dialog.choices()["snapshot_id"] == actual["snapshot_id"]
    assert dialog.choices()["keys"] == []


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "fr", "pt", "hi", "ko", "is"])
def test_actual_inventory_retention_reason_and_keep_button_use_reviewed_locale(qtbot, tmp_path, monkeypatch, language):
    import importlib
    from PySide6.QtWidgets import QPushButton
    from spacr import install_cleanup as ic
    from spacr.qt import i18n
    monkeypatch.setattr(i18n, "current_language", lambda: language)
    folder = tmp_path / ".spacr" / "news"
    folder.mkdir(parents=True)
    (folder / "releases.json").write_text("{}")
    machine = ic._Machine(environ={"HOME": str(tmp_path)}, platform="linux")
    actual = ic.preview_optional_cleanup("actual-operation", system=machine)
    dialog = consent._InstallerCleanupDialog(actual)
    qtbot.addWidget(dialog)
    catalog = importlib.import_module("spacr.qt.i18n_catalogs." + language)
    expected_reason = catalog.UI["Kept: spaCR cannot confirm these files are no longer in use."]
    tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
    assert tree.topLevelItem(0).text(3) == expected_reason
    assert "News" not in catalog.UI and "Models" not in catalog.UI
    assert tree.topLevelItem(1).text(0) == catalog.UI["Background image cache"]
    assert tree.topLevelItem(6).text(0) == catalog.UI["Hugging Face cache"]
    assert tree.topLevelItem(7).text(0) == catalog.UI["Torch cache"]
    assert i18n.tr("News") == i18n.CATALOGS[language]["News"]
    assert i18n.tr("Models") == i18n.TERM_CATALOGS[language]["Models"]
    assert tree.topLevelItem(0).text(0) == i18n.CATALOGS[language]["News"]
    assert any(tree.topLevelItem(index).text(0) == i18n.TERM_CATALOGS[language]["Models"]
               for index in range(tree.topLevelItemCount()))
    assert dialog.windowTitle() == catalog.UI["Optional cache cleanup"]
    keep = dialog.findChild(QPushButton, "InstallerCleanupKeep")
    assert keep.text() == catalog.UI["Keep all files"] and keep.isDefault()
    assert dialog.choices()["keys"] == []
    assert not tree.topLevelItem(0).flags() & Qt.ItemIsEnabled
    assert (folder / "releases.json").read_text(encoding="utf-8") == "{}"


def test_live_guard_preview_starts_unchecked_then_selected_gui_choice_reaches_backend(qtbot, tmp_path):
    from contextlib import contextmanager
    from PySide6.QtWidgets import QPushButton
    from spacr import install_cleanup as ic
    folder = tmp_path / ".spacr" / "news"
    folder.mkdir(parents=True)
    (folder / "releases.json").write_text("{}")
    machine = ic._Machine(environ={"HOME": str(tmp_path)}, platform="linux")
    @contextmanager
    def guard(_paths):
        yield None
    actual = ic.preview_optional_cleanup("gui-live-" + str(tmp_path), system=machine, guard=guard)
    dialog = consent._InstallerCleanupDialog(actual)
    qtbot.addWidget(dialog)
    tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
    remove = dialog.findChild(QPushButton, "InstallerCleanupRemove")
    assert tree.topLevelItem(0).flags() & Qt.ItemIsEnabled
    assert tree.topLevelItem(0).checkState(0) == Qt.Unchecked and not remove.isEnabled()
    tree.topLevelItem(0).setCheckState(0, Qt.Checked)
    assert remove.isEnabled() and not remove.autoDefault()
    qtbot.mouseClick(remove, Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted
    outcome = ic.apply_optional_cleanup(actual, dialog.choices(), system=machine, guard=guard)
    assert outcome["ok"] and outcome["removed"] == 1
    assert list(folder.iterdir()) == []


def _broken(preview, change):
    """A deep copy of ``preview`` with ``change`` applied to it."""
    damaged = copy.deepcopy(preview)
    change(damaged)
    return damaged


@pytest.mark.parametrize("change", [
    lambda p: p.pop("operation_id"),
    lambda p: p.update(operation_id="  "),
    lambda p: p.update(folders="news"),
    lambda p: p["folders"].append("not a row"),
    lambda p: p["folders"].append(dict(p["folders"][0])),
    lambda p: p["folders"][0].update(size_bytes=-1),
    lambda p: p["folders"][0].update(size_bytes=True),
    lambda p: p["folders"][0].update(path=""),
    lambda p: p["folders"][0].update(contents="releases.json"),
    lambda p: p["folders"][0]["contents"].append("not an entry"),
    lambda p: p["folders"][0]["contents"].append({"name": 7}),
    lambda p: p["folders"][0]["contents"].append({"name": "x", "size_bytes": -2}),
    lambda p: p["folders"][0]["contents"].append({"name": "x", "size_bytes": 1.5}),
], ids=["no-operation", "blank-operation", "folders-not-a-list", "row-not-a-mapping",
        "repeated-key", "negative-size", "boolean-size", "no-path", "contents-not-a-list",
        "entry-not-a-mapping", "entry-without-a-name", "entry-negative-size",
        "entry-fractional-size"])
def test_a_damaged_preview_is_refused_before_any_row_is_shown(preview, change):
    """Every malformed part of a preview stops the dialog and keeps all files."""
    with pytest.raises(ValueError, match="cleanup preview is invalid"):
        consent._cleanup_dialog_rows(_broken(preview, change))


def test_a_preview_that_is_not_a_mapping_is_refused():
    """Only a mapping can describe a cleanup."""
    with pytest.raises(ValueError, match="cleanup preview is invalid"):
        consent._cleanup_dialog_rows(["news"])


def test_unknown_sizes_are_shown_as_unknown_not_refused(preview):
    """A folder or entry without a size is still listed, with no size."""
    preview["folders"][0]["size_bytes"] = None
    preview["folders"][0]["contents"][0]["size_bytes"] = None
    operation, rows = consent._cleanup_dialog_rows(preview)
    assert operation == "install-operation-20261015-opaque"
    key, _path, size, entries, eligible, _reason = rows[0]
    assert (key, size, entries[0][1], eligible) == ("news", None, None, True)
