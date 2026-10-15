"""The actual UMAP screen contributes its selected databases to saved runs."""
import copy
import shutil
import sqlite3

from spacr import workspace
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets.database_set import DatabaseSetWidget


def test_actual_umap_registry_copy_restore_reads_carried_sqlite(qtbot, qt_theme_applied, tmp_path):
    workspace.clear_providers()
    plate = tmp_path / "plate"
    database = plate / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, area REAL)")
        connection.execute("INSERT INTO cell VALUES ('p1', 'r1', 'c1', 12.0)")
    first = AppScreen("umap")
    qtbot.addWidget(first)
    second = None
    try:
        source = first._settings_model._widgets["src"]
        assert isinstance(source, DatabaseSetWidget)
        source.set_value([str(plate)])
        document = workspace.collect(workspace.providers(), app_key="umap")
        assert "umap:databases" in document["sections"]
        assert document["sections"]["umap:databases"]["databases"] == [str(database)]
        run = tmp_path / "saved"
        workspace.save(run, document, mode="copy")
        first.close()
        assert "umap:databases" not in workspace.providers()
        shutil.rmtree(plate)
        relocated = tmp_path / "relocated"
        shutil.move(run, relocated)
        loaded = workspace.load(relocated)
        before = copy.deepcopy(loaded)
        second = AppScreen("umap")
        qtbot.addWidget(second)
        restored_source = second._settings_model._widgets["src"]
        assert isinstance(restored_source, DatabaseSetWidget)
        report = workspace.restore(workspace.providers(), loaded, run_dir=relocated)
        carried = next(row["path"] for row in report["files"] if row["state"] == "carried")
        qtbot.waitUntil(lambda: restored_source.database_paths() == [carried], timeout=5000)
        assert "umap:databases" in report["restored"]
        with sqlite3.connect(f"file:{carried}?mode=ro", uri=True) as connection:
            assert connection.execute("SELECT area FROM cell").fetchall() == [(12.0,)]
        assert loaded == before
    finally:
        for screen in (first, second):
            if screen is not None:
                widget = screen._settings_model._widgets.built("src")
                if isinstance(widget, DatabaseSetWidget):
                    widget.shutdown()
                screen.close()
        workspace.clear_providers()


def test_database_registry_uses_current_built_control_and_declines_deferred_or_plain(
    qtbot, qt_theme_applied, tmp_path, monkeypatch,
):
    from PySide6.QtWidgets import QLineEdit

    workspace.clear_providers()
    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    controls = screen._settings_model._widgets
    original = controls["src"]
    replacement = DatabaseSetWidget(parent=screen)
    qtbot.addWidget(replacement)
    plain = QLineEdit(screen)
    try:
        provider = workspace.providers()["umap:databases"]
        assert provider() is original
        original.set_value([str(tmp_path / "old")])
        replacement.set_value([str(tmp_path / "new")])
        controls["src"] = replacement
        states, problems = workspace.section_states({"umap:databases": provider})
        assert states["umap:databases"]["sources"] == [str(tmp_path / "new")]
        assert problems == []
        controls["src"] = plain
        assert workspace.section_states({"umap:databases": provider}) == ({}, [])
        controls.wait_for("src", object())

        def reject_build(*args, **kwargs):
            raise AssertionError("Workspace provider must not build a deferred control")

        monkeypatch.setattr(screen._settings_model, "_build_controls", reject_build)
        assert workspace.section_states({"umap:databases": provider}) == ({}, [])
        assert not controls.is_built("src")
        screen.unregister_workspace()
        assert "umap:databases" not in workspace.providers()
    finally:
        controls["src"] = original
        original.shutdown()
        replacement.shutdown()
        screen.close()
        workspace.clear_providers()
