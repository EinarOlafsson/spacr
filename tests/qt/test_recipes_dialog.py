"""The recipe dialog's five buttons, and the two ways it is reached.

``tests/qt/test_recipes.py`` proves the *format*: a recipe round-trips, it
carries the version that wrote it, and applying one from the wrong module is
refused. What it never touches is the half of the module a user actually
operates -- ``Save current settings…``, ``Import…``, ``Share…``, ``Delete``
and ``Apply`` -- nor either of the two routes that put recipes in front of
them (the button on the settings strip, the Help-menu entry). That was 150
uncovered statements: the whole interactive surface of a shipped feature,
including every error path a user meets when a file will not write.

Each slot is driven here with its dialog stubbed at the Qt boundary
(``QInputDialog.getText``, ``QFileDialog.get*FileName``, ``QMessageBox``) and
asserted on its *effect* -- the file on disk, the settings in the model, the
list afterwards, the words in the warning -- rather than on "it did not
raise". A slot that swallows its exception and does nothing passes the
second and fails the first.
"""
from __future__ import annotations

import json
import os

import pytest
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from spacr.qt import recipes as R


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Never write into the user's real ``~/.spacr/recipes``."""
    monkeypatch.setenv("SPACR_RECIPE_DIR", str(tmp_path / "recipes"))
    yield


@pytest.fixture
def mask_screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    return screen


@pytest.fixture
def dialog(mask_screen, qtbot):
    dlg = R.RecipeDialog(mask_screen)
    qtbot.addWidget(dlg)
    return dlg


@pytest.fixture
def warnings(monkeypatch):
    """Capture ``QMessageBox.warning`` calls as ``(title, text)`` pairs."""
    seen = []

    def _warning(_parent, title, text, *args, **kwargs):
        seen.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warning))
    return seen


def _stored_names(app_key="mask"):
    return sorted(r.name for r in R.list_recipes(app_key))


# ---------------------------------------------------------------------------
# 1. Save current settings…
# ---------------------------------------------------------------------------

def test_save_writes_the_named_recipe_and_relists_it(dialog, mask_screen,
                                                     monkeypatch):
    """The button's whole job: name it, and find it in the list afterwards."""
    mask_screen._settings_model.set_value_for_key("n_jobs", 9)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Toxo PVM, 40×", True)))

    dialog._on_save()

    assert _stored_names() == ["Toxo PVM, 40×"]
    assert [r.name for r in dialog.recipes()] == ["Toxo PVM, 40×"]
    # and what was saved is what the screen actually held
    assert R.list_recipes("mask")[0].settings["n_jobs"] == 9


@pytest.mark.parametrize("answer", [
    ("Toxo", False),   # the user pressed Cancel
    ("", True),        # ...or accepted an empty name
    ("   ", True),     # ...or one that is only whitespace
])
def test_save_writes_nothing_when_the_name_is_not_given(dialog, monkeypatch,
                                                        answer):
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: answer))
    dialog._on_save()
    assert _stored_names() == []


def test_save_reports_a_screen_it_cannot_capture(dialog, monkeypatch,
                                                 warnings):
    """A screen with no settings model must produce a warning, not a
    traceback and not a silent no-op."""
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Doomed", True)))
    dialog._screen = QWidget()

    dialog._on_save()

    assert _stored_names() == []
    (title, text), = warnings
    assert title == "Could not save recipe"
    assert "no settings to capture" in text


# ---------------------------------------------------------------------------
# 2. Apply
# ---------------------------------------------------------------------------

def test_apply_writes_the_settings_and_says_how_many(dialog, mask_screen):
    """Same version, nothing stale: it applies without asking anything."""
    mask_screen._settings_model.set_value_for_key("n_jobs", 7)
    R.save_recipe(R.capture_recipe(mask_screen, "Plate A"))
    dialog.reload()
    mask_screen._settings_model.set_value_for_key("n_jobs", 1)

    dialog._on_apply()

    assert mask_screen._settings_model.collect()["n_jobs"] == 7
    assert "Applied “Plate A”" in dialog.detail_text()
    assert "settings written" in dialog.detail_text()


def test_apply_with_nothing_selected_does_nothing(dialog, mask_screen):
    before = dict(mask_screen._settings_model.collect())
    assert dialog.selected() is None
    dialog._on_apply()
    assert mask_screen._settings_model.collect() == before


def test_a_version_gap_asks_first_and_cancel_means_cancel(
        dialog, mask_screen, monkeypatch):
    """The confirmation is the feature -- applying an old bundle has to be a
    decision. Cancelling must leave the screen untouched."""
    mask_screen._settings_model.set_value_for_key("n_jobs", 7)
    old = R.capture_recipe(mask_screen, "From 1.3.0")
    old.spacr_version = "1.3.0"
    R.save_recipe(old)
    dialog.reload()
    mask_screen._settings_model.set_value_for_key("n_jobs", 1)

    shown = []
    monkeypatch.setattr(QMessageBox, "exec",
                        lambda box: shown.append(box.text()) or QMessageBox.Cancel)

    dialog._on_apply()

    assert mask_screen._settings_model.collect()["n_jobs"] == 1
    assert shown and "1.3.0" in shown[0]


def test_a_version_gap_confirmed_applies_after_all(dialog, mask_screen,
                                                   monkeypatch):
    mask_screen._settings_model.set_value_for_key("n_jobs", 7)
    old = R.capture_recipe(mask_screen, "From 1.3.0")
    old.spacr_version = "1.3.0"
    R.save_recipe(old)
    dialog.reload()
    mask_screen._settings_model.set_value_for_key("n_jobs", 1)

    monkeypatch.setattr(QMessageBox, "exec", lambda box: QMessageBox.Apply)

    dialog._on_apply()

    assert mask_screen._settings_model.collect()["n_jobs"] == 7
    assert "Applied" in dialog.detail_text()


def test_stale_keys_alone_also_trigger_the_confirmation(dialog, mask_screen,
                                                        monkeypatch):
    """Same spaCR version, but keys this build has no home for: the user is
    still told before anything is written."""
    recipe = R.capture_recipe(mask_screen, "Has strays")
    recipe.settings["a_setting_that_was_removed"] = 1
    R.save_recipe(recipe)
    dialog.reload()

    shown = []
    monkeypatch.setattr(QMessageBox, "exec",
                        lambda box: shown.append(box.text()) or QMessageBox.Cancel)
    dialog._on_apply()

    assert shown and "a_setting_that_was_removed" in shown[0]


def test_apply_reports_a_screen_that_cannot_take_settings(
        dialog, mask_screen, warnings):
    R.save_recipe(R.capture_recipe(mask_screen, "Plate A"))
    dialog.reload()
    dialog._screen = QWidget()          # no apply_settings_dict

    dialog._on_apply()

    (title, text), = warnings
    assert title == "Could not apply recipe"
    assert "cannot take settings" in text


# ---------------------------------------------------------------------------
# 3. Share… (export)
# ---------------------------------------------------------------------------

def test_export_writes_a_file_that_stands_alone(dialog, mask_screen,
                                                tmp_path, monkeypatch):
    mask_screen._settings_model.set_value_for_key("n_jobs", 4)
    R.save_recipe(R.capture_recipe(mask_screen, "Shared"))
    dialog.reload()
    target = tmp_path / "out" / "shared.json"
    target.parent.mkdir()
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))

    dialog._on_export()

    reloaded = R.load_recipe(str(target))
    assert reloaded.name == "Shared"
    assert reloaded.app_key == "mask"
    assert reloaded.settings["n_jobs"] == 4


def test_export_of_nothing_selected_writes_nothing(dialog, tmp_path,
                                                   monkeypatch):
    called = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: called.append(1) or ("", "")))
    dialog._on_export()
    assert called == []


def test_a_cancelled_export_writes_nothing(dialog, mask_screen, tmp_path,
                                           monkeypatch):
    R.save_recipe(R.capture_recipe(mask_screen, "Shared"))
    dialog.reload()
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    dialog._on_export()
    assert list(tmp_path.glob("*.json")) == []


def test_an_unwritable_export_target_is_reported(dialog, mask_screen,
                                                 tmp_path, monkeypatch,
                                                 warnings):
    """Sharing is sending a file; a file that never got written has to say
    so rather than look like it worked."""
    R.save_recipe(R.capture_recipe(mask_screen, "Shared"))
    dialog.reload()
    target = tmp_path / "no_such_dir" / "shared.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))

    dialog._on_export()

    assert not target.exists()
    (title, _text), = warnings
    assert title == "Could not write the file"


# ---------------------------------------------------------------------------
# 4. Import…
# ---------------------------------------------------------------------------

def test_import_adopts_a_shared_file_into_the_store(dialog, mask_screen,
                                                    tmp_path, monkeypatch):
    incoming = tmp_path / "from_a_colleague.json"
    incoming.write_text(json.dumps({
        "spacr_recipe": R.FORMAT_VERSION,
        "name": "From a colleague",
        "app_key": "mask",
        "spacr_version": "1.3.6",
        "settings": {"n_jobs": 3},
    }))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(incoming), "")))

    dialog._on_import()

    assert _stored_names() == ["From a colleague"]
    assert R.list_recipes("mask")[0].settings["n_jobs"] == 3
    assert [r.name for r in dialog.recipes()] == ["From a colleague"]


def test_an_imported_file_with_no_module_adopts_this_one(dialog, tmp_path,
                                                         monkeypatch):
    """A hand-written bundle with no ``app_key`` is not refused -- it becomes
    a recipe of the module it was imported into."""
    incoming = tmp_path / "bare.json"
    incoming.write_text(json.dumps({
        "spacr_recipe": R.FORMAT_VERSION,
        "name": "Bare",
        "settings": {"n_jobs": 2},
    }))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(incoming), "")))

    dialog._on_import()

    stored = R.list_recipes("mask")
    assert [r.name for r in stored] == ["Bare"]
    assert stored[0].app_key == "mask"


def test_importing_another_modules_recipe_is_refused_by_name(
        dialog, tmp_path, monkeypatch, warnings):
    incoming = tmp_path / "measure.json"
    incoming.write_text(json.dumps({
        "spacr_recipe": R.FORMAT_VERSION,
        "name": "Measure setup",
        "app_key": "measure",
        "settings": {"n_jobs": 1},
    }))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(incoming), "")))

    dialog._on_import()

    assert _stored_names() == []
    (title, text), = warnings
    assert title == "Could not import recipe"
    assert "measure" in text


def test_importing_something_that_is_not_a_recipe_is_refused(
        dialog, tmp_path, monkeypatch, warnings):
    incoming = tmp_path / "package.json"
    incoming.write_text('{"name": "something", "version": "1.0.0"}')
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(incoming), "")))

    dialog._on_import()

    assert _stored_names() == []
    (_title, text), = warnings
    assert "not a spaCR settings recipe" in text


def test_a_cancelled_import_stores_nothing(dialog, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    dialog._on_import()
    assert _stored_names() == []


# ---------------------------------------------------------------------------
# 5. Delete, and selection
# ---------------------------------------------------------------------------

def test_delete_removes_the_file_and_the_row(dialog, mask_screen):
    recipe = R.capture_recipe(mask_screen, "Throwaway")
    path = R.save_recipe(recipe)
    dialog.reload()
    assert dialog.selected() is not None

    dialog._on_delete()

    assert not os.path.exists(path)
    assert dialog.recipes() == []
    assert "No recipes yet" in dialog.detail_text()


def test_delete_with_nothing_selected_does_nothing(dialog):
    dialog._on_delete()
    assert dialog.recipes() == []


def test_a_recipe_whose_file_vanished_is_reported_as_not_deleted(
        dialog, mask_screen, warnings):
    """``delete_recipe`` answers False rather than raising, and the dialog
    still relists -- the stale row has to disappear either way."""
    recipe = R.capture_recipe(mask_screen, "Ghost")
    path = R.save_recipe(recipe)
    dialog.reload()
    os.remove(path)

    dialog._on_delete()

    assert warnings == []
    assert dialog.recipes() == []


def test_deselecting_clears_the_detail_and_disables_the_row_of_buttons(
        dialog, mask_screen):
    R.save_recipe(R.capture_recipe(mask_screen, "Plate A"))
    dialog.reload()
    assert dialog._btn_apply.isEnabled()

    dialog._list.setCurrentRow(-1)

    assert dialog.selected() is None
    assert dialog.detail_text() == ""
    assert not dialog._btn_apply.isEnabled()
    assert not dialog._btn_export.isEnabled()
    assert not dialog._btn_delete.isEnabled()


def test_the_detail_line_carries_the_authors_note(dialog, mask_screen):
    recipe = R.capture_recipe(mask_screen, "Annotated", notes="Use for PVM.")
    R.save_recipe(recipe)
    dialog.reload()
    assert "Use for PVM." in dialog.detail_text()


# ---------------------------------------------------------------------------
# 6. Getting to the dialog: the strip button
# ---------------------------------------------------------------------------

def test_the_strip_button_opens_the_dialog_for_its_own_screen(
        mask_screen, qtbot):
    from spacr.qt.settings_search import install as install_search
    install_search(mask_screen)
    button = R.install(mask_screen)
    assert button is not None

    button._spacr_recipe_handler.on_clicked()

    dialogs = [w for w in mask_screen.findChildren(R.RecipeDialog)]
    opened = dialogs or [w for w in qtbot.__class__.__module__ and []]
    # the dialog is parented to the button's window, so look there too
    found = dialogs or mask_screen.window().findChildren(R.RecipeDialog)
    assert found, "clicking Recipes opened no dialog"
    assert found[0]._app_key == "mask"
    found[0].close()


def test_open_recipes_returns_a_visible_dialog_for_the_screen(mask_screen,
                                                             qtbot):
    dlg = R.open_recipes(mask_screen)
    qtbot.addWidget(dlg)
    assert dlg.isVisible()
    assert dlg._app_key == "mask"
    dlg.close()


# ---------------------------------------------------------------------------
# 7. Getting to the dialog: the Help menu
# ---------------------------------------------------------------------------

@pytest.fixture
def window(qtbot):
    win = QMainWindow()
    qtbot.addWidget(win)
    win.menuBar().addMenu("&Help")
    win._stack = QStackedWidget(win)
    win.setCentralWidget(win._stack)
    return win


def test_the_help_menu_gains_the_entry_exactly_once(window):
    action = R.install_help_action(window)
    assert action is not None
    assert action.text() == R.MENU_ACTION_TEXT
    assert R.install_help_action(window) is None
    menu = R._find_menu(window, "Help")
    assert [a.text() for a in menu.actions()].count(R.MENU_ACTION_TEXT) == 1


def test_a_window_with_no_help_menu_gets_no_entry(qtbot):
    win = QMainWindow()
    qtbot.addWidget(win)
    win.menuBar().addMenu("&File")
    assert R.install_help_action(win) is None


def test_the_entry_goes_above_the_first_separator(qtbot):
    """Menus put the About block below a separator; recipes belong with the
    working commands above it, not after About."""
    win = QMainWindow()
    qtbot.addWidget(win)
    menu = win.menuBar().addMenu("&Help")
    menu.addAction("Documentation")
    menu.addSeparator()
    menu.addAction("About")

    R.install_help_action(win)

    texts = [a.text() for a in menu.actions()]
    assert texts.index(R.MENU_ACTION_TEXT) < texts.index("About")


def test_the_menu_entry_opens_recipes_for_the_visible_module(window,
                                                            mask_screen):
    window._stack.addWidget(mask_screen)
    window._stack.setCurrentWidget(mask_screen)
    action = R.install_help_action(window)

    action._spacr_recipe_handler.on_triggered()

    found = window.findChildren(R.RecipeDialog)
    assert found and found[0]._app_key == "mask"
    found[0].close()


def test_the_menu_entry_explains_itself_when_no_module_is_open(
        window, monkeypatch):
    """A screen with no settings panel is not an error -- it is a person who
    has not opened a module yet, and they get told which thing to do."""
    window._stack.addWidget(QWidget())
    window._stack.setCurrentIndex(0)
    action = R.install_help_action(window)

    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda _p, title, text: told.append((title, text))))

    action._spacr_recipe_handler.on_triggered()

    assert window.findChildren(R.RecipeDialog) == []
    (title, text), = told
    assert title == "Settings recipes"
    assert "Open a module with a settings panel first" in text


def test_a_window_without_a_stack_still_answers_the_menu_entry(qtbot,
                                                              monkeypatch):
    win = QMainWindow()
    qtbot.addWidget(win)
    win.menuBar().addMenu("&Help")
    action = R.install_help_action(win)

    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda _p, title, text: told.append(title)))

    action._spacr_recipe_handler.on_triggered()

    assert told == ["Settings recipes"]


# ---------------------------------------------------------------------------
# 8. _find_menu's two defences
# ---------------------------------------------------------------------------

def test_find_menu_survives_a_window_with_no_menu_bar():
    class _NoBar:
        def menuBar(self):
            return None

    assert R._find_menu(_NoBar(), "Help") is None


def test_find_menu_survives_a_menu_bar_that_raises():
    class _Angry:
        def menuBar(self):
            raise RuntimeError("the C++ object is gone")

    assert R._find_menu(_Angry(), "Help") is None


def test_find_menu_skips_a_deleted_menu_and_keeps_looking(qtbot):
    """A QMenu wrapper outliving its C++ object raises on ``title()``; that
    must not hide the menu after it."""
    real = QMenu("&Help")
    qtbot.addWidget(real)

    class _Dead:
        def title(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    class _Bar:
        def findChildren(self, _kind):
            return [_Dead(), real]

    class _Win:
        def menuBar(self):
            return _Bar()

    assert R._find_menu(_Win(), "Help") is real


# ---------------------------------------------------------------------------
# 9. Window hooks
# ---------------------------------------------------------------------------

def test_window_hooks_install_the_button_on_the_screen_already_showing(
        window, mask_screen):
    from spacr.qt.settings_search import install as install_search
    install_search(mask_screen)
    window._stack.addWidget(mask_screen)
    window._stack.setCurrentWidget(mask_screen)

    watcher = R.install_window_hooks(window)

    assert watcher is not None
    assert watcher.install_current() is mask_screen._recipe_button
    assert mask_screen._recipe_button is not None


def test_window_hooks_are_installed_once_per_window(window):
    first = R.install_window_hooks(window)
    assert R.install_window_hooks(window) is first


def test_a_window_with_no_stack_gets_no_watcher(qtbot):
    win = QMainWindow()
    qtbot.addWidget(win)
    win.menuBar().addMenu("&Help")
    assert R.install_window_hooks(win) is None


def test_switching_screens_installs_into_the_new_one(window, mask_screen,
                                                    qtbot):
    """Each settings screen gets its own button as it is first shown -- the
    watcher exists because a screen built later must not miss out."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import install as install_search
    other = AppScreen("measure")
    qtbot.addWidget(other)
    install_search(mask_screen)
    install_search(other)
    window._stack.addWidget(mask_screen)
    window._stack.addWidget(other)
    window._stack.setCurrentWidget(mask_screen)
    R.install_window_hooks(window)
    first = R.install(mask_screen)

    window._stack.setCurrentWidget(other)

    # currentChanged is connected by install_window_hooks, so simply showing
    # the other screen is what installs into it.
    assert other._recipe_button is not None
    assert other._recipe_button is not first
    assert other._settings_search.isAncestorOf(other._recipe_button)


def test_the_screen_already_showing_gets_its_button_on_the_next_turn(
        window, mask_screen, qtbot):
    """The install is deferred by a zero-timer so it lands after the search
    strip's own deferred install. Nothing appears until the loop turns, and
    it must appear once it does."""
    from spacr.qt.settings_search import install as install_search
    install_search(mask_screen)
    window._stack.addWidget(mask_screen)
    window._stack.setCurrentWidget(mask_screen)

    R.install_window_hooks(window)
    assert getattr(mask_screen, "_recipe_button", None) is None

    qtbot.waitUntil(
        lambda: getattr(mask_screen, "_recipe_button", None) is not None,
        timeout=2000)
    assert mask_screen._settings_search.isAncestorOf(mask_screen._recipe_button)


def test_the_watcher_shrugs_at_an_empty_stack(window):
    watcher = R._StackWatcher(window)
    assert watcher.install_current() is None


def test_the_watcher_shrugs_at_a_window_whose_stack_went_away(qtbot):
    win = QMainWindow()
    qtbot.addWidget(win)
    watcher = R._StackWatcher(win)      # no _stack attribute at all
    assert watcher.install_current() is None


# ---------------------------------------------------------------------------
# 10. The small guards the store makes
# ---------------------------------------------------------------------------

def test_a_recipe_file_with_no_settings_block_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"spacr_recipe": R.FORMAT_VERSION,
                                "name": "Nothing", "app_key": "mask"}))
    with pytest.raises(ValueError, match="no settings"):
        R.load_recipe(str(path))


def test_a_format_version_that_is_not_a_number_is_refused(tmp_path):
    path = tmp_path / "odd.json"
    path.write_text(json.dumps({"spacr_recipe": "1", "name": "Odd",
                                "settings": {}}))
    with pytest.raises(ValueError, match="newer than this spaCR"):
        R.load_recipe(str(path))


def test_deleting_a_recipe_that_was_never_written_answers_false():
    assert R.delete_recipe(R.Recipe(name="Unsaved", app_key="mask")) is False


def test_deleting_a_recipe_whose_file_is_gone_answers_false(mask_screen):
    recipe = R.capture_recipe(mask_screen, "Ghost")
    path = R.save_recipe(recipe)
    os.remove(path)
    assert R.delete_recipe(recipe) is False


def test_a_directory_in_the_store_is_not_mistaken_for_a_recipe(mask_screen):
    R.save_recipe(R.capture_recipe(mask_screen, "Real"))
    os.makedirs(os.path.join(R.recipes_dir("mask"), "notes.json"))
    assert [r.name for r in R.list_recipes("mask")] == ["Real"]


def test_applying_to_a_screen_that_cannot_take_settings_is_refused():
    recipe = R.Recipe(name="Any", app_key="", settings={"n_jobs": 1})
    with pytest.raises(ValueError, match="cannot take settings"):
        R.apply_recipe(recipe, object())


def test_capturing_from_a_screen_with_no_model_is_refused():
    with pytest.raises(ValueError, match="no settings to capture"):
        R.capture_recipe(object(), "Nothing")


def test_the_version_is_reported_as_unknown_when_it_cannot_be_read(
        monkeypatch):
    """``spacr_version`` is called while stamping every recipe; it may not be
    the thing that makes saving fail."""
    import builtins
    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "spacr.version":
            raise ImportError("no version module here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert R.spacr_version() == "unknown"


def test_an_unnamed_recipe_still_gets_a_filename(mask_screen):
    """The slug is allowed to collapse to nothing; the file still has to
    land somewhere readable."""
    recipe = R.Recipe(name="—", app_key="mask", settings={"n_jobs": 1})
    path = R.save_recipe(recipe)
    assert os.path.basename(path) == "recipe.json"
    assert R.load_recipe(path).settings == {"n_jobs": 1}


def test_the_store_falls_back_to_a_named_folder_for_an_odd_module(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setenv("SPACR_RECIPE_DIR", str(tmp_path / "r"))
    assert os.path.basename(R.recipes_dir("—")) == "unknown"


def test_a_stray_file_in_the_store_is_ignored(mask_screen):
    """Users put notes and zips next to their recipes; only ``.json`` is
    read, and a stray must not make the folder unlistable."""
    R.save_recipe(R.capture_recipe(mask_screen, "Real"))
    with open(os.path.join(R.recipes_dir("mask"), "notes.txt"), "w") as fh:
        fh.write("remember to redo plate 3")
    assert [r.name for r in R.list_recipes("mask")] == ["Real"]


def test_a_screen_with_no_known_settings_reports_no_gap():
    """``compatibility_note`` compares against the screen's widgets; with
    none to compare against it must stay quiet rather than declare every
    setting stale."""
    recipe = R.Recipe(name="Any", app_key="mask", settings={"n_jobs": 1})

    class _NoWidgets:
        _widgets = {}

    assert R.compatibility_note(recipe, _NoWidgets()) == ""
    assert R.compatibility_note(recipe, object()) == ""


def test_a_delete_that_fails_is_reported_and_the_row_stays(
        dialog, mask_screen, tmp_path, warnings):
    """A recipe the filesystem refuses to remove must not vanish from the
    list as though it had been deleted."""
    R.save_recipe(R.capture_recipe(mask_screen, "Locked"))
    dialog.reload()
    folder = R.recipes_dir("mask")
    os.chmod(folder, 0o500)                     # readable, not writable
    try:
        dialog._on_delete()
    finally:
        os.chmod(folder, 0o700)

    (title, _text), = warnings
    assert title == "Could not delete recipe"
    assert [r.name for r in R.list_recipes("mask")] == ["Locked"]


def test_a_stack_that_refuses_the_connection_costs_no_window(qtbot):
    """Every failure in the window hook is swallowed on purpose: a missing
    recipe button must not cost a main window. The Help entry still lands."""
    win = QMainWindow()
    qtbot.addWidget(win)
    win.menuBar().addMenu("&Help")

    class _Signal:
        def connect(self, _slot):
            raise RuntimeError("this stack is not accepting connections")

    class _Stack:
        currentChanged = _Signal()

    win._stack = _Stack()

    assert R.install_window_hooks(win) is None
    assert getattr(win, "_recipe_watcher", None) is None
    menu = R._find_menu(win, "Help")
    assert R.MENU_ACTION_TEXT in [a.text() for a in menu.actions()]
