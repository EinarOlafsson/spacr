"""Recipes — a named settings bundle you can reuse and hand to someone else.

A lab does not run one set of settings; it runs a handful, each tied to a
preparation. "Toxo PVM, 40×" is a real thing people say to each other, and
until now the only way to carry it between sessions was a settings CSV in a
folder somebody had to remember, with no name on it, no record of which
module it belonged to, and no way to tell whether it was written by the
version of spaCR about to consume it.

A recipe fixes all four:

* it has a **name** the user chose;
* it knows its **module**, so applying a Mask recipe to Measure is refused
  rather than silently writing seventeen keys that mean nothing there;
* it records the **spaCR version** it was captured with, and applying it
  under a different one says so, listing the settings that no longer exist
  and the ones that have appeared since;
* it is **one file**, so sharing it is sending a file.

Storage is ``~/.spacr/recipes/<module>/<slug>.json``, honouring
:data:`RECIPE_DIR_ENV` — the same shape :func:`spacr.macro.macros_dir` uses,
so a lab that redirects one redirects both.

The file format is deliberately boring::

    {
      "spacr_recipe": 1,
      "name": "Toxo PVM, 40x",
      "app_key": "mask",
      "spacr_version": "1.3.6",
      "created": "2026-08-03T22:41:07",
      "notes": "",
      "settings": {"cell_channel": 1, ...}
    }

``spacr_recipe`` is a format version, not the spaCR version: the two change
for unrelated reasons and conflating them is how a reader ends up refusing a
file it could have read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger("spacr.qt.recipes")

#: The button this module hangs on the settings-search strip.
RECIPE_BUTTON_NAME = "SettingsRecipeButton"


def _recipe_button_qss(palette: dict, opacity=None) -> str:
    """Make the Recipes button float on the page like the one beside it.

    A named ``QToolButton`` with no rule of its own takes the blanket
    ``QWidget {{ background-color: bg }}`` — the WINDOW colour, which is
    near-black and is not a surface, so no page-opacity setting reaches
    it. It sat as a black rectangle on a strip that is meant to be type on
    the page.

    Deliberately the same shape as ``QToolButton#SettingsSearchDisclosure``
    beside it: transparent body, hairline border, accent on hover. The two
    are peers on one row and any difference between them reads as a
    mistake.
    """
    return f"""
QToolButton#{RECIPE_BUTTON_NAME} {{
    background: transparent;
    color: {palette["fg_dim"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 6px;
    padding: 3px 10px;
}}
QToolButton#{RECIPE_BUTTON_NAME}:hover {{
    color: {palette["fg"]};
    border-color: {palette["accent"]};
}}
"""


try:  # pragma: no cover - present in every real launch
    from .theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(RECIPE_BUTTON_NAME, _recipe_button_qss, replace=True)
except Exception:  # pragma: no cover
    LOG.debug("could not register the recipe-button QSS", exc_info=True)

#: Override for the recipe folder, mirroring ``SPACR_MACRO_DIR``.
RECIPE_DIR_ENV = "SPACR_RECIPE_DIR"

#: On-disk format version. Bumped only when a reader would need to behave
#: differently, which has not happened yet.
FORMAT_VERSION = 1

#: The Help-menu label. Kept verbatim — ``spacr/qt/i18n.py`` keys its
#: catalog on the English string.
MENU_ACTION_TEXT = "Settings recipes…"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def spacr_version() -> str:
    """The running spaCR version, or ``"unknown"`` if it cannot be read."""
    try:
        from spacr.version import get_version
        return str(get_version())
    except Exception:
        return "unknown"


def recipes_dir(app_key: Optional[str] = None) -> str:
    """The folder holding recipes, optionally for one module.

    ``~/.spacr/recipes`` (or :data:`RECIPE_DIR_ENV`), with one subfolder per
    module so a listing is already scoped and a user can hand over a whole
    module's worth by copying one directory. Created on first use.

    :param app_key: restrict to one module's folder.
    """
    override = os.environ.get(RECIPE_DIR_ENV, "").strip()
    root = (os.path.abspath(os.path.expanduser(override)) if override
            else os.path.join(os.path.expanduser("~"), ".spacr", "recipes"))
    if app_key:
        root = os.path.join(root, _slug(app_key) or "unknown")
    os.makedirs(root, exist_ok=True)
    return root


def _slug(text: str) -> str:
    """A filesystem-safe stem for a recipe name.

    Lower-cased, non-alphanumerics collapsed to hyphens. The display name is
    stored inside the file, so the slug only has to be unique and typeable —
    it is never what the user reads.
    """
    return _SLUG_RE.sub("-", str(text or "").lower()).strip("-")


@dataclass
class Recipe:
    """One named settings bundle."""

    name: str
    app_key: str
    settings: Dict[str, Any] = field(default_factory=dict)
    spacr_version: str = ""
    created: str = ""
    notes: str = ""
    path: str = ""

    def to_json(self) -> Dict[str, Any]:
        """The on-disk mapping. ``path`` is where it lives, not part of it."""
        return {
            "spacr_recipe": FORMAT_VERSION,
            "name": self.name,
            "app_key": self.app_key,
            "spacr_version": self.spacr_version or spacr_version(),
            "created": self.created or _dt.datetime.now().isoformat(
                timespec="seconds"),
            "notes": self.notes,
            "settings": dict(self.settings),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], path: str = "") -> "Recipe":
        """Build a recipe from a parsed file.

        :raises ValueError: when the mapping is not a recipe at all, or is a
            format version this build does not understand. Both are worth an
            explicit error: silently treating an arbitrary JSON file as a
            settings bundle is how a user ends up applying somebody's
            ``package.json`` to a segmentation run.
        """
        if not isinstance(data, dict) or "spacr_recipe" not in data:
            raise ValueError("not a spaCR settings recipe")
        version = data.get("spacr_recipe")
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ValueError(
                f"recipe format {version!r} is newer than this spaCR "
                f"understands (it reads up to {FORMAT_VERSION})")
        settings = data.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("recipe has no settings")
        return cls(
            name=str(data.get("name") or "Untitled"),
            app_key=str(data.get("app_key") or ""),
            settings=dict(settings),
            spacr_version=str(data.get("spacr_version") or "unknown"),
            created=str(data.get("created") or ""),
            notes=str(data.get("notes") or ""),
            path=path,
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def save_recipe(recipe: Recipe, directory: Optional[str] = None) -> str:
    """Write ``recipe`` and return its path.

    :param recipe: the bundle. Its ``spacr_version`` and ``created`` are
        filled in here when empty, so a caller never has to remember to
        stamp them — the stamp is the point of the format.
    :param directory: override the destination (used by export).
    """
    recipe.spacr_version = recipe.spacr_version or spacr_version()
    recipe.created = recipe.created or _dt.datetime.now().isoformat(
        timespec="seconds")
    root = directory or recipes_dir(recipe.app_key)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{_slug(recipe.name) or 'recipe'}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(recipe.to_json(), handle, indent=2, sort_keys=True,
                  default=str)
    recipe.path = path
    return path


def load_recipe(path: str) -> Recipe:
    """Read one recipe file.

    :raises ValueError: on anything that is not a readable recipe.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{os.path.basename(path)} is not valid JSON: {exc}")
    return Recipe.from_json(data, path=path)


def list_recipes(app_key: Optional[str] = None) -> List[Recipe]:
    """Every readable recipe, newest first.

    Unreadable files are logged and skipped rather than raising: one
    corrupt file must not make the whole list unopenable.

    :param app_key: restrict to one module.
    """
    out: List[Recipe] = []
    root = recipes_dir(app_key)
    for entry in sorted(os.listdir(root)):
        if not entry.endswith(".json"):
            continue
        path = os.path.join(root, entry)
        if not os.path.isfile(path):
            continue
        try:
            out.append(load_recipe(path))
        except Exception:
            LOG.debug("skipping unreadable recipe %s", path, exc_info=True)
    out.sort(key=lambda r: r.created, reverse=True)
    return out


def delete_recipe(recipe: Recipe) -> bool:
    """Remove a recipe's file. ``True`` when something was removed."""
    if not recipe.path or not os.path.isfile(recipe.path):
        return False
    os.remove(recipe.path)
    return True


# ---------------------------------------------------------------------------
# Applying one
# ---------------------------------------------------------------------------

def version_note(recipe: Recipe, current: Optional[str] = None) -> str:
    """What to tell the user about the version gap, or ``""`` if there is none.

    Returns a sentence, not a boolean, because "captured with 1.3.4, you are
    on 1.3.6" is the information — a bare warning icon says only that
    something might be wrong and leaves the user no way to judge it.

    :param recipe: the bundle about to be applied.
    :param current: override the running version (tests).
    """
    now = current if current is not None else spacr_version()
    made = recipe.spacr_version or "unknown"
    if made == now:
        return ""
    return (f"This recipe was saved with spaCR {made}; you are running "
            f"{now}. Settings that changed meaning or name in between are "
            f"applied as written.")


def compatibility_note(recipe: Recipe, model) -> str:
    """Which of the recipe's settings this build has no home for.

    The version gap says *that* something may have moved; this says *what*.
    Reported as counts plus the first few names, because a recipe from two
    releases back can differ in thirty keys and a wall of them is not a
    warning, it is noise.

    :param recipe: the bundle.
    :param model: the screen's ``SettingsWidgets``.
    """
    known = set(getattr(model, "_widgets", {}) or {})
    if not known:
        return ""
    unknown = sorted(set(recipe.settings) - known)
    if not unknown:
        return ""
    shown = ", ".join(unknown[:4])
    more = f" and {len(unknown) - 4} more" if len(unknown) > 4 else ""
    return (f"{len(unknown)} setting(s) in this recipe are not in this "
            f"module any more and will be ignored: {shown}{more}.")


def apply_recipe(recipe: Recipe, screen) -> int:
    """Write a recipe's settings into ``screen``. Returns how many landed.

    Delegates to ``AppScreen.apply_settings_dict``, which is the same path
    the settings-CSV import takes — so a recipe cannot reach a widget that
    an imported CSV could not, and neither can drift from the other.

    :raises ValueError: when the recipe belongs to a different module. It is
        refused rather than partially applied: the keys that happen to
        overlap between two modules are exactly the generic ones (``src``,
        ``verbose``, ``n_jobs``), so a "successful" cross-module apply
        writes the least meaningful half and reports success.
    """
    app_key = str(getattr(screen, "app_key", "") or "")
    if recipe.app_key and app_key and recipe.app_key != app_key:
        raise ValueError(
            f"this recipe is for the {recipe.app_key!r} module, not "
            f"{app_key!r}")
    apply_dict = getattr(screen, "apply_settings_dict", None)
    if not callable(apply_dict):
        raise ValueError("this screen cannot take settings")
    return int(apply_dict(dict(recipe.settings)) or 0)


def capture_recipe(screen, name: str, notes: str = "") -> Recipe:
    """Build a recipe from a screen's current settings.

    Uses ``SettingsWidgets.collect``, so what is captured is exactly what a
    Run would use — including the defaults the user never touched. That is
    deliberate: a recipe is meant to reproduce a result, and a bundle of
    only the edits reproduces something different the day a default changes.
    """
    model = getattr(screen, "_settings_model", None)
    if model is None:
        raise ValueError("this screen has no settings to capture")
    return Recipe(
        name=str(name or "Untitled"),
        app_key=str(getattr(screen, "app_key", "") or ""),
        settings=dict(model.collect()),
        notes=str(notes or ""),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class RecipeDialog(QDialog):
    """List, apply, share and delete the recipes for one module."""

    def __init__(self, screen, parent: Optional[QWidget] = None):
        super().__init__(parent or screen)
        self._screen = screen
        self._app_key = str(getattr(screen, "app_key", "") or "")
        self.setWindowTitle(f"Settings recipes — {self._app_key or 'module'}")
        self.setMinimumWidth(520)
        self.setObjectName("RecipeDialog")

        column = QVBoxLayout(self)
        column.setSpacing(8)

        self._intro = QLabel(
            "A recipe is this module's settings under a name you chose. "
            "Save one when a plate is set up the way you want it; apply it "
            "next time instead of retyping.", self)
        self._intro.setWordWrap(True)
        column.addWidget(self._intro)

        self._list = QListWidget(self)
        self._list.setObjectName("RecipeList")
        self._list.currentRowChanged.connect(self._on_selection_changed)
        column.addWidget(self._list, 1)

        self._detail = QLabel("", self)
        self._detail.setObjectName("RecipeDetail")
        self._detail.setWordWrap(True)
        column.addWidget(self._detail)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._btn_save = QPushButton("Save current settings…", self)
        self._btn_save.clicked.connect(self._on_save)
        row.addWidget(self._btn_save)
        self._btn_import = QPushButton("Import…", self)
        self._btn_import.clicked.connect(self._on_import)
        row.addWidget(self._btn_import)
        row.addStretch(1)
        self._btn_export = QPushButton("Share…", self)
        self._btn_export.clicked.connect(self._on_export)
        row.addWidget(self._btn_export)
        self._btn_delete = QPushButton("Delete", self)
        self._btn_delete.clicked.connect(self._on_delete)
        row.addWidget(self._btn_delete)
        self._btn_apply = QPushButton("Apply", self)
        self._btn_apply.setDefault(True)
        self._btn_apply.clicked.connect(self._on_apply)
        row.addWidget(self._btn_apply)
        column.addLayout(row)

        self.reload()

    # -- public -------------------------------------------------------
    def recipes(self) -> List[Recipe]:
        """The recipes currently listed."""
        return list(self._recipes)

    def reload(self) -> None:
        """Re-read the module's recipe folder and repopulate the list."""
        self._recipes = list_recipes(self._app_key)
        self._list.clear()
        for recipe in self._recipes:
            item = QListWidgetItem(
                f"{recipe.name}    ·    spaCR {recipe.spacr_version}")
            item.setData(Qt.UserRole, recipe.path)
            self._list.addItem(item)
        if self._recipes:
            self._list.setCurrentRow(0)
        else:
            self._detail.setText(
                "No recipes yet for this module. Set the settings up the way "
                "you want them, then use “Save current settings…”.")
        self._refresh_buttons()

    def selected(self) -> Optional[Recipe]:
        """The highlighted recipe, or ``None``."""
        row = self._list.currentRow()
        if 0 <= row < len(self._recipes):
            return self._recipes[row]
        return None

    def detail_text(self) -> str:
        """The line under the list. Public so tests read what users read."""
        return self._detail.text()

    # -- slots --------------------------------------------------------
    def _on_selection_changed(self, _row: int) -> None:
        recipe = self.selected()
        if recipe is None:
            self._detail.setText("")
            self._refresh_buttons()
            return
        parts = [f"{len(recipe.settings)} settings, saved {recipe.created}."]
        note = version_note(recipe)
        if note:
            parts.append(note)
        model = getattr(self._screen, "_settings_model", None)
        if model is not None:
            gap = compatibility_note(recipe, model)
            if gap:
                parts.append(gap)
        if recipe.notes:
            parts.append(recipe.notes)
        self._detail.setText(" ".join(parts))
        self._refresh_buttons()

    def _on_save(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save recipe",
            "Name this recipe — something you would say out loud, "
            "like “Toxo PVM, 40×”:")
        if not ok or not str(name).strip():
            return
        try:
            recipe = capture_recipe(self._screen, str(name).strip())
            save_recipe(recipe)
        except Exception as exc:
            QMessageBox.warning(self, "Could not save recipe", str(exc))
            return
        self.reload()

    def _on_apply(self) -> None:
        recipe = self.selected()
        if recipe is None:
            return
        note = version_note(recipe)
        model = getattr(self._screen, "_settings_model", None)
        gap = compatibility_note(recipe, model) if model is not None else ""
        if note or gap:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("Recipe from a different spaCR")
            box.setText(" ".join(part for part in (note, gap) if part))
            box.setInformativeText("Apply it anyway?")
            box.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
            if box.exec() != QMessageBox.Apply:
                return
        try:
            applied = apply_recipe(recipe, self._screen)
        except Exception as exc:
            QMessageBox.warning(self, "Could not apply recipe", str(exc))
            return
        self._detail.setText(
            f"Applied “{recipe.name}” — {applied} settings written.")

    def _on_export(self) -> None:
        recipe = self.selected()
        if recipe is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Share recipe", f"{_slug(recipe.name)}.json",
            "spaCR recipe (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(recipe.to_json(), handle, indent=2, sort_keys=True,
                          default=str)
        except Exception as exc:
            QMessageBox.warning(self, "Could not write the file", str(exc))

    def _on_import(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import recipe", "", "spaCR recipe (*.json);;All files (*)")
        if not path:
            return
        try:
            recipe = load_recipe(path)
            if recipe.app_key and self._app_key and \
                    recipe.app_key != self._app_key:
                raise ValueError(
                    f"that recipe is for the {recipe.app_key!r} module")
            recipe.app_key = recipe.app_key or self._app_key
            save_recipe(recipe)
        except Exception as exc:
            QMessageBox.warning(self, "Could not import recipe", str(exc))
            return
        self.reload()

    def _on_delete(self) -> None:
        recipe = self.selected()
        if recipe is None:
            return
        try:
            delete_recipe(recipe)
        except Exception as exc:
            QMessageBox.warning(self, "Could not delete recipe", str(exc))
            return
        self.reload()

    def _refresh_buttons(self) -> None:
        has = self.selected() is not None
        for button in (self._btn_apply, self._btn_export, self._btn_delete):
            button.setEnabled(has)


def open_recipes(screen, parent: Optional[QWidget] = None) -> RecipeDialog:
    """Open the recipe dialog for ``screen``."""
    dialog = RecipeDialog(screen, parent=parent)
    dialog.show()
    return dialog


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install(screen) -> Optional[QToolButton]:
    """Add a Recipes button to ``screen``'s settings search strip.

    The strip is where a settings bundle belongs — directly above the
    settings it bundles — and it already exists, so this costs no chrome of
    its own. Returns ``None`` when the screen has no strip (a bespoke
    screen), or when one is already installed.
    """
    if getattr(screen, "_recipe_button", None) is not None:
        return screen._recipe_button
    bar = getattr(screen, "_settings_search", None)
    if bar is None or not hasattr(bar, "add_trailing_widget"):
        return None
    button = QToolButton(bar)
    button.setObjectName(RECIPE_BUTTON_NAME)
    button.setText("Recipes")
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(
        "Save these settings under a name, reuse a saved one, or share it "
        "as a file.")
    handler = _RecipeButtonHandler(screen, button)
    button.clicked.connect(handler.on_clicked)
    button._spacr_recipe_handler = handler
    bar.add_trailing_widget(button)
    screen._recipe_button = button
    return button


class _RecipeButtonHandler:
    """Bound-method target for the Recipes button.

    A plain object rather than a lambda so the connection holds a reference
    to something that is not the screen's closure environment; the button
    owns it, and it dies with the button.
    """

    def __init__(self, screen, button: QToolButton):
        self._screen = screen
        self._button = button

    def on_clicked(self, _checked: bool = False) -> None:
        """Open the recipe dialog for this handler's screen."""
        open_recipes(self._screen, parent=self._button.window())


def _find_menu(window: QMainWindow, title: str) -> Optional[QMenu]:
    """The window's menu-bar menu titled ``title``, ignoring ``&``.

    ``findChildren`` rather than walking ``menuBar().actions()`` and calling
    ``QAction.menu()`` — see the note on
    :func:`spacr.qt.widgets.feature_dictionary._find_menu`; the obvious
    reading hands back a QMenu wrapper that dies with the QAction wrapper it
    came off, and keeping the owners alive segfaults instead.
    """
    try:
        bar = window.menuBar()
        if bar is None:
            return None
        menus = bar.findChildren(QMenu)
    except Exception:
        return None
    for menu in menus:
        try:
            if menu.title().replace("&", "") == title:
                return menu
        except RuntimeError:
            continue
    return None


class _RecipeMenuHandler:
    """Bound-method target for the Help-menu entry."""

    def __init__(self, window: QMainWindow):
        self._window = window

    def on_triggered(self, _checked: bool = False) -> None:
        """Open recipes for whichever module is on screen."""
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            screen = None
        if screen is None or getattr(screen, "_settings_model", None) is None:
            QMessageBox.information(
                self._window, "Settings recipes",
                "Open a module with a settings panel first — a recipe is a "
                "bundle of one module's settings.")
            return
        open_recipes(screen, parent=self._window)


def install_help_action(window: QMainWindow) -> Optional[QAction]:
    """Add **Settings recipes…** to the window's Help menu.

    Returns the action, or ``None`` when there is no Help menu or one is
    already installed. The command palette mirrors menu actions, so this
    also makes recipes reachable from Ctrl+K for free.
    """
    menu = _find_menu(window, "Help")
    if menu is None:
        return None
    for act in menu.actions():
        if act.text() == MENU_ACTION_TEXT:
            return None
    action = QAction(MENU_ACTION_TEXT, window)
    action.setStatusTip(
        "Save the current module's settings under a name, reuse a saved "
        "bundle, or share one as a file.")
    handler = _RecipeMenuHandler(window)
    action.triggered.connect(handler.on_triggered)
    action._spacr_recipe_handler = handler
    before = None
    for act in menu.actions():
        if act.isSeparator():
            before = act
            break
    if before is not None:
        menu.insertAction(before, action)
    else:
        menu.addAction(action)
    return action


class _StackWatcher(QObject):
    """Adds the Recipes button to each settings screen as it is shown.

    A ``QObject`` parented to the window so the connection dies with the
    window, and the slot is a bound method rather than a closure.
    """

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window = window

    def on_current_changed(self, _index: int) -> None:
        """Install into whatever screen the stack just switched to."""
        self.install_current()

    def install_current(self) -> Optional[QToolButton]:
        """Install into the stack's current widget, if it has a strip."""
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            return None
        if screen is None:
            return None
        return install(screen)


def install_window_hooks(window: QMainWindow) -> Optional[_StackWatcher]:
    """Wire recipes into a live main window.

    Called once from :func:`spacr.qt.shortcuts.install`. Every failure is
    logged and swallowed: a missing recipe button must not cost a window.
    """
    install_help_action(window)
    stack = getattr(window, "_stack", None)
    if stack is None:
        return None
    if getattr(window, "_recipe_watcher", None) is not None:
        return window._recipe_watcher
    watcher = _StackWatcher(window)
    try:
        # Queued after the search strip's own watcher, which is connected
        # first in `shortcuts._install_window_hooks` — Qt delivers to slots
        # in connection order, so the strip exists by the time this runs.
        stack.currentChanged.connect(watcher.on_current_changed)
    except Exception:
        LOG.debug("could not follow the screen stack", exc_info=True)
        return None
    window._recipe_watcher = watcher
    # Scheduled after the search strip's own deferred install, because Qt
    # runs zero-timers in the order they were started and `shortcuts`
    # installs the strip first.
    QTimer.singleShot(0, watcher.install_current)
    return watcher
