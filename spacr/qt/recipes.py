"""Settings templates — named bundles you can reuse and share.

The historical Recipe API, JSON format and storage paths remain compatible.

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

import datetime as _dt
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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

from .widgets.flow import FlowLayout
from .i18n import tr

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


try:
    from .theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(RECIPE_BUTTON_NAME, _recipe_button_qss, replace=True)
except Exception:
    LOG.debug("could not register the recipe-button QSS", exc_info=True)

#: Override for the recipe folder, mirroring ``SPACR_MACRO_DIR``.
RECIPE_DIR_ENV = "SPACR_RECIPE_DIR"

#: On-disk format version. Bumped only when a reader would need to behave
#: differently, which has not happened yet.
FORMAT_VERSION = 1

#: The Help-menu label. Kept verbatim — ``spacr/qt/i18n.py`` keys its
#: catalog on the English string.
MENU_ACTION_TEXT = "Settings templates…"

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
    """One named settings bundle.

    :param name: display name chosen by the user; its slug becomes the file
        stem when the recipe is saved.
    :param app_key: key of the module the settings belong to; it picks the
        recipe's subfolder and is checked against the screen on apply.
    :param settings: setting key to value mapping written into the screen.
    :param spacr_version: spaCR version the recipe was saved with; filled
        in on save when empty.
    :param created: ISO-8601 timestamp of creation, to the second; filled
        in on save when empty and used to sort listings newest first.
    :param notes: free-text notes stored with the recipe.
    :param path: file the recipe was loaded from or saved to; not written
        into the file itself.
    """

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

        :param data: the parsed JSON mapping; it must carry an integer
            ``spacr_recipe`` format version no newer than this build reads
            and a ``settings`` dict. Missing ``name`` falls back to
            ``"Untitled"``.
        :raises ValueError: when the mapping is not a recipe at all, or is a
            format version this build does not understand. Both are worth an
            explicit error: silently treating an arbitrary JSON file as a
            settings bundle is how a user ends up applying somebody's
            ``package.json`` to a segmentation run.
        """
        if not isinstance(data, dict) or "spacr_recipe" not in data:
            raise ValueError(tr("not a spaCR settings template"))
        version = data.get("spacr_recipe")
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ValueError(
                tr("Template format {version} is newer than this spaCR "
                   "understands (it reads up to {supported}).",
                   version=repr(version), supported=FORMAT_VERSION))
        settings = data.get("settings")
        if not isinstance(settings, dict):
            raise ValueError(tr("template has no settings"))
        return cls(
            name=str(data.get("name") or "Untitled"),
            app_key=str(data.get("app_key") or ""),
            settings=dict(settings),
            spacr_version=str(data.get("spacr_version") or "unknown"),
            created=str(data.get("created") or ""),
            notes=str(data.get("notes") or ""),
            path=path,
        )



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

    :param path: path to a recipe JSON file; it is recorded on the returned
        recipe.
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
    """Remove a recipe's file. ``True`` when something was removed.

    :param recipe: the recipe whose ``path`` is deleted; an empty or missing
        path removes nothing.
    """
    if not recipe.path or not os.path.isfile(recipe.path):
        return False
    os.remove(recipe.path)
    return True



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
    return tr("This template was saved with spaCR {made}; you are running "
              "{now}. Settings that changed meaning or name in between are "
              "applied as written.", made=made, now=now)


def compatibility_note(recipe: Recipe, model) -> str:
    """Which of the recipe's settings this build has no home for.

    The version gap says *that* something may have moved; this says *what*.
    Reported as counts plus the first few names, because a recipe from two
    releases back can differ in thirty keys and a wall of them is not a
    warning, it is noise.

    :param recipe: the bundle.
    :param model: the screen's ``SettingsWidgets``.
    """
    known = (set(getattr(model, "_widgets", {}) or {})
             | set(getattr(model, "_defaults", {}) or {}))
    if not known:
        return ""
    unknown = sorted(set(recipe.settings) - known)
    if not unknown:
        return ""
    shown = ", ".join(unknown[:4])
    more = tr(" and {count} more", count=len(unknown) - 4) if len(unknown) > 4 else ""
    return tr("{count} setting(s) in this template are not in this "
              "module any more and will be ignored: {shown}{more}.",
              count=len(unknown), shown=shown, more=more)


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

    :param recipe: the recipe to apply; its settings are copied, not
        mutated.
    :param screen: the module screen to write into; its ``app_key`` must
        match the recipe's (when both are set) and it must provide
        ``apply_settings_dict``.
    """
    app_key = str(getattr(screen, "app_key", "") or "")
    if recipe.app_key and app_key and recipe.app_key != app_key:
        raise ValueError(
            tr("This template is for the {source} module, not {target}.",
               source=repr(recipe.app_key), target=repr(app_key)))
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

    :param screen: the module screen to capture; it must have a settings
        model (``_settings_model``) or :class:`ValueError` is raised, and
        its ``app_key`` is stored on the recipe.
    :param name: display name for the recipe; empty gives ``"Untitled"``.
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



class RecipeDialog(QDialog):
    """List, apply, share and delete the recipes for one module.

    :param screen: the module screen whose recipes these are. Its `app_key`
        is what the list is filtered by, and it becomes the dialog's parent
        when none is given.
    :param parent: parent widget. Defaults to ``screen``.
    """

    def __init__(self, screen, parent: Optional[QWidget] = None):
        """Build the recipe dialog for one module screen.

        :param screen: the module screen whose settings recipes are saved and
            applied; its ``app_key`` scopes which recipes are listed.
        :param parent: parent widget; defaults to ``screen``.
        """
        super().__init__(parent or screen)
        self._screen = screen
        self._app_key = str(getattr(screen, "app_key", "") or "")
        self._confirmation_runner: Callable[[QMessageBox], Any] = (
            lambda box: box.exec())
        self.setWindowTitle(tr("Settings templates — {module}", module=self._app_key or tr("module")))
        from .preferences import scaled_px
        
        self.setMinimumWidth(scaled_px(520))
        self.setObjectName("RecipeDialog")

        column = QVBoxLayout(self)
        column.setSpacing(8)

        self._intro = QLabel(
            tr("A template is this module's settings under a name you chose. "
            "Save one when a plate is set up the way you want it; apply it "
            "next time instead of retyping."), self)
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

        row = FlowLayout(spacing=6)
        self._btn_save = QPushButton("Save current settings…", self)
        self._btn_save.clicked.connect(self._on_save)
        row.addWidget(self._btn_save)
        self._btn_import = QPushButton("Import…", self)
        self._btn_import.clicked.connect(self._on_import)
        row.addWidget(self._btn_import)
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
                tr("No templates yet for this module. Set the settings up the way "
                   "you want them, then use “Save current settings…”."))
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

    def set_confirmation_runner(
            self, runner: Callable[[QMessageBox], Any]) -> None:
        """Replace how the version/compatibility confirmation is run.

        The default enters the real modal Apply/Cancel loop. Tests inject a
        runner that inspects the fully configured message box and returns an
        answer without blocking; a host can use the same seam for a custom
        presentation.

        :param runner: callable given the configured Apply/Cancel
            ``QMessageBox``; the recipe is applied only when it returns
            ``QMessageBox.Apply``.
        """
        self._confirmation_runner = runner

    def _on_selection_changed(self, _row: int) -> None:
        """Describe the selected recipe, including anything that will not carry.

        The description states the setting count and date, then any note about
        the spaCR version it was written under and any settings this module no
        longer has -- both before the user applies it rather than after.

        :param _row: the newly selected row; the recipe is re-read from the
            list, so it is not used.
        """
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
        """Ask for a name and save the screen's current settings under it.

        A failure is reported in a dialog rather than raised: a recipe that
        cannot be written is a normal condition, not a crash.
        """
        name, ok = QInputDialog.getText(
            self, tr("Save template"),
            tr("Name this template — something you would say out loud, "
               "like “Toxo PVM, 40×”:"))
        if not ok or not str(name).strip():
            return
        try:
            recipe = capture_recipe(self._screen, str(name).strip())
            save_recipe(recipe)
        except Exception as exc:
            QMessageBox.warning(self, tr("Could not save template"), str(exc))
            return
        self.reload()

    def _on_apply(self) -> None:
        """Apply the selected recipe to the screen, confirming first if it may not fit.

        A recipe from a different spaCR version, or one naming settings this
        module no longer has, is applied only after the user says so -- with the
        reason on screen, so the choice is informed rather than a warning to
        click through.
        """
        recipe = self.selected()
        if recipe is None:
            return
        note = version_note(recipe)
        model = getattr(self._screen, "_settings_model", None)
        gap = compatibility_note(recipe, model) if model is not None else ""
        if note or gap:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle(tr("Template from a different spaCR"))
            box.setText(" ".join(part for part in (note, gap) if part))
            box.setInformativeText("Apply it anyway?")
            box.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
            if self._confirmation_runner(box) != QMessageBox.Apply:
                return
        try:
            applied = apply_recipe(recipe, self._screen)
        except Exception as exc:
            QMessageBox.warning(self, tr("Could not apply template"), str(exc))
            return
        self._detail.setText(
            f"Applied “{recipe.name}” — {applied} settings written.")

    def _on_export(self) -> None:
        """Write the selected recipe to a JSON file for sharing."""
        recipe = self.selected()
        if recipe is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, tr("Share template"), f"{_slug(recipe.name)}.json",
            tr("spaCR settings template (*.json)"))
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(recipe.to_json(), handle, indent=2, sort_keys=True,
                          default=str)
        except Exception as exc:
            QMessageBox.warning(self, "Could not write the file", str(exc))

    def _on_import(self) -> None:
        """Load a recipe from a JSON file into this module's collection.

        A recipe belonging to another module is refused: applying it here would
        write settings this screen does not have. One that names no module is
        adopted by this one.
        """
        path, _filter = QFileDialog.getOpenFileName(
            self, tr("Import template"), "", tr("spaCR settings template (*.json);;All files (*)"))
        if not path:
            return
        try:
            recipe = load_recipe(path)
            if recipe.app_key and self._app_key and \
                    recipe.app_key != self._app_key:
                raise ValueError(
                    tr("That template is for the {module} module.", module=repr(recipe.app_key)))
            recipe.app_key = recipe.app_key or self._app_key
            save_recipe(recipe)
        except Exception as exc:
            QMessageBox.warning(self, tr("Could not import template"), str(exc))
            return
        self.reload()

    def _on_delete(self) -> None:
        """Delete the selected recipe."""
        recipe = self.selected()
        if recipe is None:
            return
        try:
            delete_recipe(recipe)
        except Exception as exc:
            QMessageBox.warning(self, tr("Could not delete template"), str(exc))
            return
        self.reload()

    def _refresh_buttons(self) -> None:
        """Enable Apply, Export and Delete only while a recipe is selected."""
        has = self.selected() is not None
        for button in (self._btn_apply, self._btn_export, self._btn_delete):
            button.setEnabled(has)


def open_recipes(screen, parent: Optional[QWidget] = None) -> RecipeDialog:
    """Open the recipe dialog for ``screen``.

    :param screen: the module screen whose recipes are listed and applied,
        passed to :class:`RecipeDialog`.
    """
    dialog = RecipeDialog(screen, parent=parent)
    dialog.show()
    return dialog



def install(screen) -> Optional[QToolButton]:
    """Add a Templates button to ``screen``'s settings search strip.

    The strip is where a settings bundle belongs — directly above the
    settings it bundles — and it already exists, so this costs no chrome of
    its own. Returns ``None`` when the screen has no strip (a bespoke
    screen), or when one is already installed.

    :param screen: the module screen; its ``_settings_search`` strip must
        provide ``add_trailing_widget``. If the screen already has a
        ``_recipe_button``, that button is returned unchanged.
    """
    if getattr(screen, "_recipe_button", None) is not None:
        return screen._recipe_button
    bar = getattr(screen, "_settings_search", None)
    if bar is None or not hasattr(bar, "add_trailing_widget"):
        return None
    button = QToolButton(bar)
    button.setObjectName(RECIPE_BUTTON_NAME)
    caption = "Templates"
    button.setProperty("_spacr_i18n_text", caption)
    button.setText(tr(caption))
    button.setCursor(Qt.PointingHandCursor)
    hint = ("Save these settings under a name, reuse a saved one, or share "
            "it as a file.")
    button.setProperty("_spacr_i18n_tooltip", hint)
    button.setToolTip(tr(hint))
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

    :param screen: the screen whose recipes are opened.
    :param button: the button this is connected to. Held for its
        ``window()``, which is what the dialog is parented to -- so the
        dialog follows the real window even when the screen is reparented.
        It is ALSO what owns this handler, per the note above.
    """

    def __init__(self, screen, button: QToolButton):
        """Hold the screen and the button that owns this handler."""
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
        """Bind the Help-menu entry to whichever module is on screen.

        :param window: the main window; its stack is read at trigger time
            rather than here, so one entry serves every module.

        NOT A QOBJECT and so NOT PARENTED: a bound method connected to a
        signal does not keep its object alive, which is why the caller
        stashes this handler on the action it connected.
        """
        self._window = window

    def on_triggered(self, _checked: bool = False) -> None:
        """Open recipes for whichever module is on screen."""
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            screen = None
        if screen is None or getattr(screen, "_settings_model", None) is None:
            QMessageBox.information(
                self._window, tr("Settings templates"),
                tr("Open a module with a settings panel first — a template is a "
                   "bundle of one module's settings."))
            return
        open_recipes(screen, parent=self._window)


def install_help_action(window: QMainWindow) -> Optional[QAction]:
    """Add **Settings templates…** to the window's Help menu.

    Returns the action, or ``None`` when there is no Help menu or one is
    already installed. The command palette mirrors menu actions, so this
    also makes recipes reachable from Ctrl+K for free.

    :param window: the main window; its menu-bar menu titled "Help" (``&``
        ignored) receives the action, before the first separator.
    """
    menu = _find_menu(window, "Help")
    if menu is None:
        return None
    for act in menu.actions():
        if act.property("_spacr_i18n_text") == MENU_ACTION_TEXT or act.text() == tr(MENU_ACTION_TEXT):
            return None
    action = QAction(tr(MENU_ACTION_TEXT), window)
    action.setProperty("_spacr_i18n_text", MENU_ACTION_TEXT)
    from .menus import set_menu_role
    set_menu_role(action, "none")
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
        """Install the recipe button into each screen as it is shown.

        :param window: the main window whose stack is watched; also the
            QObject parent.
        """
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

    :param window: the main window; the Help-menu action is added to it and
        its ``_stack`` screen stack is watched so each shown screen gets a
        Templates button. Without a ``_stack`` the result is ``None``.
    """
    install_help_action(window)
    stack = getattr(window, "_stack", None)
    if stack is None:
        return None
    if getattr(window, "_recipe_watcher", None) is not None:
        return window._recipe_watcher
    watcher = _StackWatcher(window)
    try:
        stack.currentChanged.connect(watcher.on_current_changed)
    except Exception:
        LOG.debug("could not follow the screen stack", exc_info=True)
        return None
    window._recipe_watcher = watcher
    QTimer.singleShot(0, watcher.install_current)
    return watcher
