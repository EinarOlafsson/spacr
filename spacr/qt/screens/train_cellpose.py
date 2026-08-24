"""Cellpose Workbench — fine-tune a model, then segment a folder with it.

Two modules used to sit two rows apart in the registry with nothing in
either line telling a user which one to open: "Train Cellpose — Train
custom Cellpose models" and "Cellpose Masks — Cellpose mask generation",
and a third, "Mask — Generate cellpose masks for cells, nuclei and
pathogens", doing a different job under a name that reads like the second
one. They are not two modules. They are the two halves of one loop:
annotate a handful of fields, fine-tune ``cpsam`` on them, run the
checkpoint over the rest of the folder, look at the masks, annotate the
ones it got wrong, train again. This screen is that loop.

Layout::

    ┌──────────────────────────────────────────────────────────────────┐
    │ Cellpose Workbench   Fine-tune a Cellpose model on your own …  ⓘ │
    │ Train reads <src>/train/images and <src>/train/masks and writes  │
    │ the checkpoint under <src>/models.                               │
    ├──────────────────────────────────────────────────────────────────┤
    │ ╭───────╮╭───────╮                                               │
    │ │ Train ││ Apply │                                               │
    │ ├───────┴┴───────────────────────────────────────────────────────┤
    │ │  the module's own settings form, console, figures and Run row  │
    │ ╰────────────────────────────────────────────────────────────────┤
    │ Apply is set to the model you trained: pv_cpsam_e500_X1000_….    │
    └──────────────────────────────────────────────────────────────────┘

Design notes:

* **Two tabs, two path fields, on purpose.** ``src`` does not mean the
  same thing on the two halves: training reads ``<src>/train/images``
  beside ``<src>/train/masks``, and applying reads ``<src>/*.tif`` and
  writes ``<src>/masks``. One box that silently means either is worse
  than two screens were, so each tab keeps its own — and the line under
  the title says which reading is in force on the tab you are looking
  at. Nothing is carried between the two ``src`` fields, ever.

* **One screen, two live modules.** Each tab is the ordinary
  :class:`~spacr.qt.screens.app_screen.AppScreen` for its key, not a
  reimplementation of it: its own settings form, its own console, its
  own Run button, its own drop target, and — for the Apply half — the
  Live Preview that :mod:`spacr.qt.preview_registry` declares for
  ``cellpose_masks``. The seams a screen normally gets from the window's
  stack watcher are installed here instead, because these two never
  become the stack's current widget.

* **The knobs cross, the paths do not.** Switching tabs copies the
  segmentation knobs from the tab you left into the tab you entered, so
  a diameter chosen for training is the diameter the masks are made at.
  Which knobs those are is read from the propagation map
  :mod:`spacr.qt.preview_registry` already declares for ``cellpose_masks``
  rather than written down a second time here.

* **The model crosses as a checkpoint, not as a string.** ``model_name``
  is the one shared name that means two different things — on Train it is
  the name to save the new model under, on Apply it is which model to
  segment with — so copying the string across would point Apply at a
  stock model that does not exist. What crosses is the file: once a
  training run has actually written a checkpoint, opening Apply sets
  ``custom_model`` to it, which is the setting
  :func:`spacr.spacr_cellpose.identify_masks_finetune` prefers over
  ``model_name``. Before a run has produced one, Apply keeps its stock
  model, so "segment this folder with cpsam" still works on a folder
  nobody has trained anything on.

* **Both keys stay real.** The screen owns one registry row, but the two
  modules underneath are untouched: ``spacr-run train_cellpose`` and
  ``spacr-run cellpose_masks`` run the same entry points with the same
  settings keys, and a settings CSV written from either tab still loads.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ..i18n import tr
from ..theme import SPACING
from .app_screen import AppScreen, ModuleHeader

LOG = logging.getLogger("spacr.qt.screens.train_cellpose")

#: The registry key this screen answers to, and the training half's key.
TRAIN_KEY = "train_cellpose"

#: The applying half. Still a CLI module and still a validate entry; it is
#: the registry ROW that this screen replaced, not the module.
APPLY_KEY = "cellpose_masks"

#: The name the whole loop goes by. Named for the loop rather than for the
#: training half so somebody looking for "segment this folder with a stock
#: model" finds it.
WORKBENCH_TITLE = "Cellpose Workbench"

#: The one line under the name in the registry and beside it on the page.
WORKBENCH_INTRO = (
    "Fine-tune a Cellpose model on your own labelled fields, then segment "
    "a folder of images with it or with a stock model"
)

#: tab index -> (module key, the sentence that says what ``src`` means there).
#: The sentence is the whole reason there are two path fields, so it is shown
#: rather than left for the user to infer from the folder they picked.
TABS: Tuple[Tuple[str, str, str], ...] = (
    (TRAIN_KEY, "Train",
     "Train reads <src>/train/images and <src>/train/masks and writes the "
     "checkpoint under <src>/models."),
    (APPLY_KEY, "Apply",
     "Apply reads every .tif in <src> and writes one mask per image into "
     "<src>/masks."),
)

#: How ``spacr.submodules.train_cellpose`` lays out what it saves, relative
#: to the training ``src``: it hands Cellpose ``<src>/models/cellpose_model``
#: as a save path, and Cellpose puts the checkpoint in a ``models`` folder
#: inside that.
CHECKPOINT_DIR = ("models", "cellpose_model", "models")

#: Infix Cellpose stamps onto the periodic saves it makes during a run. The
#: final save has no infix, so a finished run is preferred over a mid-run
#: snapshot of itself.
EPOCH_INFIX = "_epoch_"


def carried_setting_keys() -> Tuple[str, ...]:
    """The knobs copied from one tab to the other when the tab changes.

    Read from the propagation map :mod:`spacr.qt.preview_registry` declares
    for ``cellpose_masks`` — the map already answers "which settings is a
    Cellpose run judged by", and a second copy of that list here would be
    one that could disagree with it.

    ``model_name`` is excluded: it is the one shared name whose MEANING
    differs between the two halves, and it crosses as a checkpoint path
    instead (see :func:`CellposeWorkbenchScreen.trained_checkpoint`).
    ``src`` is not in the map at all, which is what keeps the two path
    fields independent.
    """
    try:
        from ..preview_registry import PREVIEWS
        spec = PREVIEWS.get(APPLY_KEY)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the preview propagation map", exc_info=True)
        return ()
    if spec is None:
        return ()
    return tuple(name for name in spec.propagation.values()
                 if name not in ("model_name", "src"))


def _install_screen_seams(screen: AppScreen) -> None:
    """Give an embedded module the strip, the recipes button and the preview.

    A module page normally collects these when the window's stack switches
    to it. These two are tabs rather than stack pages, so the watchers never
    see them and the seams are installed directly. Order matters: the search
    strip is what the other two hang their buttons on.

    Every step is guarded on its own — a missing preview toggle must not
    cost anyone the screen it would have sat above.
    """
    try:
        from ..settings_search import install as install_search
        install_search(screen)
    except Exception:                                        # noqa: BLE001
        LOG.debug("no settings search strip for %s", screen.app_key,
                  exc_info=True)
    try:
        from ..recipes import install as install_recipes
        install_recipes(screen)
    except Exception:                                        # noqa: BLE001
        LOG.debug("no recipes button for %s", screen.app_key, exc_info=True)
    try:
        from ..preview_registry import install as install_preview
        install_preview(screen)
    except Exception:                                        # noqa: BLE001
        LOG.debug("no preview for %s", screen.app_key, exc_info=True)


class CellposeWorkbenchScreen(QWidget):
    """Train and Apply as two tabs of one module page.

    :ivar error_explain_requested: re-emitted from whichever tab raised,
        with ``(traceback, app_key)`` — the app key is the TAB's, so the AI
        console is asked about the module that actually failed.
    :ivar remote_submit_requested: re-emitted the same way, so submitting a
        run to Distributed Jobs from either tab carries that tab's key and
        that tab's settings.
    """

    error_explain_requested = Signal(str, str)
    remote_submit_requested = Signal(str, dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        #: The registry key this page was opened under. Fixed, unlike
        #: :meth:`active_app_key` — the window keyed its screen table and
        #: its navigation by this, and a value that moved with the tab
        #: would make the page answer to a key it is not filed under.
        self.app_key = TRAIN_KEY

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        self._header = ModuleHeader(
            tr(WORKBENCH_TITLE),
            description=tr(WORKBENCH_INTRO),
            instruction=tr(TABS[0][2]),
            app_key=TRAIN_KEY,
        )
        outer.addWidget(self._header)

        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("CellposeWorkbenchTabs")
        self._screens: List[AppScreen] = []
        for app_key, label, _instruction in TABS:
            screen = AppScreen(app_key=app_key)
            # One masthead per page. A module's own is 30px of title
            # under this page's own 30px of title, and the applying half
            # no longer has a registry row for its to read a name or a
            # description out of, so it would render "Cellpose_Masks"
            # over nothing. The tab bar says which half you are on, the
            # line under the title says what that half reads, and the
            # API link on this page's header follows the visible tab
            # (see `_sync_instruction`).
            header = getattr(screen, "_header", None)
            if header is not None:
                header.setVisible(False)
            _install_screen_seams(screen)
            screen.error_explain_requested.connect(self.error_explain_requested)
            screen.remote_submit_requested.connect(self.remote_submit_requested)
            self._screens.append(screen)
            self._tabs.addTab(screen, tr(label))
        outer.addWidget(self._tabs, 1)

        #: Says what crossed when a checkpoint did. Hidden until one does —
        #: an empty reserved line reads as a thing that failed to load.
        self._carry_note = QLabel("", self)
        self._carry_note.setObjectName("Muted")
        self._carry_note.setWordWrap(True)
        self._carry_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._carry_note.setVisible(False)
        outer.addWidget(self._carry_note)

        self._current = self._tabs.currentIndex()
        self._tabs.currentChanged.connect(self._on_tab_changed)

    # -- the two halves -----------------------------------------------------

    @property
    def train_screen(self) -> AppScreen:
        """The Train tab's module page."""
        return self._screens[0]

    @property
    def apply_screen(self) -> AppScreen:
        """The Apply tab's module page."""
        return self._screens[1]

    def screen_for(self, app_key: str) -> Optional[AppScreen]:
        """The tab that runs ``app_key``, or ``None``."""
        for screen in self._screens:
            if screen.app_key == str(app_key):
                return screen
        return None

    def active_screen(self) -> AppScreen:
        """The module page the user is looking at."""
        index = self._tabs.currentIndex()
        return self._screens[index if 0 <= index < len(self._screens) else 0]

    def active_app_key(self) -> str:
        """The key of the module the user is looking at."""
        return self.active_screen().app_key

    @property
    def _settings_model(self):
        """The settings model of the visible tab.

        The name the shared tooling reaches for (the command palette's
        jump-to-setting, the recipes dialog, the walkthrough), so this page
        answers it with the form that is actually on screen rather than
        looking like a screen with no settings at all.
        """
        return self.active_screen()._settings_model

    def current_settings(self) -> Tuple[str, Dict]:
        """``(app_key, settings)`` for the visible tab."""
        screen = self.active_screen()
        return screen.app_key, dict(screen._settings_model.collect())

    # -- taking settings from outside ---------------------------------------

    def apply_settings_dict(self, settings: Dict) -> int:
        """Push ``settings`` into whichever tab the dict is for.

        A settings CSV, a restored session or a recipe belongs to one of the
        two modules, and applying it to both would write a training folder
        into the Apply tab's path field — the one thing the two-field split
        exists to prevent. The tab is chosen by which one owns more of the
        keys the OTHER one does not have, so ``n_epochs`` picks Train and
        ``flow_threshold`` picks Apply; a dict that distinguishes neither
        goes to the tab already on screen. The chosen tab is raised, so the
        settings are visible where they landed.

        :param settings: key/value pairs to apply.
        :returns: how many keys the chosen tab actually took.
        """
        settings = dict(settings or {})
        if not settings:
            return 0
        target = self._tab_for_settings(settings)
        index = self._screens.index(target)
        if index != self._tabs.currentIndex():
            self._tabs.setCurrentIndex(index)
        return target.apply_settings_dict(settings)

    def _tab_for_settings(self, settings: Dict) -> AppScreen:
        """The tab a settings dict belongs to. Ties go to the visible one.

        Scored on the keys a tab does NOT share with the other, because the
        shared ones say nothing: ``diameter`` is both modules', ``n_epochs``
        is only training's and ``flow_threshold`` is only applying's.
        """
        given = set(settings)
        owned = [set(screen._settings_model._widgets)
                 for screen in self._screens]
        best, best_score = self.active_screen(), 0
        for index, screen in enumerate(self._screens):
            others = set().union(*(keys for position, keys in enumerate(owned)
                                   if position != index))
            score = len(given & (owned[index] - others))
            if score > best_score:
                best, best_score = screen, score
        return best

    def apply_seed(self, seed: Dict) -> int:
        """Take a seed handed over by another screen. See
        :meth:`apply_settings_dict`, which decides where it lands."""
        return self.apply_settings_dict(seed)

    # -- carrying between the tabs ------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        """Carry the knobs into the tab being opened, and say what src means."""
        previous, self._current = self._current, index
        if 0 <= previous < len(self._screens) and previous != index:
            try:
                self.carry(self._screens[previous], self.active_screen())
            except Exception:                                # noqa: BLE001
                LOG.exception("could not carry the Cellpose settings across")
        self._sync_instruction()

    def _sync_instruction(self) -> None:
        """Put the visible tab's reading of ``src`` under the title."""
        index = self._tabs.currentIndex()
        _key, _label, instruction = TABS[index if 0 <= index < len(TABS) else 0]
        label = getattr(self._header, "instruction_label", None)
        if label is None:
            return
        label.setProperty("_spacr_i18n_text", instruction)
        label.setText(tr(instruction))
        label.setVisible(True)
        link = getattr(self._header, "info_link", None)
        if link is not None and hasattr(link, "set_url"):
            from .settings_model import api_docs_url
            link.set_url(api_docs_url(self.active_app_key()))

    def carry(self, source: AppScreen, target: AppScreen) -> Dict:
        """Copy the shared knobs from ``source`` into ``target``.

        Only the keys :func:`carried_setting_keys` names, and only those
        ``target`` actually has a widget for — the two forms overlap
        partially, and a key the target does not offer would otherwise be
        written into its hidden values where nobody can see or change it.
        Entering the Apply tab additionally picks up the trained checkpoint.

        :param source: the tab being left.
        :param target: the tab being opened.
        :returns: what was written into ``target``.
        """
        try:
            values = dict(source._settings_model.collect())
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read %s's settings", source.app_key,
                      exc_info=True)
            values = {}
        widgets = target._settings_model._widgets
        carried = {key: values[key] for key in carried_setting_keys()
                   if key in values and widgets.get(key) is not None}
        if carried:
            target.apply_settings_dict(carried)
        if target is self.apply_screen:
            checkpoint = self.carry_trained_model()
            if checkpoint:
                carried["custom_model"] = checkpoint
        return carried

    def carry_trained_model(self) -> str:
        """Point the Apply tab at the checkpoint the Train tab produced.

        Does nothing until a training run has actually written one: a
        ``custom_model`` naming a file that is not there stops
        :func:`spacr.spacr_cellpose.identify_masks_finetune` before it
        segments anything, which would break "run cpsam over this folder"
        for everyone who has never trained a model.

        :returns: the checkpoint path that was set, or ``""``.
        """
        checkpoint = self.trained_checkpoint()
        if not checkpoint:
            return ""
        self.apply_screen.apply_settings_dict({"custom_model": checkpoint})
        note = "Apply is set to the model you trained: {name}"
        self._carry_note.setProperty("_spacr_i18n_text", note)
        self._carry_note.setText(
            tr(note, name=os.path.basename(checkpoint)))
        self._carry_note.setVisible(True)
        return checkpoint

    def trained_checkpoint(self) -> str:
        """The newest checkpoint the Train tab's settings would have written.

        Found by looking, not by rebuilding the file name: the training
        module stamps the architecture, the epoch count and the patch size
        into what it saves, and a second copy of that formula here would go
        stale the first time it changed. Everything under the training
        ``src`` whose name starts with the model name counts; a finished
        run's save is preferred over the periodic ones it made on the way,
        and among equals the most recently written wins.

        :returns: an absolute path, or ``""`` when there is nothing to find.
        """
        try:
            values = self.train_screen._settings_model.collect()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the training settings", exc_info=True)
            return ""
        src = str(values.get("src") or "").strip()
        name = str(values.get("model_name") or "").strip()
        if not src or not name:
            return ""
        folder = os.path.join(src, *CHECKPOINT_DIR)
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            return ""
        prefix = f"{name}_"
        found = [os.path.join(folder, entry) for entry in entries
                 if entry.startswith(prefix)
                 and os.path.isfile(os.path.join(folder, entry))]
        finished = [path for path in found
                    if EPOCH_INFIX not in os.path.basename(path)]
        candidates = finished or found
        if not candidates:
            return ""
        try:
            return max(candidates, key=os.path.getmtime)
        except OSError:
            return candidates[-1]


def build_screen(app_key: str = TRAIN_KEY, host=None) -> CellposeWorkbenchScreen:
    """Screen factory for the registry.

    Takes ``app_key`` and ``host`` because that is the contract
    ``spacr.qt.app._call_screen_factory`` offers; the key is fixed (this
    screen serves one row) and the host is used only to make the same two
    connections the window makes on a generic module page.
    """
    screen = CellposeWorkbenchScreen()
    if host is not None:
        for signal_name, slot_name in (
                ("error_explain_requested", "_on_explain_error"),
                ("remote_submit_requested", "_on_remote_submit_requested")):
            signal = getattr(screen, signal_name, None)
            slot = getattr(host, slot_name, None)
            if signal is not None and callable(slot):
                signal.connect(slot)
    return screen
