"""The chaining strip: where a module says what it will read, and what is stale.

Three things a pipeline screen has never been able to say, all of them one
strip above the Run button — because that is the moment the user is about to
act, and a panel they would have to go and open is a panel nobody opens:

* **what it will read.**  :mod:`spacr.chaining` asks the artifact registry
  where the previous module actually wrote and fills this module's source
  folder with it, so opening Measure after Mask no longer means retyping the
  plate.  A path the user typed themselves is *pinned* and never overwritten
  — when the upstream later moves, the new location is offered beside the
  pinned one with a button, never pushed into the field;
* **what is out of date.**  :func:`spacr.chaining.staleness_notes` turns the
  registry's cause codes into a sentence and a fix, so "these measurements
  came from a Mask run you have since redone" is on screen *before* the
  figure is opened rather than discovered afterwards;
* **what comes next.**  A finished run offers its successors from
  :func:`spacr.ports.next_modules`, each pre-filled with the artifact just
  produced and each checked with :func:`spacr.ports.check_ready` — so an
  offer either works, or is shown greyed with the reason it cannot.

Installation goes through the two seams that already exist rather than
through the shared screen: :data:`spacr.qt.app.APP_FACTORIES` (consulted by
``MainWindow._build_screen`` before its built-in chain) and
:data:`spacr.qt.SELF_REGISTERING_MODULES`, which imports this module after
``app.py`` is loaded and before the first window is built.  ``AppScreen``
itself is untouched.

:func:`register` is the whole of the installation and is idempotent, so a
build whose launch list has not yet learned about this module can call it
from anywhere that runs before the first window.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from .. import chaining as _chaining
from .. import ports as _ports
from ..chaining import ChainedInput, HeldPin, NextStep, StaleNote

LOG = logging.getLogger("spacr.qt.chaining")

__all__ = [
    "ChainingBar",
    "HOST_CONNECTIONS",
    "chaining_bar",
    "chained_app_keys",
    "install_chaining",
    "register",
    "unregister",
]

#: Refresh debounce. A user typing a path should not open the registry once
#: per keystroke, and every trigger in this module funnels through it.
REFRESH_DELAY_MS = 450

#: The signals ``MainWindow._build_screen`` connects on the generic
#: ``AppScreen`` it builds, mapped to the host slot each goes to.
#:
#: This module builds that screen itself (that is what a registered factory
#: *is*), so it has to make the same connections. Duplicated wiring rots, so
#: ``tests/qt/test_chaining_gui.py`` reads ``_build_screen``'s own source and
#: fails if the two ever disagree — the table is checked, not trusted.
HOST_CONNECTIONS: Dict[str, str] = {
    "error_explain_requested": "_on_explain_error",
    "remote_submit_requested": "_on_remote_submit_requested",
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def _qss(palette: Dict[str, Any], opacity) -> str:
    """Return the strip's stylesheet for one palette.

    Registered through :func:`spacr.qt.theme.register_widget_qss`, so the
    colours follow the user's theme without a line in ``theme.py``.

    :param palette: the theme palette, surfaces already rendered through the
        page opacity.
    :param opacity: the user's page-opacity preference, passed through.
    """
    return f"""
    QFrame#ChainingBar {{
        background: transparent;
        border: none;
        border-top: 1px solid {palette['border_soft']};
        padding-top: 4px;
    }}
    QLabel#ChainingSource {{
        color: {palette['fg_muted']};
        background: transparent;
    }}
    QLabel#ChainingStale {{
        color: {palette['warning']};
        background: transparent;
    }}
    QLabel#ChainingFix, QLabel#ChainingPinned {{
        color: {palette['fg_dim']};
        background: transparent;
    }}
    QLabel#ChainingNext {{
        color: {palette['fg']};
        background: transparent;
    }}
    QPushButton#ChainingStepBlocked {{
        color: {palette['fg_muted']};
    }}
    """


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def _widget_value(widget) -> Any:
    """Return what a settings widget currently holds, or None.

    Prefers the ``get_value`` contract every list and scalar editor in
    :mod:`spacr.qt.screens.settings_model` implements, and falls back to
    ``text()`` for a plain ``QLineEdit``.
    """
    getter = getattr(widget, "get_value", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    text = getattr(widget, "text", None)
    if callable(text):
        try:
            return text()
        except Exception:
            return None
    return None


class _ShowFilter(QObject):
    """Calls back when the watched widget is shown.

    A module screen is built once and kept, so ``__init__`` fires exactly
    once while returning to the screen — which is when a run on another tab
    may have produced the thing this strip is about — fires ``Show``.
    """

    def __init__(self, on_show, parent=None) -> None:
        super().__init__(parent)
        self._on_show = on_show

    def eventFilter(self, obj, event) -> bool:      # noqa: N802 - Qt override
        """Forward a Show event and never consume it."""
        if event.type() == QEvent.Show:
            try:
                self._on_show()
            except Exception:
                LOG.exception("chaining refresh failed on show")
        return False


# ---------------------------------------------------------------------------
# The strip
# ---------------------------------------------------------------------------

class ChainingBar(QFrame):
    """The strip above a module's Run button.

    Four rows, each hidden when it has nothing to say, so a project with no
    recorded runs sees exactly what it saw before this existed:

    1. where the inputs came from;
    2. a pinned path whose upstream has since moved, with a button to take
       the new one;
    3. what is stale, why (by cause), and what to do;
    4. after a finished run, what to do next.

    :param screen: the :class:`spacr.qt.screens.app_screen.AppScreen` this
        belongs to.
    :param pins: the pin store to use. Defaults to the shared one; tests
        hand in their own so the developer's real pins are never touched.
    """

    def __init__(self, screen: QWidget, *, pins=None, parent=None) -> None:
        super().__init__(parent or screen)
        self.setObjectName("ChainingBar")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._screen = screen
        self.app_key = str(getattr(screen, "app_key", ""))
        self._pins = pins if pins is not None else _chaining.pin_store()
        #: settings key → the value this strip last wrote into the widget.
        #: What separates "the user chose this" from "we put it there", which
        #: no settings dict can answer on its own.
        self._offered: Dict[str, Any] = {}
        #: settings key → the last non-empty value seen in the widget. An
        #: empty field only means "the user cleared it" if it once held
        #: something; without this, the first refresh after a restart — when
        #: every field still holds its ``"path"`` placeholder — would read as
        #: the user clearing every path and would delete their pins.
        self._seen: Dict[str, Any] = {}
        self._held: Dict[str, HeldPin] = {}
        #: Cached result of :meth:`_bound_settings`.
        self._bound: Optional[Tuple[str, ...]] = None
        #: True when ``collect()`` returned a whole settings dict. A partial
        #: one must not be hashed against a recorded run: half a dict has a
        #: different digest, and every result would report as stale.
        self._collect_ok = False
        self._last_steps: Tuple[NextStep, ...] = ()

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(2)

        self._source = QLabel()
        self._source.setObjectName("ChainingSource")
        self._source.setWordWrap(True)
        self._source.hide()
        column.addWidget(self._source)

        self._pinned_row = QWidget()
        pinned_layout = QHBoxLayout(self._pinned_row)
        pinned_layout.setContentsMargins(0, 0, 0, 0)
        pinned_layout.setSpacing(8)
        self._pinned = QLabel()
        self._pinned.setObjectName("ChainingPinned")
        self._pinned.setWordWrap(True)
        pinned_layout.addWidget(self._pinned, 1)
        self._btn_use = QPushButton("Use it")
        self._btn_use.setObjectName("GhostButton")
        self._btn_use.setCursor(Qt.PointingHandCursor)
        self._btn_use.clicked.connect(self._on_use_offered)
        pinned_layout.addWidget(self._btn_use)
        self._pinned_row.hide()
        column.addWidget(self._pinned_row)

        self._stale = QLabel()
        self._stale.setObjectName("ChainingStale")
        self._stale.setWordWrap(True)
        self._stale.hide()
        column.addWidget(self._stale)

        self._fix = QLabel()
        self._fix.setObjectName("ChainingFix")
        self._fix.setWordWrap(True)
        self._fix.hide()
        column.addWidget(self._fix)

        self._next_row = QWidget()
        self._next_layout = QHBoxLayout(self._next_row)
        self._next_layout.setContentsMargins(0, 0, 0, 0)
        self._next_layout.setSpacing(8)
        self._next_label = QLabel("Continue to:")
        self._next_label.setObjectName("ChainingNext")
        self._next_layout.addWidget(self._next_label)
        self._next_layout.addStretch(1)
        self._next_row.hide()
        column.addWidget(self._next_row)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFRESH_DELAY_MS)
        self._timer.timeout.connect(self.refresh)

        self._show_filter = _ShowFilter(self._on_screen_shown, self)
        try:
            screen.installEventFilter(self._show_filter)
        except Exception:
            LOG.exception("could not watch %s for show events", self.app_key)

        self._wire_edit_signals()
        self._wire_run_button()

    # -- wiring -----------------------------------------------------------

    def _widgets(self) -> Dict[str, QWidget]:
        """Return the screen's settings widgets, keyed by settings key."""
        model = getattr(self._screen, "_settings_model", None)
        return dict(getattr(model, "_widgets", {}) or {})

    def _bound_settings(self) -> Tuple[str, ...]:
        """Return the settings keys this module's input ports fill.

        Derived from :data:`spacr.ports.PORTS` through
        :func:`spacr.chaining.binding_for`, so a module that declares a port
        is chained without a line here.

        Computed once: the settings panel's widget map is built before this
        strip exists and does not change afterwards, and Mask's has two
        hundred entries that would otherwise be walked on every keystroke.
        """
        if self._bound is not None:
            return self._bound
        try:
            spec = _ports.module_ports(self.app_key)
        except Exception:
            self._bound = ()
            return self._bound
        widgets = self._widgets()
        keys: List[str] = []
        for port in spec.consumes:
            setting = _chaining.binding_for(self.app_key, port).setting
            if setting in widgets and setting not in keys:
                keys.append(setting)
        self._bound = tuple(keys)
        return self._bound

    def _wire_edit_signals(self) -> None:
        """Notice a typed path as it is typed.

        ``textEdited`` and not ``textChanged``: Qt emits the former only for
        input the *user* produced, so the strip filling the field itself, a
        settings CSV import and the Continue button do not read back as edits.
        Widgets without it (the chip list Classify uses for its source) are
        covered by :meth:`_capture_edits`, which compares the value against
        what this strip last wrote and needs no signal at all.
        """
        widgets = self._widgets()
        for key in self._bound_settings():
            widget = widgets.get(key)
            signal = getattr(widget, "textEdited", None)
            if signal is None:
                continue
            try:
                signal.connect(lambda _text: self._on_edited())
            except Exception:
                LOG.exception("could not watch %s.%s for edits",
                              self.app_key, key)

    def _wire_run_button(self) -> None:
        """Follow the run this screen is about to start.

        Connected to the same ``clicked`` signal ``AppScreen._on_run`` is on,
        and connected *after* it, so by the time this slot runs the worker
        exists and can be followed to completion. That is what lets the strip
        offer the next step without a line inside the shared screen.
        """
        button = getattr(self._screen, "_btn_run", None)
        if button is None:
            return
        try:
            button.clicked.connect(self._on_run_clicked)
        except Exception:
            LOG.exception("could not follow the Run button on %s",
                          self.app_key)

    # -- the pin ----------------------------------------------------------

    def _capture_edits(self) -> None:
        """Record any bound path the user has changed since we last wrote it.

        The general form of edit detection, and the only one that works for
        every widget: whatever the field holds now is compared against the
        value this strip put there. Different and non-empty means the user
        chose it, and it is pinned. Emptied means "give me the automatic
        default back", and the pin is dropped.
        """
        widgets = self._widgets()
        for key in self._bound_settings():
            value = _widget_value(widgets.get(key))
            if _chaining.is_empty_path(value):
                # Empty is only a *clearing* if the field held something
                # first. On the first refresh of a fresh screen it just means
                # nobody has typed anything yet, and unpinning there would
                # throw away the path the user chose in a previous session —
                # the exact promise this store exists to keep.
                if key in self._seen:
                    self._pins.unpin(self.app_key, key)
                    self._offered.pop(key, None)
                    self._seen.pop(key, None)
                continue
            self._seen[key] = value
            if _chaining.same_path(value, self._offered.get(key)):
                continue
            pinned = self._pins.pinned(self.app_key, key)
            if pinned is not None and _chaining.same_path(value, pinned):
                continue
            self._pins.pin(self.app_key, key, value)

    def adopt(self, values: Dict[str, Any]) -> int:
        """Apply ``values`` as though this strip had chained them itself.

        Used by the Continue button on the *previous* module's strip: the
        seed it hands over is an artifact the registry resolved, not a path
        the user typed, so it must not become a pin.

        :param values: settings key → value.
        :returns: how many keys the screen accepted.
        """
        applied = 0
        apply = getattr(self._screen, "apply_settings_dict", None)
        if callable(apply):
            try:
                applied = int(apply(dict(values)))
            except Exception:
                LOG.exception("could not seed %s", self.app_key)
        for key, value in values.items():
            self._offered[key] = value
        self.refresh()
        return applied

    # -- slots ------------------------------------------------------------

    def _on_screen_shown(self) -> None:
        """Re-read the registry when the user comes back to this screen."""
        self._timer.start()

    def _on_edited(self) -> None:
        """A keystroke in a bound path field."""
        self._timer.start()

    def _remember_project(self) -> str:
        """Record the folder this module is being run in, and return it.

        Until now only the two interactive screens (Annotate, Make Masks)
        called :func:`spacr.qt.prefs.push_recent_source`, so nothing knew
        which plate Mask had last worked on — and :meth:`search_roots`, which
        is what lets a *blank* Measure screen find that plate, had nothing to
        go on. The registry lives inside the project root, so without this
        there is no registry to ask and chaining could only ever fill a screen
        whose source was already set.
        """
        root = _ports.project_root(self.current_settings(), self.app_key)
        if not root:
            return ""
        try:
            from .prefs import push_recent_source
            push_recent_source(self.app_key, root)
        except Exception:
            LOG.exception("could not remember %s's project folder",
                          self.app_key)
        return root

    def _on_run_clicked(self) -> None:
        """Pin whatever the user typed, then follow the run to its end."""
        self._capture_edits()
        self._remember_project()
        worker = getattr(self._screen, "_worker", None)
        if worker is None:
            return
        try:
            worker.finished.connect(self._on_run_finished)
        except Exception:
            LOG.exception("could not follow the %s run to its end",
                          self.app_key)

    def _on_run_finished(self, ok: bool) -> None:
        """Refresh, and offer the next step when the run actually worked."""
        self.refresh(finished=bool(ok))

    def _on_use_offered(self) -> None:
        """Take the location the upstream has moved to, dropping the pin."""
        for key, pin in list(self._held.items()):
            if not pin.differs:
                continue
            self._pins.unpin(self.app_key, key)
            self._offered[key] = pin.offered
            apply = getattr(self._screen, "apply_settings_dict", None)
            if callable(apply):
                try:
                    apply({key: pin.offered})
                except Exception:
                    LOG.exception("could not apply the offered path for %s",
                                  self.app_key)
        self.refresh()

    def host_window(self):
        """Return the window that owns navigation, or None.

        A method rather than an inline ``self._screen.window()`` so a test can
        stand a window in without reaching into Qt's ownership chain, and so a
        screen shown outside a MainWindow simply finds nothing to navigate.
        """
        try:
            return self._screen.window()
        except Exception:
            return None

    def _on_continue(self, step: NextStep) -> None:
        """Open the successor, pre-filled with what this run produced."""
        window = self.host_window()
        navigate = getattr(window, "_on_nav_selected", None)
        if not callable(navigate):
            return
        try:
            navigate(step.module)
        except Exception:
            LOG.exception("could not open %s", step.module)
            return
        target = (getattr(window, "_screens", {}) or {}).get(step.module)
        if target is None:
            return
        bar = getattr(target, "_chaining_bar", None)
        if bar is not None:
            bar.adopt(step.seed)
            return
        apply = getattr(target, "apply_settings_dict", None)
        if callable(apply):
            try:
                apply(dict(step.seed))
            except Exception:
                LOG.exception("could not seed %s", step.module)

    # -- the refresh ------------------------------------------------------

    def current_settings(self) -> Dict[str, Any]:
        """Return the screen's settings, or ``{}`` when they will not collect.

        ``collect`` raises on a half-filled form, which is the normal state of
        a screen the user is still working on — and the strip has to keep
        working there, because that is exactly when it is useful.
        """
        self._collect_ok = False
        model = getattr(self._screen, "_settings_model", None)
        collect = getattr(model, "collect", None)
        if callable(collect):
            try:
                settings = dict(collect())
            except Exception:
                settings = None
            if settings is not None:
                self._collect_ok = True
                return settings
        widgets = self._widgets()
        return {key: _widget_value(widgets.get(key))
                for key in self._bound_settings()}

    def search_roots(self) -> Tuple[str, ...]:
        """Return the projects to look in, in preference order.

        This module's own folder first, then the folders the modules upstream
        of it last ran in.  The second is what makes a *blank* Measure screen
        find the plate Mask just finished: the registry lives in the project
        root, so without a candidate root there is no registry to ask.
        """
        roots: List[str] = []
        try:
            from .prefs import get_last_source, get_recent_sources
        except Exception:
            return ()
        keys: List[str] = [self.app_key]
        try:
            keys.extend(_ports.upstream_modules(self.app_key))
        except Exception:
            pass
        for key in keys:
            for candidate in (get_last_source(key),
                              *get_recent_sources(key, limit=4)):
                if candidate and candidate not in roots:
                    roots.append(candidate)
        return tuple(roots)

    def refresh(self, *, finished: bool = False) -> None:
        """Re-read the registry and redraw every row.

        Never raises: the strip is an aid, and an aid that can take a module
        screen down with it is worse than no aid.

        :param finished: a run just finished successfully, so offer the next
            step as well.
        """
        try:
            self._refresh(finished=finished)
        except Exception:
            LOG.exception("could not refresh the chaining strip for %s",
                          self.app_key)

    def _refresh(self, *, finished: bool) -> None:
        """The body of :meth:`refresh`, without the guard."""
        self._capture_edits()
        settings = self.current_settings()
        roots = self.search_roots()
        resolution = _chaining.resolve_settings(
            self.app_key, settings, roots=roots, pins=self._pins)
        self._held = dict(resolution.held)

        # Everything the resolution decided — a restored pin as much as a
        # chained default — goes into the field, but only where the field is
        # empty. That single rule is what makes a pin survive a restart (the
        # widget starts on its placeholder and the pin fills it) while never
        # overwriting anything the user can see.
        widgets = self._widgets()
        applied: Dict[str, Any] = {}
        for key in self._bound_settings():
            value = resolution.settings.get(key)
            if _chaining.is_empty_path(value):
                continue
            if not _chaining.is_empty_path(_widget_value(widgets.get(key))):
                continue
            applied[key] = value
        if applied:
            apply = getattr(self._screen, "apply_settings_dict", None)
            if callable(apply):
                apply(applied)
            self._offered.update(applied)
            self._seen.update(applied)

        self._draw_sources(resolution.inputs, resolution.filled)
        self._draw_pins(resolution.moved)
        self._draw_staleness(resolution.settings)
        self._draw_next(resolution.settings, finished=finished)
        # ``isHidden`` and not ``isVisible``: a widget whose window has not
        # been shown yet is not *visible*, so asking that question during the
        # screen's construction would answer "nothing to say" every time and
        # latch the strip hidden for the life of the screen.
        self.setVisible(any(not w.isHidden() for w in (
            self._source, self._pinned_row, self._stale, self._next_row)))

    def _draw_sources(self, inputs: Sequence[ChainedInput],
                      filled: Dict[str, ChainedInput]) -> None:
        """Row 1 — where the inputs come from."""
        if not inputs:
            self._source.hide()
            return
        parts = []
        for chained in inputs:
            verb = "using" if chained.setting in filled else "found"
            parts.append(f"{verb} {chained.kind} from "
                         f"{chained.producer} at {chained.artifact.path}")
        self._source.setText("Inputs: " + " · ".join(parts))
        self._source.setToolTip(
            "Resolved from the artifact registry — where the run that "
            "produced these actually wrote, not a guessed folder name.")
        self._source.show()

    def _draw_pins(self, moved: Sequence[HeldPin]) -> None:
        """Row 2 — a pinned path whose upstream has moved on."""
        if not moved:
            self._pinned_row.hide()
            return
        pin = moved[0]
        producer = pin.chained.producer if pin.chained else "the previous step"
        where = pin.chained.artifact.path if pin.chained else ""
        self._pinned.setText(
            f"{pin.setting} is set to {pin.value} — yours, and kept. "
            f"{producer} now writes to {where}.")
        self._pinned.setToolTip(
            "A path you entered is never overwritten. This is the offer, "
            "not the change.")
        self._pinned_row.show()

    def _draw_staleness(self, settings: Dict[str, Any]) -> None:
        """Row 3 — what is out of date, why, and what to do."""
        notes = self._staleness(settings)
        if not notes:
            self._stale.hide()
            self._fix.hide()
            return
        note = notes[0]
        more = len(notes) - 1
        self._stale.setText(
            f"⚠ {note.headline}" + (f" (+{more} more)" if more else ""))
        self._stale.setToolTip(
            note.detail or note.headline)
        self._stale.show()
        self._fix.setText(note.fix)
        self._fix.show()

    def _draw_next(self, settings: Dict[str, Any], *, finished: bool) -> None:
        """Row 4 — what can run on what this one just produced."""
        while self._next_layout.count() > 2:
            item = self._next_layout.takeAt(1)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if not finished:
            self._next_row.hide()
            self._last_steps = ()
            return
        try:
            steps = _chaining.next_steps(
                self.app_key, settings,
                root=_ports.project_root(settings, self.app_key))
        except Exception:
            LOG.exception("could not work out what comes after %s",
                          self.app_key)
            steps = ()
        self._last_steps = steps
        if not steps:
            self._next_row.hide()
            return
        from .app import APPS
        titles = {key: name for key, name, _d, _s in APPS}
        for index, step in enumerate(steps):
            title = titles.get(step.module, step.module.replace("_", " ").title())
            button = QPushButton(title if step.ok else f"{title} — not ready")
            button.setObjectName("PrimaryButton" if step.ok
                                 else "ChainingStepBlocked")
            button.setCursor(Qt.PointingHandCursor)
            button.setEnabled(bool(step.ok))
            button.setToolTip(
                f"Opens {title} with {', '.join(step.kinds) or 'this project'} "
                f"from this run."
                if step.ok else f"{step.blocked}\n\n{step.fix}")
            button.clicked.connect(
                lambda _checked=False, s=step: self._on_continue(s))
            self._next_layout.insertWidget(1 + index, button)
        self._next_row.show()

    # -- introspection, for tests and for the next module's Continue -------

    @property
    def steps(self) -> Tuple[NextStep, ...]:
        """The successors currently offered."""
        return self._last_steps

    @property
    def held(self) -> Dict[str, HeldPin]:
        """The settings keys a pin is holding, keyed by setting."""
        return dict(getattr(self, "_held", {}))

    def stale_notes(self) -> Tuple[StaleNote, ...]:
        """Return the staleness the strip would show right now."""
        return self._staleness(self.current_settings())

    def _staleness(self, settings: Dict[str, Any]) -> Tuple[StaleNote, ...]:
        """Ask the registry what is out of date around this module.

        The settings are only handed over for the hash comparison when
        ``collect()`` gave a whole dict.  A partial one hashes differently
        from the run that wrote the result, so passing it would report every
        result on a half-filled screen as "the settings changed" — a warning
        that is always on is a warning nobody reads.
        """
        return _chaining.staleness_notes(
            self.app_key, settings if self._collect_ok else None,
            root=_ports.project_root(settings, self.app_key))


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def chaining_bar(screen) -> Optional[ChainingBar]:
    """Return the strip installed on ``screen``, or None."""
    bar = getattr(screen, "_chaining_bar", None)
    return bar if isinstance(bar, ChainingBar) else None


def install_chaining(screen, *, pins=None) -> Optional[ChainingBar]:
    """Put a :class:`ChainingBar` above ``screen``'s Run row.

    The strip goes into the runtime panel immediately above the actions row,
    which is the last thing the eye crosses on its way to Run.  Both anchors
    (``_runtime_wrap`` and ``_actions_row``) are attributes ``AppScreen``
    keeps for exactly this kind of reach, so nothing here depends on the
    panel's internal layout order.

    :param screen: an ``AppScreen``.
    :param pins: a pin store, for tests.
    :returns: the strip, or None when this screen cannot carry one — a module
        with no declared ports, or a screen that failed to build its panels.
        Never raises: a screen that opens without the strip is the old
        behaviour, and that is always better than a screen that does not open.
    """
    try:
        app_key = str(getattr(screen, "app_key", ""))
        try:
            _ports.module_ports(app_key)
        except Exception:
            return None
        if chaining_bar(screen) is not None:
            return chaining_bar(screen)
        wrap = getattr(screen, "_runtime_wrap", None)
        actions = getattr(screen, "_actions_row", None)
        if wrap is None or actions is None:
            return None
        layout = wrap.layout()
        if layout is None:
            return None
        bar = ChainingBar(screen, pins=pins)
        index = layout.indexOf(actions)
        layout.insertWidget(index if index >= 0 else layout.count(), bar)
        screen._chaining_bar = bar
        bar.refresh()
        return bar
    except Exception:
        LOG.exception("could not install the chaining strip on %s",
                      getattr(screen, "app_key", "?"))
        return None


def _connect_host(screen, host) -> None:
    """Make the connections ``_build_screen`` makes on a generic AppScreen.

    Defensive on both sides: a host that does not define a slot is skipped
    rather than crashed on, because the unbound ``_build_screen`` is called
    against a stand-in host by the module smoke test and must keep working.
    """
    if host is None:
        return
    for signal_name, slot_name in HOST_CONNECTIONS.items():
        signal = getattr(screen, signal_name, None)
        slot = getattr(host, slot_name, None)
        if signal is None or not callable(slot):
            continue
        try:
            signal.connect(slot)
        except Exception:
            LOG.exception("could not connect %s to %s", signal_name, slot_name)


def _chained_app_screen(app_key: str, host=None):
    """Build the generic module screen and give it a chaining strip.

    Registered into :data:`spacr.qt.app.APP_FACTORIES`, which
    ``MainWindow._build_screen`` consults before its own chain — so the
    shared ``AppScreen`` needs no line about chaining, and a module that
    declares ports gets the strip by declaring them.

    :param app_key: the module key.
    :param host: the ``MainWindow``, when there is one.
    """
    from .screens.app_screen import AppScreen

    screen = AppScreen(app_key=app_key)
    _connect_host(screen, host)
    install_chaining(screen)
    return screen


def chained_app_keys() -> Tuple[str, ...]:
    """Return the app keys that get a chaining strip, sorted.

    Every module that both declares ports and is a registered app: the graph
    decides, not a list kept in step by hand.
    """
    from .app import APPS

    registered = {row[0] for row in APPS}
    return tuple(sorted(registered & set(_ports.known_modules())))


def register() -> bool:
    """Install the chaining strip on every ported module screen.

    Idempotent, and called by :func:`spacr.qt.register_self_registering_modules`
    after ``app.py`` has finished importing and before the first window is
    built.

    :returns: True when anything was registered.
    """
    from .app import APP_FACTORIES

    try:
        from .theme import register_widget_qss
        register_widget_qss("ChainingBar", _qss, replace=True)
    except Exception:
        LOG.exception("could not register the chaining strip's stylesheet")

    installed = False
    for key in chained_app_keys():
        existing = APP_FACTORIES.get(key)
        if existing is not None and existing is not _chained_app_screen:
            # Somebody else owns this screen — a plugin, or a module that
            # ships its own. Theirs wins; a strip is not worth overriding a
            # whole screen for.
            continue
        APP_FACTORIES[key] = _chained_app_screen
        installed = True
    return installed


def unregister() -> int:
    """Undo :func:`register`. Returns how many factories were removed."""
    from .app import APP_FACTORIES

    removed = 0
    for key in list(APP_FACTORIES):
        if APP_FACTORIES[key] is _chained_app_screen:
            APP_FACTORIES.pop(key)
            removed += 1
    return removed
