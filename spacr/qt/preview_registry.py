"""Which modules get a Live Preview, and how one is attached from outside.

Four modules have a preview — Mask, Measure, Timelapse, Motility — and each
one costs a thirteen-line arm in ``AppScreen._build_runtime_panel``, two
attribute names in a null-out block, and a row in a toggle table two hundred
lines further down. The fifth would cost the same, which is why there has
never been a fifth: the modules that would most obviously benefit from one
are the two whose entire job is "did the mask come out right" — Cellpose
Masks and Plaque Assay — and neither was worth touching the shared screen
for.

This module is the seam that makes the fifth free. A module declares a
preview here; the strip above the settings form grows a toggle for it; the
card is inserted above the Run row through the same ``_runtime_wrap`` /
``_actions_row`` anchors :mod:`spacr.qt.prerun` uses. Nothing inside
``AppScreen`` changes, and the four previews it already builds are left
alone — a module the shared screen has already served is skipped here rather
than given a second card.

**The sampling contract is inherited, not reimplemented.** The panels reached
through this registry are the shipped ones, which group a plate into image
sets from file names alone and open a bounded, reproducible random sample of
it. Nothing here enumerates, opens or lists a directory, so nothing here can
regress that. A new panel registered through this seam must keep the same
promise — see :mod:`spacr.qt.widgets.preview_controls`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QToolButton, QWidget

LOG = logging.getLogger("spacr.qt.preview_registry")


@dataclass(frozen=True)
class PreviewSpec:
    """One module's preview declaration.

    :ivar builder: ``"module:function"`` returning ``(panel, card)``, the
        shape every existing ``build_*_preview_card`` already has. Named
        rather than imported so declaring a preview costs no import at
        launch — a preview panel drags in the imaging stack.
    :ivar title: the toggle's label.
    :ivar tooltip: what the toggle promises.
    :ivar propagation: rename map applied to whatever the panel hands back
        through ``set_propagate_callback``, so a panel written for one
        module's setting names can serve another's.
    :ivar owned_by_screen: True for the four ``AppScreen`` already builds.
        They are declared here so this registry is the single answer to
        "which modules have a preview", and skipped at install time so they
        do not get a second card.
    """
    builder: str
    title: str = "Live preview"
    tooltip: str = ""
    propagation: Dict[str, str] = field(default_factory=dict)
    owned_by_screen: bool = False


#: app key -> its preview. The four marked ``owned_by_screen`` are built by
#: ``AppScreen`` itself; the rest are attached by :func:`install`.
PREVIEWS: Dict[str, PreviewSpec] = {
    "mask": PreviewSpec(
        builder="spacr.qt.screens.app_screen:_build_live_preview_card",
        owned_by_screen=True),
    "measure": PreviewSpec(
        builder="spacr.qt.screens.app_screen:_build_measure_preview_card",
        title="Crop preview", owned_by_screen=True),
    "timelapse": PreviewSpec(
        builder="spacr.qt.widgets.timelapse_preview:"
                "build_timelapse_preview_card",
        title="Track preview", owned_by_screen=True),
    "motility": PreviewSpec(
        builder="spacr.qt.widgets.motility_preview:"
                "build_motility_preview_card",
        title="Track preview", owned_by_screen=True),
    # -- attached through this seam ---------------------------------------
    #
    # Both of these run Cellpose over one field and are judged entirely by
    # whether the mask came out right, which is exactly the question the
    # Mask panel answers. Their settings even share its names — the panel
    # reads `diameter`, `flow_threshold` and `CP_prob` straight out of the
    # dict, which is what makes reuse honest rather than approximate.
    #
    # The reverse direction does need translating: the panel speaks Mask's
    # per-compartment names, and `cell_diameter` means nothing to a module
    # that has one object type and calls it `diameter`.
    "cellpose_masks": PreviewSpec(
        builder="spacr.qt.screens.app_screen:_build_live_preview_card",
        tooltip="Segment one sampled field with these settings before "
                "committing the plate.",
        propagation={
            "cell_diameter": "diameter",
            "cell_FT": "flow_threshold",
            "cell_CP_prob": "CP_prob",
            "model_name": "model_name",
            "normalize": "normalize",
        }),
    "analyze_plaques": PreviewSpec(
        builder="spacr.qt.screens.app_screen:_build_live_preview_card",
        tooltip="Check the plaque diameter and thresholds on one sampled "
                "field before running the assay.",
        propagation={
            "cell_diameter": "diameter",
            "cell_FT": "flow_threshold",
            "cell_CP_prob": "CP_prob",
        }),
}


def register_preview(app_key: str, spec: PreviewSpec,
                     *, replace: bool = False) -> PreviewSpec:
    """Declare a preview for ``app_key``.

    :param app_key: the module's app key.
    :param spec: its declaration.
    :param replace: overwrite an existing declaration instead of raising.
    :raises ValueError: on a second declaration without ``replace`` — two
        modules quietly claiming one key is the failure a registry exists to
        make loud.
    """
    key = str(app_key)
    if key in PREVIEWS and not replace:
        raise ValueError(
            f"a preview for {key!r} is already registered; pass "
            "replace=True if that is really what you mean")
    PREVIEWS[key] = spec
    return spec


def unregister_preview(app_key: str) -> bool:
    """Drop a declaration. ``True`` if there was one."""
    return PREVIEWS.pop(str(app_key), None) is not None


def preview_app_keys() -> Tuple[str, ...]:
    """Every module with a preview, however it is attached."""
    return tuple(PREVIEWS)


def _resolve(builder: str) -> Optional[Callable[[Any], Tuple[Any, Any]]]:
    module_name, _, func_name = str(builder).partition(":")
    if not module_name or not func_name:
        return None
    import importlib
    try:
        return getattr(importlib.import_module(module_name), func_name)
    except Exception:
        LOG.debug("could not resolve preview builder %r", builder,
                  exc_info=True)
        return None


class _PreviewHost(QObject):
    """Owns one attached preview: its card, its toggle, and the translation.

    A ``QObject`` parented to the screen, with bound-method slots — the
    alternative, closures captured by the toggle, keeps the screen alive
    through its own button.
    """

    def __init__(self, screen: QWidget, spec: PreviewSpec, panel, card):
        super().__init__(screen)
        self._screen = screen
        self._spec = spec
        self.panel = panel
        self.card = card
        self._primed = False

    def on_toggled(self, on: bool) -> None:
        """Show or hide the preview card."""
        if on and not self._primed:
            self.prime()
        self.card.setVisible(bool(on))

    def prime(self) -> None:
        """Push the module's current settings into the panel, once.

        Deferred to the first time the card is shown rather than done at
        install: reading the form costs a pass over every widget, and a
        preview nobody opened should cost nothing.
        """
        self._primed = True
        apply_settings = getattr(self.panel, "apply_settings", None)
        model = getattr(self._screen, "_settings_model", None)
        if not callable(apply_settings) or model is None:
            return
        try:
            apply_settings(model.collect())
        except Exception:
            LOG.debug("could not prime the preview for %r",
                      getattr(self._screen, "app_key", "?"), exc_info=True)

    def on_propagate(self, values: Dict[str, Any]) -> None:
        """Translate the panel's setting names, then write them to the form.

        Unmapped names are dropped rather than passed through: a module that
        has no ``cell_channel`` gains nothing from being offered one, and
        ``set_value_for_key`` would return False for each in silence,
        leaving "propagate" looking like it worked.
        """
        rename = self._spec.propagation
        model = getattr(self._screen, "_settings_model", None)
        if model is None:
            return
        for source, value in dict(values or {}).items():
            target = rename.get(source) if rename else source
            if target is None:
                continue
            try:
                model.set_value_for_key(target, value)
            except Exception:
                LOG.debug("could not propagate %r", target, exc_info=True)


def install(screen: QWidget) -> Optional[_PreviewHost]:
    """Attach ``screen``'s declared preview, if it has one to attach.

    Returns ``None`` when the module declares no preview, when
    ``AppScreen`` already built one for it, when the screen has no runtime
    panel to insert into, or when one is already installed. Never raises: a
    missing preview must not cost anyone a module.
    """
    if getattr(screen, "_registry_preview", None) is not None:
        return screen._registry_preview
    app_key = str(getattr(screen, "app_key", "") or "")
    spec = PREVIEWS.get(app_key)
    if spec is None or spec.owned_by_screen:
        return None
    build = _resolve(spec.builder)
    if build is None:
        return None
    try:
        panel, card = build(screen)
    except Exception:
        LOG.debug("preview builder failed for %r", app_key, exc_info=True)
        return None
    if not _insert_above_actions(screen, card):
        card.setParent(None)
        card.deleteLater()
        return None
    card.setVisible(False)

    host = _PreviewHost(screen, spec, panel, card)
    register_cb = getattr(panel, "set_propagate_callback", None)
    if callable(register_cb):
        register_cb(host.on_propagate)

    toggle = QToolButton()
    toggle.setObjectName("SettingsPreviewToggle")
    toggle.setText(spec.title)
    toggle.setCheckable(True)
    toggle.setCursor(Qt.PointingHandCursor)
    toggle.setToolTip(spec.tooltip or
                      "Show a preview of what these settings produce.")
    toggle.toggled.connect(host.on_toggled)
    host.toggle = toggle

    bar = getattr(screen, "_settings_search", None)
    if bar is not None and hasattr(bar, "add_trailing_widget"):
        bar.add_trailing_widget(toggle)
    else:
        # No strip on this screen — put the toggle above the card so the
        # preview is still reachable rather than permanently hidden.
        toggle.setParent(screen)
        _insert_above_actions(screen, toggle)
    screen._registry_preview = host
    return host


def _insert_above_actions(screen: QWidget, widget: QWidget) -> bool:
    """Put ``widget`` in the runtime panel just above the Run row.

    Both anchors are attributes ``AppScreen`` keeps for exactly this kind of
    reach, so nothing here depends on that panel's internal layout order.
    The same helper :mod:`spacr.qt.prerun` uses, for the same reason: above
    the actions row is the last thing the eye crosses on the way to Run.
    """
    wrap = getattr(screen, "_runtime_wrap", None)
    actions = getattr(screen, "_actions_row", None)
    if wrap is None or actions is None:
        return False
    layout = wrap.layout()
    if layout is None:
        return False
    index = layout.indexOf(actions)
    layout.insertWidget(index if index >= 0 else layout.count(), widget)
    return True


class _StackWatcher(QObject):
    """Attaches a declared preview to each screen as it is first shown."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window = window

    def on_current_changed(self, _index: int) -> None:
        """Install into whatever screen the stack just switched to."""
        self.install_current()

    def install_current(self) -> Optional[_PreviewHost]:
        """Install into the stack's current widget, if it declares one."""
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            return None
        if screen is None:
            return None
        return install(screen)


def install_window_hooks(window: QMainWindow) -> Optional[_StackWatcher]:
    """Follow ``window``'s screen stack, attaching declared previews.

    Called once from :func:`spacr.qt.shortcuts.install`, after the settings
    strip's own hook so the toggle has somewhere to go.
    """
    stack = getattr(window, "_stack", None)
    if stack is None:
        return None
    if getattr(window, "_preview_watcher", None) is not None:
        return window._preview_watcher
    watcher = _StackWatcher(window)
    try:
        stack.currentChanged.connect(watcher.on_current_changed)
    except Exception:
        LOG.debug("could not follow the screen stack", exc_info=True)
        return None
    window._preview_watcher = watcher
    QTimer.singleShot(0, watcher.install_current)
    return watcher
