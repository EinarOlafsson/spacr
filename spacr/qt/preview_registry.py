"""Which modules get a Live Preview, and how one is attached from outside.

Four modules have a preview: Mask, Measure, Timelapse and Motility. Each one
costs a thirteen-line arm in ``AppScreen._build_runtime_panel``, two attribute
names in a null-out block, and a row in a toggle table two hundred lines
further down.

A fifth would cost the same, which is why there has never been one. The two
modules that would benefit most are the ones whose entire job is "did the mask
come out right", Cellpose Masks and Plaque Assay, and neither was worth
touching the shared screen for.

This module is the seam that makes the fifth free. A module declares a
preview here; the strip above the settings form grows a toggle for it; the
card is inserted above the Run row through the same ``_runtime_wrap`` /
``_actions_row`` anchors :mod:`spacr.qt.prerun` uses. Nothing inside
``AppScreen`` changes, and the four previews it already builds are left
alone — a module the shared screen has already served is skipped here rather
than given a second card.

A module that has been FOLDED into a host reaches the same machinery from
the other end. Its own screen is not built any more, so ``install`` --
which answers for the screen's own key -- can never attach its panel;
:func:`attach_folded` lets the host ask for it by name and keeps it hidden
behind whatever the host uses to reveal the rest of that module. Mask
Generation's tracking switch is the case it was written for.

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
    :ivar owned_by_screen: True for the ones ``AppScreen`` already builds.
        They are declared here so this registry is the single answer to
        "which modules have a preview", and skipped at install time so they
        do not get a second card.
    :ivar fill: ``"module:function"`` taking ``(host, card)`` and building
        the panel into the card, returning it. Given, a preview ATTACHED
        through this registry builds only its card at install --
        ``builder`` is called with ``panel_later=True`` -- and the panel the
        first time the card is shown or the panel is asked for, so a hidden
        preview costs a module's open nothing.
    """
    builder: str
    title: str = "Live preview"
    tooltip: str = ""
    propagation: Dict[str, str] = field(default_factory=dict)
    owned_by_screen: bool = False
    fill: str = ""


#: app key -> its preview. The ones marked ``owned_by_screen`` are built by
#: ``AppScreen`` itself; the rest are attached by :func:`install`.
PREVIEWS: Dict[str, PreviewSpec] = {
    'host_pathogen': PreviewSpec(
        builder='spacr.qt.widgets.host_pathogen_preview:build_host_pathogen_preview_card',
        fill='spacr.qt.widgets.host_pathogen_preview:fill_host_pathogen_preview_card',
        title='Live preview',
        tooltip='Inspect one measured field, its host/vacuole/parasite masks and recruitment results.'),
    "mask": PreviewSpec(
        builder="spacr.qt.screens.app_screen:_build_live_preview_card",
        owned_by_screen=True),
    "measure": PreviewSpec(
        builder="spacr.qt.screens.app_screen:_build_measure_preview_card",
        title="Crop preview", owned_by_screen=True),
    "timelapse": PreviewSpec(
        builder="spacr.qt.widgets.timelapse_preview:"
                "build_timelapse_preview_card",
        fill="spacr.qt.widgets.timelapse_preview:"
             "_fill_timelapse_preview_card",
        title="Track preview", owned_by_screen=True),
    "motility": PreviewSpec(
        builder="spacr.qt.widgets.motility_preview:"
                "build_motility_preview_card",
        title="Track preview", owned_by_screen=True),
    "cellpose_masks": PreviewSpec(
        builder="spacr.qt.screens.app_screen:_build_live_preview_card",
        tooltip="Segment one sampled field with these settings before "
                "committing the plate.",
        propagation={
            "cell_diameter": "diameter",
            "cell_flow_threshold": "flow_threshold",
            "cell_cellprob_threshold": "CP_prob",
            "model_name": "model_name",
            # The run loads `custom_model` over `model_name` when it is set;
            # the panel writes it back only if it was (333).
            "custom_model": "custom_model",
            "normalize": "normalize",
        }),
    "analyze_plaques": PreviewSpec(
        builder="spacr.qt.widgets.plaque_preview:build_plaque_preview_card",
        owned_by_screen=True,
        tooltip="Check the plaque diameter and thresholds on one sampled "
                "field before running the assay.",
        propagation={
            "cell_diameter": "diameter",
            "cell_flow_threshold": "flow_threshold",
            "cell_cellprob_threshold": "CP_prob",
            # The plaque run segments with `plaque_model`, never
            # `model_name`; the panel writes it only for a checkpoint the
            # user picked (333).
            "plaque_model": "plaque_model",
            "diameter": "diameter",
            "flow_threshold": "flow_threshold",
            "CP_prob": "CP_prob",
            "plaque_mode": "plaque_mode",
            "figure_detector": "figure_detector",
            "figure_imgsz": "figure_imgsz",
            "figure_confidence": "figure_confidence",
            "figure_read_text": "figure_read_text",
            "confirm_annotations": "confirm_annotations",
            "text_reach_above": "text_reach_above",
            "text_reach_left": "text_reach_left",
            "text_reach_below": "text_reach_below",
            "text_use_above": "text_use_above",
            "text_use_left": "text_use_left",
            "text_use_below": "text_use_below",
            "text_panel_reach": "text_panel_reach",
            "text_min_confidence": "text_min_confidence",
            "text_ignore": "text_ignore",
            "text_order": "text_order",
            "text_separator": "text_separator",
            "text_reread": "text_reread",
            "text_reread_scale": "text_reread_scale",
            "src": "src",
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
    """Import a preview builder named as ``module:function``.

    :param builder: the dotted module path and function name.
    :returns: the callable, or ``None`` when the string is malformed or the
        import fails -- a module whose preview cannot be resolved still
        opens, without one.
    """
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

    def __init__(self, screen: QWidget, spec: PreviewSpec, panel, card,
                 fill: Optional[Callable[[Any, Any], Any]] = None):
        """Bind one preview to the screen that shows it.

        :param screen: the module screen the preview belongs to, and this
            object's Qt parent.
        :param spec: what the preview is and how to build it.
        :param panel: the preview widget itself, or ``None`` when ``fill``
            builds it later.
        :param card: the container the panel sits in, shown and hidden by
            :meth:`on_toggled`.
        :param fill: builds the panel into ``card`` the first time the card
            is shown or :attr:`panel` is read; see :attr:`PreviewSpec.fill`.

        Nothing is built here. The panel is PRIMED on first show, so a screen
        with a preview costs nothing until the user opens it.
        """
        super().__init__(screen)
        self._screen = screen
        self._spec = spec
        self._panel = panel
        self._fill = fill
        self.card = card
        self._primed = False
        if fill is not None:
            build_later = getattr(card, "build_body_when_first_shown", None)
            if callable(build_later):
                build_later(self._build_panel)
        else:
            self._connect_panel()

    @property
    def panel(self):
        """The preview widget, built first if it is still waiting."""
        if self._fill is not None:
            self._build_panel()
        return self._panel

    @panel.setter
    def panel(self, value) -> None:
        """Replace the preview widget; a panel still waiting is not built."""
        self._fill = None
        self._panel = value

    def panel_is_built(self) -> bool:
        """Whether the panel exists yet. For tests and diagnostics."""
        return self._fill is None

    def _connect_panel(self) -> None:
        """Route the panel's propagated settings to the form."""
        register_cb = getattr(self._panel, "set_propagate_callback", None)
        if callable(register_cb):
            register_cb(self.on_propagate)

    def _build_panel(self) -> None:
        """Build the panel into its card, once, as install used to.

        It is then translated and polished; see
        :func:`_dress_a_late_panel`.
        """
        fill, self._fill = self._fill, None
        if fill is None:
            return
        self._panel = fill(self._screen, self.card)
        self._connect_panel()
        _dress_a_late_panel(self._screen, self._panel)

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
    host = _attach(screen, app_key, spec)
    if host is not None:
        screen._registry_preview = host
    return host


def attach_folded(screen: QWidget, app_key: str) -> Optional[_PreviewHost]:
    """Attach ANOTHER module's declared preview to ``screen``.

    A module folded into a host as settings categories brings its panel
    with it. Mask Generation has the Cellpose live preview and no track
    preview; Timelapse's whole preview is the tracking one, built by the
    screen the fold means nobody opens any more. Without this the switch
    would reveal the tracking settings and nothing that shows what they
    do -- a capability the tile had and the button did not, which is the
    one thing a fold must not cost.

    ``owned_by_screen`` is deliberately ignored: it means "``AppScreen``
    builds this one for its own key", and the point here is that the key
    is somebody else's. The card and its toggle both start hidden; the
    host reveals them when the fold is switched on.

    :param screen: the HOST screen.
    :param app_key: the folded module's key.
    :returns: the host, or None when there is nothing to attach.
    """
    key = str(app_key)
    attached = getattr(screen, "_folded_previews", None)
    if attached is None:
        attached = screen._folded_previews = {}
    if key in attached:
        return attached[key]
    spec = PREVIEWS.get(key)
    if spec is None:
        return None
    host = _attach(screen, key, spec)
    if host is None:
        return None
    host.toggle.setVisible(False)
    attached[key] = host
    return host


def _attach(screen: QWidget, app_key: str,
            spec: PreviewSpec) -> Optional[_PreviewHost]:
    """Build ``spec``'s card, insert it hidden, and give it a toggle."""
    build = _resolve(spec.builder)
    if build is None:
        return None
    fill = _resolve(spec.fill) if spec.fill else None
    try:
        if fill is not None:
            panel, card = build(screen, panel_later=True)
        else:
            panel, card = build(screen)
    except Exception:
        LOG.debug("preview builder failed for %r", app_key, exc_info=True)
        return None
    if not _insert_above_console(screen, card):
        card.setParent(None)
        card.deleteLater()
        return None
    card.setVisible(False)
    adopt = getattr(screen, "adopt_runtime_pane", None)
    if callable(adopt):
        try:
            adopt(card, focus=True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not name the preview's pane", exc_info=True)

    from .widgets.preview_refresh import install_refresh_button

    host = _PreviewHost(screen, spec, panel, card, fill)
    if fill is None:
        install_refresh_button(screen, card, panel)
    else:
        install_refresh_button(screen, card, None,
                               panel_getter=lambda: host.panel)

    if app_key == 'host_pathogen':
        from .widgets.ai_toggle_label import AiToggleLabel

        toggle = AiToggleLabel(screen, text='Live', tooltip=spec.tooltip)
    else:
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
    heading = getattr(screen, '_actions_heading_row', None)
    if app_key == 'host_pathogen' and heading is not None:
        heading.addWidget(toggle)
    elif bar is not None and hasattr(bar, "add_trailing_widget"):
        bar.add_trailing_widget(toggle)
    else:
        toggle.setParent(screen)
        _insert_above_actions(screen, toggle)
    return host


def _dress_a_late_panel(screen: QWidget, panel: QWidget) -> None:
    """Translate and polish a preview panel built after its screen opened.

    Through the screen's own hook, the one
    :class:`~spacr.qt.screens.app_screen.AppScreen` runs over its deferred
    parts; a screen without it gets nothing. The layout-container sweep
    that hook's sibling runs is deliberately NOT run: a card attached here
    always arrived after the screen's sweep, so its containers were never
    made transparent at install either. The language pass is new -- a
    card inserted into the runtime splitter was never reached by the
    screen's late-caption watcher, so an attached preview opened in English
    on a translated screen.

    Never raises.
    """
    hook = getattr(screen, "_translate_a_late_part", None)
    if callable(hook):
        try:
            hook(panel)
        except Exception:
            LOG.debug("could not translate the preview panel", exc_info=True)


def _insert_above_console(screen: QWidget, widget: QWidget) -> bool:

    """Put a preview card in the runtime splitter, directly ABOVE the console.

    WHY IT WOULD OTHERWISE BE UNDER IT. Every preview card went through
    :func:`_insert_above_actions`, which puts a widget in the runtime panel just above
    the Run row -- and the figures/console splitter is added to that same panel BEFORE
    the actions row. So "above the Run button" is below the console, and the preview
    landed under the log it was supposed to be read beside.

    Mask never showed the bug and that is why it went unnoticed: its screen builds the
    live preview into the splitter itself, between the figures and the console, and
    never calls this path. The modules that get their preview from the registry --
    Plaque Assay and Cellpose Masks -- got the Run-row placement instead, so the same
    card sat in two different places depending on which screen mounted it.

    The splitter is the right home rather than a different index in the panel: a card
    above the console INSIDE it can be resized against the console, which is the whole
    reason Mask's is there.

    :param screen: the module screen.
    :param widget: the card to insert.
    :returns: whether it went into the splitter. False means the caller should fall
        back, and :func:`_insert_above_actions` is still that fallback -- a screen with
        no splitter, or one whose console is not in it, is better off with the preview
        above the Run row than with no preview at all.
    """
    splitter = getattr(screen, "_runtime_splitter", None)
    console = getattr(screen, "_console_wrap", None)
    if splitter is None or console is None:
        return _insert_above_actions(screen, widget)
    try:
        index = splitter.indexOf(console)
    except (AttributeError, RuntimeError):
        return _insert_above_actions(screen, widget)
    if index < 0:
        return _insert_above_actions(screen, widget)
    splitter.insertWidget(index, widget)
    return True


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
    holder = actions.parentWidget()
    if (holder is not None and holder is not wrap
            and wrap.isAncestorOf(holder)
            and holder.layout() is not None
            and holder.layout().indexOf(actions) >= 0):
        layout = holder.layout()
    if layout is None:
        return False
    index = layout.indexOf(actions)
    layout.insertWidget(index if index >= 0 else layout.count(), widget)
    return True


class _StackWatcher(QObject):
    """Attaches a declared preview to each screen as it is first shown."""

    def __init__(self, window: QMainWindow):
        """Watch a window's stack and install into each screen as it is shown.

        :param window: the main window. Its stack is read at install time,
            not here, so this works for screens created after the watcher --
            and it is the QObject PARENT, so a currentChanged arriving during
            teardown cannot reach a watcher holding a deleted stack.
        """
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
