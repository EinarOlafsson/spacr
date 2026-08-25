"""Attaching a preview from outside, and every way that can fail quietly.

The promise this seam makes is in its own docstring: "a missing preview must
not cost anyone a module". That is a promise about failure paths -- a builder
that cannot be imported, a builder that raises, a screen with no runtime
panel to insert into, a settings model that rejects a propagated value -- and
it is worth nothing unless each of those has been walked.

The screens here are real ``QWidget``s carrying the same two anchor
attributes ``AppScreen`` exposes (``_runtime_wrap`` and ``_actions_row``),
built one attribute short at a time. That is what makes the negative results
mean something: the insertion really is being asked to find an anchor that is
really not there.
"""

import pytest
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from spacr.qt.preview_registry import (PREVIEWS, PreviewSpec, _PreviewHost,
                                       _resolve, _StackWatcher, install,
                                       install_window_hooks,
                                       preview_app_keys, register_preview,
                                       unregister_preview)


# ---------------------------------------------------------------------------
# stand-ins for the pieces of a real screen this module reaches into
# ---------------------------------------------------------------------------

class _Model:
    """The settings model's two-method surface, and nothing else."""

    def __init__(self, values=None, refuse=False):
        self.values = dict(values or {})
        self.written = {}
        self.refuse = refuse

    def collect(self):
        return dict(self.values)

    def set_value_for_key(self, key, value):
        if self.refuse:
            raise KeyError(key)
        self.written[key] = value
        return True


class _Panel(QWidget):
    """A preview panel: takes settings in, hands propagated values back."""

    def __init__(self, blow_up=False):
        super().__init__()
        self.applied = []
        self.callback = None
        self.blow_up = blow_up

    def apply_settings(self, settings):
        if self.blow_up:
            raise RuntimeError("the imaging stack is not installed")
        self.applied.append(dict(settings))

    def set_propagate_callback(self, fn):
        self.callback = fn


class _Screen(QWidget):
    """A screen with the two anchors `AppScreen` keeps for this kind of reach."""

    def __init__(self, app_key, *, anchored=True, model=None, bar=None):
        super().__init__()
        self.app_key = app_key
        self._settings_model = model
        self._settings_search = bar
        if anchored:
            self._runtime_wrap = QWidget(self)
            layout = QVBoxLayout(self._runtime_wrap)
            self._actions_row = QLabel("Run", self._runtime_wrap)
            layout.addWidget(self._actions_row)


#: Resolved by name through the registry, exactly as a real builder is.
_LAST_BUILT = {}


def _build_test_card(screen):
    """A builder with the shape every `build_*_preview_card` already has."""
    panel = _Panel()
    card = QWidget()
    _LAST_BUILT["panel"] = panel
    _LAST_BUILT["card"] = card
    return panel, card


def _build_that_raises(screen):
    raise RuntimeError("this preview needs a GPU")


_HERE = "tests.qt.test_cov_w2_2_preview_registry"


# ---------------------------------------------------------------------------
# the declaration table
# ---------------------------------------------------------------------------

def test_a_second_declaration_for_one_key_is_loud(monkeypatch):
    """Registering twice raises rather than silently replacing.

    Two modules quietly claiming one key is the failure a registry exists to
    make visible.
    """
    spec = PreviewSpec(builder=f"{_HERE}:_build_test_card")
    monkeypatch.setitem(PREVIEWS, "w2_2_probe", spec)

    with pytest.raises(ValueError) as raised:
        register_preview("w2_2_probe", spec)
    assert "w2_2_probe" in str(raised.value)
    assert "replace=True" in str(raised.value)


def test_replacing_a_declaration_is_possible_when_asked_for(monkeypatch):
    """`replace=True` is the way to mean it, and it is honoured."""
    first = PreviewSpec(builder=f"{_HERE}:_build_test_card", title="One")
    second = PreviewSpec(builder=f"{_HERE}:_build_test_card", title="Two")
    monkeypatch.setitem(PREVIEWS, "w2_2_probe", first)

    assert register_preview("w2_2_probe", second, replace=True) is second
    assert PREVIEWS["w2_2_probe"].title == "Two"
    assert "w2_2_probe" in preview_app_keys()

    assert unregister_preview("w2_2_probe") is True
    assert unregister_preview("w2_2_probe") is False
    assert "w2_2_probe" not in preview_app_keys()


# ---------------------------------------------------------------------------
# resolving a builder by name
# ---------------------------------------------------------------------------

def test_a_builder_name_with_no_function_half_resolves_to_nothing():
    """A name that is not `module:function` is not guessed at."""
    assert _resolve("") is None
    assert _resolve("spacr.qt.screens.app_screen") is None
    assert _resolve(":_build_live_preview_card") is None


def test_a_builder_that_cannot_be_imported_resolves_to_nothing():
    """A missing module or a missing attribute is None, not an exception.

    Declaring a preview costs no import at launch precisely because the
    builder is a name; the price is that the name can be wrong, and being
    wrong must not take a module out.
    """
    assert _resolve("spacr.qt.no_such_preview_module:build") is None
    assert _resolve("spacr.qt.preview_registry:no_such_function") is None
    # and a good one really does resolve, so the negatives above mean something
    assert _resolve(f"{_HERE}:_build_test_card") is _build_test_card


# ---------------------------------------------------------------------------
# installing
# ---------------------------------------------------------------------------

def test_a_module_the_screen_already_serves_gets_no_second_card(qapp):
    """The four `AppScreen` builds itself are declared but skipped here."""
    for key in ("mask", "measure", "timelapse", "motility"):
        assert PREVIEWS[key].owned_by_screen is True
        assert install(_Screen(key)) is None


def test_a_module_with_no_declaration_gets_nothing(qapp):
    """An undeclared module is not an error, it is simply not previewed."""
    assert install(_Screen("no_such_module")) is None
    assert install(_Screen("")) is None


def test_a_builder_that_will_not_resolve_costs_nobody_a_module(qapp,
                                                               monkeypatch):
    """A bad builder name leaves the screen untouched and returns None."""
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder="nowhere.at.all:build"))
    screen = _Screen("w2_2_probe")
    assert install(screen) is None
    assert getattr(screen, "_registry_preview", None) is None


def test_a_builder_that_raises_costs_nobody_a_module(qapp, monkeypatch):
    """An exception inside the builder is logged and swallowed."""
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_that_raises"))
    screen = _Screen("w2_2_probe")
    assert install(screen) is None
    assert getattr(screen, "_registry_preview", None) is None


def test_a_screen_with_no_runtime_panel_gets_no_orphan_card(qapp,
                                                            monkeypatch):
    """With nowhere to insert, the built card is disowned rather than leaked.

    A card left parented to nothing is a top-level window: it would appear as
    a stray frame on the desktop.
    """
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card"))
    screen = _Screen("w2_2_probe", anchored=False)

    assert install(screen) is None
    assert _LAST_BUILT["card"].parent() is None
    assert getattr(screen, "_registry_preview", None) is None


def test_an_installed_preview_lands_above_the_run_row(qapp, monkeypatch):
    """The card goes in immediately before the actions row, hidden."""
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card",
                                    title="Try it",
                                    tooltip="see what comes out"))
    screen = _Screen("w2_2_probe", model=_Model({"cell_diameter": 30}))

    host = install(screen)
    assert host is not None
    layout = screen._runtime_wrap.layout()
    assert layout.indexOf(host.card) < layout.indexOf(screen._actions_row)
    assert host.card.isVisible() is False
    assert host.toggle.text() == "Try it"
    assert host.toggle.toolTip() == "see what comes out"
    assert host.toggle.isCheckable() is True

    # installed once, and asking again returns the same host
    assert install(screen) is host


def test_the_toggle_shows_the_card_and_primes_it_once(qapp, monkeypatch):
    """Priming is deferred to the first show, and does not repeat."""
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card"))
    model = _Model({"cell_diameter": 30, "cell_FT": 0.4})
    screen = _Screen("w2_2_probe", model=model)
    host = install(screen)
    panel = _LAST_BUILT["panel"]

    assert panel.applied == [], "a preview nobody opened cost a form read"

    host.toggle.setChecked(True)
    assert panel.applied == [{"cell_diameter": 30, "cell_FT": 0.4}]

    host.toggle.setChecked(False)
    host.toggle.setChecked(True)
    assert len(panel.applied) == 1, "the form was read again on a re-show"


def test_a_screen_with_no_settings_strip_still_reaches_its_toggle(qapp,
                                                                  monkeypatch):
    """With no strip to hang it on, the toggle goes into the runtime panel.

    A toggle with nowhere to go would leave the preview permanently hidden,
    which is worse than not declaring one.
    """
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card"))
    screen = _Screen("w2_2_probe", model=_Model())
    host = install(screen)

    layout = screen._runtime_wrap.layout()
    assert layout.indexOf(host.toggle) >= 0
    assert host.toggle.parent() is screen._runtime_wrap


def test_a_settings_strip_takes_the_toggle_when_there_is_one(qapp,
                                                             monkeypatch):
    """With a strip present the toggle joins it rather than the panel."""
    class _Bar:
        def __init__(self):
            self.trailing = []

        def add_trailing_widget(self, widget):
            self.trailing.append(widget)

    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card"))
    bar = _Bar()
    screen = _Screen("w2_2_probe", model=_Model(), bar=bar)
    host = install(screen)

    assert bar.trailing == [host.toggle]
    assert screen._runtime_wrap.layout().indexOf(host.toggle) == -1


# ---------------------------------------------------------------------------
# priming and propagating
# ---------------------------------------------------------------------------

def test_priming_a_panel_that_cannot_take_settings_does_nothing(qapp):
    """No `apply_settings`, or no model, is a no-op rather than a crash."""
    spec = PreviewSpec(builder=f"{_HERE}:_build_test_card")

    bare = _PreviewHost(_Screen("x", model=_Model()), spec, QWidget(),
                        QWidget())
    bare.prime()                      # panel has no apply_settings

    modelless = _PreviewHost(_Screen("x", model=None), spec, _Panel(),
                             QWidget())
    modelless.prime()
    assert modelless.panel.applied == []


def test_a_panel_that_throws_while_priming_leaves_the_screen_alive(qapp):
    """An exception inside the panel's own apply is logged, not raised."""
    spec = PreviewSpec(builder=f"{_HERE}:_build_test_card")
    host = _PreviewHost(_Screen("x", model=_Model({"a": 1})), spec,
                        _Panel(blow_up=True), QWidget())
    host.prime()                      # must not raise
    assert host._primed is True


def test_propagation_renames_what_it_can_and_drops_what_it_cannot(qapp,
                                                                  monkeypatch):
    """Mapped names are written under the module's own spelling.

    An unmapped name is dropped rather than passed through: the form would
    refuse it silently and "propagate" would look like it had worked.
    """
    spec = PREVIEWS["cellpose_masks"]
    model = _Model()
    host = _PreviewHost(_Screen("cellpose_masks", model=model), spec,
                        _Panel(), QWidget())

    host.on_propagate({"cell_diameter": 42, "cell_FT": 0.7,
                       "cell_channel": 1})

    assert model.written == {"diameter": 42, "flow_threshold": 0.7}
    assert "cell_channel" not in model.written


def test_propagation_with_no_rename_map_passes_names_straight_through(qapp):
    """A spec that declares no translation writes the names it was given."""
    spec = PreviewSpec(builder=f"{_HERE}:_build_test_card")
    model = _Model()
    host = _PreviewHost(_Screen("x", model=model), spec, _Panel(), QWidget())

    host.on_propagate({"diameter": 12})
    assert model.written == {"diameter": 12}


def test_propagating_into_a_screen_with_no_model_does_nothing(qapp):
    """Nothing to write to is a no-op, not an attribute error."""
    spec = PreviewSpec(builder=f"{_HERE}:_build_test_card")
    host = _PreviewHost(_Screen("x", model=None), spec, _Panel(), QWidget())
    host.on_propagate({"diameter": 12})      # must not raise
    host.on_propagate(None)


def test_a_form_that_refuses_a_value_does_not_stop_the_rest(qapp):
    """One rejected setting is logged; the others are still written."""
    spec = PreviewSpec(builder=f"{_HERE}:_build_test_card")
    model = _Model(refuse=True)
    host = _PreviewHost(_Screen("x", model=model), spec, _Panel(), QWidget())

    host.on_propagate({"diameter": 12, "flow_threshold": 0.4})
    assert model.written == {}


def test_the_panel_is_wired_to_the_form_at_install(qapp, monkeypatch):
    """A panel offering `set_propagate_callback` gets the host's translator."""
    monkeypatch.setitem(
        PREVIEWS, "w2_2_probe",
        PreviewSpec(builder=f"{_HERE}:_build_test_card",
                    propagation={"cell_diameter": "diameter"}))
    model = _Model()
    screen = _Screen("w2_2_probe", model=model)
    host = install(screen)

    assert _LAST_BUILT["panel"].callback == host.on_propagate
    _LAST_BUILT["panel"].callback({"cell_diameter": 19})
    assert model.written == {"diameter": 19}


# ---------------------------------------------------------------------------
# following the screen stack
# ---------------------------------------------------------------------------

class _Stack(QWidget):
    """A stand-in stack with the one signal and the one method used here."""

    def __init__(self, current=None):
        super().__init__()
        self._current = current

    def currentWidget(self):
        return self._current


def test_a_window_with_no_stack_is_not_followed(qapp):
    """Nothing to follow returns None rather than raising."""
    window = QWidget()
    assert install_window_hooks(window) is None


def test_the_watcher_ignores_a_stack_it_cannot_read(qapp):
    """A window whose stack attribute throws yields no install."""
    class _Hostile(QWidget):
        @property
        def _stack(self):
            raise RuntimeError("the window is being torn down")

    watcher = _StackWatcher(QWidget())
    watcher._window = _Hostile()
    assert watcher.install_current() is None


def test_an_empty_stack_installs_nothing(qapp):
    """A stack showing no widget is not a screen to install into."""
    window = QWidget()
    window._stack = _Stack(current=None)
    watcher = _StackWatcher(window)
    assert watcher.install_current() is None


def test_switching_screens_installs_into_the_new_one(qapp, monkeypatch):
    """The watcher's slot attaches whatever the stack just switched to."""
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card"))
    screen = _Screen("w2_2_probe", model=_Model())
    window = QWidget()
    window._stack = _Stack(current=screen)
    watcher = _StackWatcher(window)

    watcher.on_current_changed(0)
    assert getattr(screen, "_registry_preview", None) is not None


def test_a_stack_whose_signal_cannot_be_connected_is_given_up_on(qapp):
    """A stack with no `currentChanged` leaves no half-installed watcher."""
    class _Signalless:
        currentChanged = None

    window = QWidget()
    window._stack = _Signalless()
    assert install_window_hooks(window) is None
    assert getattr(window, "_preview_watcher", None) is None


def test_a_runtime_panel_with_no_layout_is_no_anchor(qapp, monkeypatch):
    """A wrap that has not been laid out yet cannot take the card."""
    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card"))
    screen = _Screen("w2_2_probe", anchored=False, model=_Model())
    screen._runtime_wrap = QWidget(screen)     # no layout on it
    screen._actions_row = QLabel("Run", screen._runtime_wrap)

    assert screen._runtime_wrap.layout() is None
    assert install(screen) is None


def test_a_real_stack_is_followed_and_followed_only_once(qapp, qtbot,
                                                         monkeypatch):
    """The hook connects to the stack's signal and installs on the way in.

    Driven through a real ``QStackedWidget`` so the connection, the deferred
    first install, and the switch that follows are the Qt ones.
    """
    from PySide6.QtWidgets import QStackedWidget

    monkeypatch.setitem(PREVIEWS, "w2_2_probe",
                        PreviewSpec(builder=f"{_HERE}:_build_test_card"))
    first = _Screen("w2_2_probe", model=_Model())
    second = _Screen("w2_2_probe", model=_Model())

    window = QWidget()
    stack = QStackedWidget(window)
    stack.addWidget(first)
    stack.addWidget(second)
    window._stack = stack

    watcher = install_window_hooks(window)
    assert watcher is not None
    assert window._preview_watcher is watcher
    # asked twice, the same watcher comes back rather than a second one
    assert install_window_hooks(window) is watcher

    # the deferred first install lands on the current screen
    qtbot.waitUntil(lambda: getattr(first, "_registry_preview", None)
                    is not None, timeout=5000)

    stack.setCurrentIndex(1)
    assert getattr(second, "_registry_preview", None) is not None
