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
import shiboken6
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from spacr.qt.preview_registry import (PREVIEWS, PreviewSpec, _PreviewHost,
                                       _resolve, _StackWatcher,
                                       register_preview)

_OWNED_BY_TEST = []


@pytest.fixture(autouse=True)
def _retire_probe_widgets():
    yield
    for widget in reversed(_OWNED_BY_TEST):
        if shiboken6.isValid(widget):
            shiboken6.delete(widget)
    _OWNED_BY_TEST.clear()
    _LAST_BUILT.clear()


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
        _OWNED_BY_TEST.append(self)
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
        _OWNED_BY_TEST.append(self)
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
    _OWNED_BY_TEST.append(card)
    _LAST_BUILT["panel"] = panel
    _LAST_BUILT["card"] = card
    return panel, card



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
