"""Attaching a declared Live Preview to a screen, and every quiet way it fails.

The registry's promise is that "a missing preview must not cost anyone a
module": an undeclared or screen-owned key, a builder that will not resolve
or raises, a screen with no runtime panel (or one not laid out yet) all
leave the screen untouched and ``install`` returns ``None``.  An installed
preview lands hidden above the Run row, its toggle goes to the settings
strip when there is one, priming reads the form once on first show, and the
host translates propagated setting names before writing them -- dropping
the ones it cannot map, and carrying on past a form that refuses one.
``install_window_hooks`` follows a window's screen stack exactly once.

The screens are real ``QWidget``s carrying the two anchors ``AppScreen``
exposes (``_runtime_wrap`` and ``_actions_row``), built one attribute short
at a time, so a negative result means the anchor really is not there.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtWidgets import (QLabel, QStackedWidget, QVBoxLayout,  # noqa: E402
                               QWidget)

from spacr.qt.preview_registry import (PREVIEWS, PreviewSpec,  # noqa: E402
                                       _PreviewHost, install,
                                       install_window_hooks,
                                       preview_app_keys, register_preview,
                                       unregister_preview)

pytestmark = pytest.mark.qt

_OWNED_BY_TEST = []
_LAST_BUILT = {}
_HERE = "tests.qt.test_preview_registry_attach"
_PROBE = "b12_probe"


@pytest.fixture(autouse=True)
def _retire_probe_widgets():
    yield
    for widget in reversed(_OWNED_BY_TEST):
        if shiboken6.isValid(widget):
            shiboken6.delete(widget)
    _OWNED_BY_TEST.clear()
    _LAST_BUILT.clear()


class _Model:
    """The settings model's two-method surface, and nothing else."""

    def __init__(self, values=None, refuse=()):
        self.values = dict(values or {})
        self.written = {}
        self.refuse = set(refuse)

    def collect(self):
        return dict(self.values)

    def set_value_for_key(self, key, value):
        if key in self.refuse:
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
    """A screen with the two anchors ``AppScreen`` keeps for this kind of reach."""

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


class _Bar:
    """A settings strip that collects trailing widgets."""

    def __init__(self):
        self.trailing = []

    def add_trailing_widget(self, widget):
        self.trailing.append(widget)


def _build_test_card(screen):
    """A builder with the shape every ``build_*_preview_card`` already has."""
    panel = _Panel()
    card = QWidget()
    _OWNED_BY_TEST.append(card)
    _LAST_BUILT["panel"] = panel
    _LAST_BUILT["card"] = card
    return panel, card


def _build_that_raises(screen):
    raise RuntimeError("this preview needs a GPU")


def _spec(builder="_build_test_card", **fields):
    return PreviewSpec(builder=f"{_HERE}:{builder}", **fields)


def _unlaid_screen():
    screen = _Screen(_PROBE, anchored=False, model=_Model())
    screen._runtime_wrap = QWidget(screen)     # no layout on it
    screen._actions_row = QLabel("Run", screen._runtime_wrap)
    return screen


def test_install_returns_none_and_leaves_the_screen_alone_when_it_cannot_attach(
        qapp, monkeypatch, subtests):
    """Every unattachable screen gets None, no host and no stray card."""
    for key in ("mask", "measure", "timelapse", "motility"):
        with subtests.test(msg=f"{key} is served by its own screen"):
            assert PREVIEWS[key].owned_by_screen is True
            screen = _Screen(key)
            assert install(screen) is None
            assert getattr(screen, "_registry_preview", None) is None

    # case, spec for the probe key (None: undeclared), screen factory,
    # whether a card was built and then disowned
    cases = [
        ("an undeclared module", None, lambda: _Screen("no_such_module"),
         False),
        ("a screen with no key", None, lambda: _Screen(""), False),
        ("a builder that will not resolve",
         PreviewSpec(builder="nowhere.at.all:build"),
         lambda: _Screen(_PROBE), False),
        ("a builder name with no function", PreviewSpec(builder=_HERE),
         lambda: _Screen(_PROBE), False),
        ("a builder that raises", _spec("_build_that_raises"),
         lambda: _Screen(_PROBE), False),
        ("no runtime panel: the card is disowned", _spec(),
         lambda: _Screen(_PROBE, anchored=False), True),
        ("a runtime panel with no layout", _spec(), _unlaid_screen, True),
    ]
    for case, spec, make_screen, built in cases:
        with subtests.test(msg=case):
            _LAST_BUILT.clear()
            with monkeypatch.context() as patch:
                if spec is not None:
                    patch.setitem(PREVIEWS, _PROBE, spec)
                screen = make_screen()
                assert install(screen) is None
            assert getattr(screen, "_registry_preview", None) is None
            assert ("card" in _LAST_BUILT) is built
            if built:
                assert _LAST_BUILT["card"].parent() is None


def test_install_puts_a_hidden_card_above_the_run_row_with_its_toggle(
        qapp, monkeypatch, subtests):
    """The card, toggle, priming and propagation of an installed preview are exact."""
    monkeypatch.setitem(PREVIEWS, _PROBE, _spec(
        title="Try it", tooltip="see what comes out",
        propagation={"cell_diameter": "diameter"}))
    model = _Model({"cell_diameter": 30, "cell_flow_threshold": 0.4})
    screen = _Screen(_PROBE, model=model)
    host = install(screen)
    panel = _LAST_BUILT["panel"]
    layout = screen._runtime_wrap.layout()

    with subtests.test(msg="placement and toggle"):
        assert host.card is _LAST_BUILT["card"]
        assert layout.indexOf(host.card) == 0
        assert layout.indexOf(screen._actions_row) == 2
        assert layout.indexOf(host.toggle) == 1
        assert host.toggle.parent() is screen._runtime_wrap
        assert host.card.isVisibleTo(screen) is False
        assert host.toggle.text() == "Try it"
        assert host.toggle.toolTip() == "see what comes out"
        assert host.toggle.isCheckable() is True
        assert host.toggle.objectName() == "SettingsPreviewToggle"
        assert screen._registry_preview is host
        assert install(screen) is host, "a second install built another"

    with subtests.test(msg="primed once, on first show"):
        assert panel.applied == [], "a preview nobody opened read the form"
        host.toggle.setChecked(True)
        assert host.card.isVisibleTo(screen) is True
        assert panel.applied == [{"cell_diameter": 30,
                                  "cell_flow_threshold": 0.4}]
        host.toggle.setChecked(False)
        assert host.card.isVisibleTo(screen) is False
        host.toggle.setChecked(True)
        assert len(panel.applied) == 1, "the form was read again on a re-show"

    with subtests.test(msg="the panel is wired to the host's translator"):
        assert panel.callback == host.on_propagate
        panel.callback({"cell_diameter": 19, "cell_channel": 2})
        assert model.written == {"diameter": 19}

    with subtests.test(msg="a settings strip takes the toggle"):
        bar = _Bar()
        strip_screen = _Screen(_PROBE, model=_Model(), bar=bar)
        monkeypatch.setitem(PREVIEWS, _PROBE, _spec())
        strip_host = install(strip_screen)
        assert bar.trailing == [strip_host.toggle]
        assert strip_screen._runtime_wrap.layout().indexOf(
            strip_host.toggle) == -1
        assert strip_host.toggle.text() == "Live preview"
        assert strip_host.toggle.toolTip() == (
            "Show a preview of what these settings produce.")


def test_preview_host_primes_and_propagates_only_what_the_form_can_take(
        qapp, subtests):
    """Each host gives the exact form writes and panel reads, never raising."""
    renamed = PREVIEWS["cellpose_masks"]
    plain = _spec()
    # case, spec, model, values, expected writes
    cases = [
        ("renames what it can, drops what it cannot", renamed, _Model(),
         {"cell_diameter": 42, "cell_flow_threshold": 0.7, "cell_channel": 1},
         {"diameter": 42, "flow_threshold": 0.7}),
        ("no rename map passes names straight through", plain, _Model(),
         {"diameter": 12}, {"diameter": 12}),
        ("a refused value does not stop the rest", plain,
         _Model(refuse={"diameter"}), {"diameter": 12, "flow_threshold": 0.4},
         {"flow_threshold": 0.4}),
        ("nothing to propagate", plain, _Model(), None, {}),
    ]
    for case, spec, model, values, written in cases:
        with subtests.test(msg=case):
            host = _PreviewHost(_Screen("x", model=model), spec, _Panel(),
                                QWidget())
            host.on_propagate(values)
            assert model.written == written

    with subtests.test(msg="a screen with no model is left with none"):
        screen = _Screen("x", model=None)
        host = _PreviewHost(screen, plain, _Panel(), QWidget())
        host.on_propagate({"diameter": 12})
        host.prime()
        assert screen._settings_model is None
        assert host.panel.applied == [] and host._primed is True

    with subtests.test(msg="a panel without apply_settings is not primed"):
        host = _PreviewHost(_Screen("x", model=_Model({"a": 1})), plain,
                            QWidget(), QWidget())
        host.prime()
        assert host._primed is True

    with subtests.test(msg="a panel that throws while priming is survived"):
        panel = _Panel(blow_up=True)
        host = _PreviewHost(_Screen("x", model=_Model({"a": 1})), plain,
                            panel, QWidget())
        host.prime()
        assert host._primed is True and panel.applied == []


class _Signalless:
    currentChanged = None


def test_install_window_hooks_follows_a_real_stack_once(qapp, qtbot,
                                                        monkeypatch,
                                                        subtests):
    """No stack or an unconnectable one is not followed; a real one, once."""
    for case, stack in [("no stack", None), ("no signal", _Signalless())]:
        with subtests.test(msg=case):
            window = QWidget()
            _OWNED_BY_TEST.append(window)
            if stack is not None:
                window._stack = stack
            assert install_window_hooks(window) is None
            assert getattr(window, "_preview_watcher", None) is None

    with subtests.test(msg="a real stack"):
        monkeypatch.setitem(PREVIEWS, _PROBE, _spec())
        first = _Screen(_PROBE, model=_Model())
        second = _Screen(_PROBE, model=_Model())
        window = QWidget()
        _OWNED_BY_TEST.append(window)
        stack = QStackedWidget(window)
        stack.addWidget(first)
        stack.addWidget(second)
        window._stack = stack

        watcher = install_window_hooks(window)
        assert watcher is not None and watcher.parent() is window
        assert window._preview_watcher is watcher
        assert install_window_hooks(window) is watcher
        assert getattr(first, "_registry_preview", None) is None, (
            "the first install was not deferred")
        qtbot.waitUntil(lambda: getattr(first, "_registry_preview", None)
                        is not None, timeout=5000)
        assert getattr(second, "_registry_preview", None) is None
        stack.setCurrentIndex(1)
        assert getattr(second, "_registry_preview", None) is not None


def test_register_and_unregister_preview_keep_one_declaration_per_key(
        monkeypatch, subtests):
    """A second declaration needs replace=True; unregistering reports what it removed."""
    first = _spec(title="One")
    second = _spec(title="Two")
    monkeypatch.setitem(PREVIEWS, _PROBE, first)

    with subtests.test(msg="a second declaration is refused by name"):
        with pytest.raises(ValueError) as raised:
            register_preview(_PROBE, second)
        assert str(raised.value) == (
            f"a preview for {_PROBE!r} is already registered; pass "
            "replace=True to replace it")
        assert PREVIEWS[_PROBE] is first

    with subtests.test(msg="replace=True replaces"):
        assert register_preview(_PROBE, second, replace=True) is second
        assert PREVIEWS[_PROBE] is second
        assert _PROBE in preview_app_keys()

    with subtests.test(msg="unregister reports whether it removed one"):
        assert unregister_preview(_PROBE) is True
        assert unregister_preview(_PROBE) is False
        assert _PROBE not in preview_app_keys()
        assert register_preview(_PROBE, first) is first
