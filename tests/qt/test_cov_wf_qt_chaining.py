"""The chaining strip on a screen that cannot take a settings dict.

Every path the strip has for *writing* into a module screen goes through the
same seam — ``apply_settings_dict`` — and every one of them asks whether the
seam is there before using it.  ``tests/qt/test_cov_w3_7_chaining.py`` covers
the seam that is present and throws; what is covered here is the seam that is
simply **absent**, which is the real state of a stand-in host, of a screen
built by a plugin, and of any screen written before that method existed.

The promise being held to in each case is the same one: the strip is an aid.
A screen it cannot write into must still get everything the strip can give it
without writing — the pin dropped, the offer remembered, the successor opened
— and must never get a crash instead.  The last test covers the other end of
the same widget: the successor row clearing itself of a layout that holds
more than the buttons it put there.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLineEdit, QPushButton, QWidget

from spacr import chaining as core_chaining
from spacr.chaining import ChainedInput, HeldPin, NextStep, Resolution
from spacr.qt.chaining import ChainingBar


@pytest.fixture(autouse=True)
def _own_pins(monkeypatch, tmp_path):
    """Never read or write the developer's real pin file."""
    monkeypatch.setenv(core_chaining.PIN_STATE_ENV, str(tmp_path / "pins.json"))
    core_chaining.pin_store(refresh=True)
    yield
    core_chaining.pin_store(refresh=True)


@pytest.fixture
def pins(tmp_path):
    return core_chaining.PinStore(str(tmp_path / "pins.json"))


class _Model:
    """The one attribute the strip reads off a settings model here."""

    def __init__(self, widgets=None):
        self._widgets = dict(widgets or {})


class _Host(QWidget):
    """A module screen offering only the seams a test asks for.

    ``apply_settings_dict`` is set only when ``apply`` is given, which is the
    whole point: these tests are about the host that does not have it.
    """

    def __init__(self, app_key="measure", *, model=None, apply=None):
        super().__init__()
        self.app_key = app_key
        if model is not None:
            self._settings_model = model
        if apply is not None:
            self.apply_settings_dict = apply


def _bar(qtbot, pins, *, model=None, apply=None):
    host = _Host(model=model, apply=apply)
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    return strip


def _moved(value="/old/plate", offered="/new/plate", setting="src"):
    """A pin whose upstream has since written somewhere else.

    ``HeldPin.differs`` is only true when there is a resolved upstream to
    differ from, so the artifact behind the offer is built rather than
    implied.
    """
    from spacr.artifacts import Artifact

    artifact = Artifact(
        artifact_id="0" * 16, project="/plate", kind="merged-arrays",
        role="merged", path=offered, module="mask", run_id="run",
        settings_hash="h", spacr_version="0", created_ns=0,
        created_utc="1970-01-01T00:00:00Z", fingerprint="f",
        fingerprint_method="none", size_bytes=0, n_files=1,
        status="complete")
    chained = ChainedInput(module="measure", setting=setting, role="merged",
                           kind="merged-arrays", value=offered,
                           artifact=artifact, producer="mask", root="/plate")
    return HeldPin(setting, value, offered=offered, chained=chained)


def _step(module="classify", seed=None):
    return NextStep(module=module, source="measure", root="/plate",
                    kinds=("crops",), seed=dict(seed or {"src": "/plate"}),
                    readiness=None, artifacts=())


# ---------------------------------------------------------------------------
# Handing a seed to a screen that cannot take one
# ---------------------------------------------------------------------------

def test_a_seed_a_screen_cannot_take_is_still_remembered_as_offered(qtbot,
                                                                    pins):
    """The Continue button hands a resolved artifact to the next module.

    If the successor has no way to accept a settings dict, the count it
    reports has to be an honest zero rather than a guess -- but the strip
    must still record the value as *offered*, because ``_capture_edits``
    reads that record to tell "the user typed this path" from "we put it
    there".  Lose it and the next refresh would see an unexplained value in
    the field, pin it as the user's own choice, and the path would then
    outlive the run it came from and never update again.
    """
    taken = {}
    accepting = _bar(qtbot, pins,
                     apply=lambda values: taken.update(values) or len(values))
    assert accepting.adopt({"src": "/plate/a", "nr": 4}) == 2
    assert taken == {"src": "/plate/a", "nr": 4}

    refusing = _bar(qtbot, pins)
    assert not hasattr(refusing._screen, "apply_settings_dict")
    assert refusing.adopt({"src": "/plate/a", "nr": 4}) == 0
    assert refusing._offered == {"src": "/plate/a", "nr": 4}, \
        "a seed nobody could apply must still not read back as a user's pin"


def test_the_offered_path_is_taken_even_by_a_screen_that_cannot_be_written(
        qtbot, pins):
    """"Use it" is how a user follows an upstream that has moved.

    Dropping the pin is the half that matters and the half that persists: if
    a screen without a settings seam kept its pin, the button would look like
    it did nothing and the stale path would be restored on every restart, for
    ever.  The pin has to go whether or not anybody could fill the field.
    """
    strip = _bar(qtbot, pins)
    pins.pin("measure", "src", "/old/plate")
    pins.pin("measure", "other", "/kept")
    strip._held = {"src": _moved(),
                   "other": HeldPin("other", "/kept", offered=None)}

    strip._on_use_offered()

    assert pins.pinned("measure", "src") is None
    assert strip._offered == {"src": "/new/plate"}
    assert pins.pinned("measure", "other") == "/kept", \
        "a pin whose upstream has not moved is not what the button is about"


def test_a_successor_with_no_way_in_is_still_opened(qtbot, pins, monkeypatch):
    """Continue is a navigation first and a seed second.

    A successor screen that offers neither its own chaining strip nor a
    settings seam -- a plugin page, or one built before either existed --
    still has to be the page the user lands on when they press the button.
    Letting the missing seam abort the continuation would leave the user
    staring at the module they had just finished, with nothing said.
    """
    seeded_directly = {}
    reached = []

    class Openable:
        _chaining_bar = None

        def apply_settings_dict(self, values):
            seeded_directly.update(values)
            return len(values)

    class Closed:
        def __getattr__(self, name):
            reached.append(name)
            raise AttributeError(name)

    class Window:
        def __init__(self, target):
            self._screens = {"classify": target}
            self.opened = None

        def _on_nav_selected(self, module):
            self.opened = module

    strip = _bar(qtbot, pins)
    open_window = Window(Openable())
    monkeypatch.setattr(type(strip), "host_window", lambda self: open_window)
    strip._on_continue(_step(seed={"src": "/plate/x"}))
    assert open_window.opened == "classify"
    assert seeded_directly == {"src": "/plate/x"}

    closed_window = Window(Closed())
    monkeypatch.setattr(type(strip), "host_window", lambda self: closed_window)
    strip._on_continue(_step(seed={"src": "/plate/y"}))
    assert closed_window.opened == "classify"
    assert reached == ["_chaining_bar", "apply_settings_dict"], \
        "the strip asked for both seams and gave up rather than crashing"


# ---------------------------------------------------------------------------
# The refresh that has a value but nowhere to put it
# ---------------------------------------------------------------------------

def _resolution(value="/plate/from-the-registry"):
    return Resolution(module="measure", settings={"src": value},
                      filled={}, held={}, inputs=())


def test_a_chained_path_is_recorded_even_when_no_field_can_be_filled(
        qtbot, pins, monkeypatch):
    """The registry answered; the screen has no way to be written to.

    The strip's bookkeeping (``_offered`` / ``_seen``) is what makes a pin
    survive a restart, and it is updated from the resolution rather than from
    whatever the write did -- so a screen with no settings seam still learns
    which paths were chained for it.  If that update were skipped along with
    the write, the very next ``_capture_edits`` on a screen that later grew
    the seam would read the chained path as the user's own and pin it.
    """
    monkeypatch.setattr(core_chaining, "resolve_settings",
                        lambda *a, **k: _resolution())

    field = QLineEdit("")
    qtbot.addWidget(field)
    filled = _bar(qtbot, pins, model=_Model({"src": field}),
                  apply=lambda values: field.setText(values["src"]))
    assert filled._bound_settings() == ("src",)
    filled.refresh()
    assert field.text() == "/plate/from-the-registry"

    blank = QLineEdit("")
    qtbot.addWidget(blank)
    unwritable = _bar(qtbot, pins, model=_Model({"src": blank}))
    unwritable.refresh()
    assert blank.text() == "", "there was no seam, so nothing was written"
    assert unwritable._offered == {"src": "/plate/from-the-registry"}
    assert unwritable._seen == {"src": "/plate/from-the-registry"}


def test_the_successor_row_clears_whatever_it_finds_between_its_ends(qtbot,
                                                                     pins):
    """Row 4 is rebuilt from scratch on every refresh.

    It empties itself by taking items from position 1 until only its label
    and its trailing stretch are left, and what it takes need not be one of
    the buttons it made -- a theme or a layout change can leave a spacer in
    there.  ``QLayoutItem.widget()`` is None for one of those, and treating
    it as a widget would raise inside the redraw, so the strip would go dark
    for the rest of the session on a screen that had merely been restyled.
    """
    strip = _bar(qtbot, pins)
    stale_button = QPushButton("Continue to Classify")
    strip._next_layout.insertWidget(1, stale_button)
    strip._next_layout.insertStretch(1)
    strip._next_row.show()
    strip._last_steps = (_step(),)
    assert strip._next_layout.count() == 4

    strip._draw_next({"src": "/plate"}, finished=False)

    assert strip._next_layout.count() == 2, \
        "the spacer and the stale button both went; the ends stayed"
    assert stale_button.parent() is None
    assert strip._next_row.isHidden()
    assert strip._last_steps == ()
