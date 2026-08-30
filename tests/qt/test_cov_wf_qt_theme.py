"""The theme module's solvers and styling helpers at the ends of their ranges.

Every path here is one the running application takes when the colours it is
handed do not work out: a scrim no alpha can make readable, an ink caught
between the two surfaces it has to be read against, a drift offset no amount
of damping can rescue, a widget whose Qt style has already gone, and the
import-time batch that a screen's QSS block is registered into. None of them
may raise and each has a defined answer, because they run inside a
``paintEvent``, a widget constructor, or a module import where there is
nobody left to catch anything.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                      # noqa: E402
from PySide6.QtWidgets import QWidget              # noqa: E402

from spacr.qt import theme                         # noqa: E402

pytestmark = pytest.mark.qt


# --- widgets whose Qt half has gone ---------------------------------------

class _RecordingStyle:
    """A style that writes down the polish calls it is asked for."""

    def __init__(self):
        self.calls = []

    def unpolish(self, widget):
        self.calls.append(("unpolish", widget))

    def polish(self, widget):
        self.calls.append(("polish", widget))


class _StyledWidget(QWidget):
    """A widget whose style is a recorder, and which counts its repaints."""

    def __init__(self):
        super().__init__()
        self.recorder = _RecordingStyle()
        self.repaints = 0

    def style(self):
        return self.recorder

    def update(self):
        self.repaints += 1


class _StylelessWidget(QWidget):
    """A widget caught after the style that would polish it has gone."""

    def __init__(self):
        super().__init__()
        self.repaints = 0

    def style(self):
        return None

    def update(self):
        self.repaints += 1


@pytest.fixture()
def spare_block_names():
    """Three registry names, released again however the test ends."""
    names = ("ACovWfThemeBlockOne", "ACovWfThemeBlockTwo",
             "ACovWfThemeBlockSolo")
    for name in names:
        theme.unregister_widget_qss(name)
    yield names
    for name in names:
        theme.unregister_widget_qss(name)
    theme._QSS_BATCH_PENDING.clear()


# --- scrim bounds ---------------------------------------------------------

def test_a_panel_that_can_never_be_read_still_answers_with_a_pair():
    """``_scrim_bounds`` is called once per role per drift offset while the
    spaceout palette is being solved, and the solver divides and clamps with
    whatever it returns. A palette in which the ink is the panel colour --
    which a user theme file or a half-applied dressing can produce -- must
    therefore come back as the widest possible admission of failure, an
    opaque floor and a zero ceiling, and not as a hang or an exception that
    would take the whole palette down with it.
    """
    readable = dict(theme.palette_for("dark"))
    readable["page"] = "#ffffff"
    for role, _required in theme._scrim_rules("page"):
        readable[role] = "#000000"

    floor, ceiling = theme._scrim_bounds(readable, "page", "page",
                                         (255, 255, 255))

    # Black on white over white: legible from the first step, and the
    # picture survives until the panel is nearly opaque.
    assert floor == 0.0
    assert 0.0 < ceiling < 1.0

    hopeless = dict(readable)
    hopeless["page"] = "#000000"

    assert theme._scrim_bounds(hopeless, "page", "page", (0, 0, 0)) == \
        (1.0, 0.0)


# --- ink bands ------------------------------------------------------------

def test_an_ink_between_its_two_surfaces_is_given_no_band_to_spend(
        monkeypatch):
    """A band is the luminance an ink may move through and stay readable, and
    the spaceout hue solver spends it on colour. It is computed by deciding
    whether the ink is the lighter or the darker half of every pair it is
    read against; an ink that is lighter than one of its surfaces and darker
    than another has no direction to move in at all, and the honest answer is
    ``None`` -- "leave this role on the plain hue shift". Answering with a
    band anyway would let the solver push a role until it failed WCAG against
    the surface it did not consider.
    """
    real = theme._ink_band("dark", "fg_dim")

    assert real is not None
    low, high = real
    assert 0.0 < low < high <= 1.0

    # `fg_dim` sits above `bg` and below `fg_muted`, so read against both it
    # is neither the lighter nor the darker of the pair.
    monkeypatch.setattr(theme, "CONTRAST_RULES",
                        (("fg_dim", "bg", 3.0), ("fg_dim", "fg_muted", 3.0)))

    assert theme._ink_band("dark", "fg_dim") is None


def test_a_role_with_no_band_is_left_out_of_the_solved_table(monkeypatch):
    """The solved table is read with ``.get(role)`` at paint time, so a role
    whose band came back ``None`` has to be ABSENT rather than present with a
    ``None`` value: a stored ``None`` would be unpacked into ``low, high``
    the next time the hue solver reached for it, and the theme would fail to
    build at all.
    """
    def band(name, role):
        return None if role == "fg_dim" else (0.25, 0.75)

    monkeypatch.setattr(theme, "_ink_band", band)

    solved = theme._solve_ink_bands()

    assert set(solved) == set(theme.THEMES)
    for name in theme.THEMES:
        assert solved[name] == {"fg": (0.25, 0.75),
                                "fg_muted": (0.25, 0.75)}
        assert "fg_dim" not in solved[name]


# --- page damping ---------------------------------------------------------

def test_an_offset_no_damping_can_rescue_keeps_the_dimmest_step(monkeypatch):
    """Damping is how much colour a theme gives up so its panels stay visible
    against its page. The solver walks the steps from full colour downwards
    and stops at the first that works; if none of them works it must keep the
    dimmest step it tried rather than falling back to full colour, because
    full colour is exactly the setting that was measured as invisible. An
    offset that is rescued at step 1.0 is dropped instead -- storing it would
    make every palette lookup pay for a table entry that changes nothing.
    """
    monkeypatch.setattr(theme, "THEMES", ("dark",))
    monkeypatch.setattr(theme, "_drift_grid", lambda: (0.0, 12.0))
    monkeypatch.setattr(theme, "_PAGE_DAMPING", {})

    monkeypatch.setattr(theme, "page_separation_failures",
                        lambda name: ["surface is 1.01:1 against page"])
    hopeless = theme._solve_page_damping()

    dimmest = min(theme.SPACEOUT_DAMPING_STEPS)

    assert hopeless == {"dark": {0.0: dimmest, 12.0: dimmest}}

    monkeypatch.setattr(theme, "page_separation_failures", lambda name: [])
    contented = theme._solve_page_damping()

    assert contented == {"dark": {}}


# --- marking a surface ----------------------------------------------------

def test_a_widget_with_no_style_is_still_marked_and_the_sweep_goes_on(qtbot):
    """``mark_surface`` is variadic and screens hand it a whole column of
    panels at once. A widget whose style has gone -- the C++ half torn down
    while the screen was being rebuilt -- must still get its surface property
    and must not stop the widgets listed after it: the property is what the
    stylesheet selects on, so a swallowed sweep leaves the rest of the page
    unpainted, which is the bug this guard exists for.
    """
    styleless = _StylelessWidget()
    qtbot.addWidget(styleless)
    after = QWidget()
    qtbot.addWidget(after)

    assert theme.is_surface(after) is False

    theme.mark_surface(styleless, after)

    assert theme.is_surface(styleless) is True
    assert theme.is_surface(after) is True
    assert styleless.property(theme.TRANSPARENT_PROPERTY) is False
    assert after.testAttribute(Qt.WA_StyledBackground) is True


def test_repolish_repaints_a_widget_whose_style_has_gone(qtbot):
    """Every dynamic-property change in the application ends in ``repolish``:
    a button going busy, a field going invalid, a row going selected. Qt only
    re-reads the stylesheet after an unpolish/polish pair, and it only shows
    the result after an update, so the update has to happen even when there
    is no style left to polish with -- otherwise a widget caught mid-teardown
    keeps the old colours until something else happens to repaint it.
    """
    styled = _StyledWidget()
    qtbot.addWidget(styled)

    theme.repolish(styled)

    assert [call for call, _ in styled.recorder.calls] == ["unpolish",
                                                           "polish"]
    assert styled.repaints == 1

    styleless = _StylelessWidget()
    qtbot.addWidget(styleless)

    theme.repolish(styleless)

    assert styleless.repaints == 1


# --- batched QSS registration ---------------------------------------------

def test_a_run_of_blocks_in_one_batch_restyles_the_application_once(
        monkeypatch, spare_block_names):
    """Registering a QSS block normally rebuilds the live stylesheet at once,
    so a screen imported long after startup is styled before its first paint.
    A cold launch imports four such screens in a row and paid for four full
    rebuilds, three of them thrown away. Inside a batch the names are
    collected and the rebuild is owed once, on the way out -- and a name
    registered twice inside the batch must be queued once, or the flush pays
    per registration again and the saving is gone.
    """
    first, second, solo = spare_block_names
    applied = []

    def record(*names):
        applied.append(names)
        return True

    monkeypatch.setattr(theme, "ensure_widget_qss_applied", record)

    theme.register_widget_qss(solo, lambda palette, opacity: "QWidget#S {}")

    assert applied == [(solo,)]

    with theme.batched_widget_qss():
        theme.register_widget_qss(first,
                                  lambda palette, opacity: "QWidget#A {}")
        theme.register_widget_qss(second,
                                  lambda palette, opacity: "QWidget#B {}")
        theme.register_widget_qss(first,
                                  lambda palette, opacity: "QWidget#C {}",
                                  replace=True)
        assert applied == [(solo,)]
        assert list(theme._QSS_BATCH_PENDING) == [first, second]

    assert applied == [(solo,), (first, second)]
    assert list(theme._QSS_BATCH_PENDING) == []

    sheet = theme.stylesheet(theme="dark")

    assert "QWidget#C {}" in sheet
    assert "QWidget#A {}" not in sheet
    assert "QWidget#B {}" in sheet
