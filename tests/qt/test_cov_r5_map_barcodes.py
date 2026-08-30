"""The fold seams in :mod:`spacr.qt.screens.map_barcodes`, at their edges.

This module is the shared fold infrastructure every host screen uses: it
restates a folded module's button, hangs the module on its host as a page,
and -- for the modules that are not a second screen at all -- mounts their
settings categories on the host's own form.

What is pinned here is the behaviour at the edges of those seams, which is
where a fold stops being decoration and starts being a module the user
cannot reach:

* a fold that records a sentence but no name still gets its sentence and
  its maturity colour onto the button;
* a button already wearing the right stage is left alone rather than
  re-polished, because a repolish on a switch is a visible flicker;
* a page the registry ships no art for still opens, and a close request
  for a page that is not there leaves the strip exactly as it was;
* reopening a fold that is already on screen raises it without showing it
  a second time;
* a settings category whose every control the host has meanwhile acquired
  is not mounted, because two controls for one key give ``collect()`` two
  answers;
* and a dependency naming a fold this host does not carry is a no-op
  rather than a crash.
"""
from __future__ import annotations

import importlib.util
import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QIcon                                  # noqa: E402
from PySide6.QtWidgets import (                                  # noqa: E402
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spacr.qt.screens import map_barcodes as mb                  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# The button a folded module leaves behind
# ---------------------------------------------------------------------------

def test_a_fold_that_records_no_name_still_says_what_it_does(qtbot,
                                                             monkeypatch):
    """A record with a sentence and no name puts the sentence on the button.

    ``fold_description`` merges the registry row with
    :data:`~spacr.qt.screens.map_barcodes.FOLD_FALLBACK`, and the two are
    merged FIELD BY FIELD -- a module whose row is gone keeps whichever
    fields the fallback still records. So "no name" is not "no record":
    the tooltip and the maturity colour are both still owed, and skipping
    them would leave the switch unlabelled and painted stable-blue.
    """
    monkeypatch.setitem(mb.FOLD_FALLBACK, "zz_nameless",
                        ("", "Ranked hits with effect size and FDR", "alpha"))
    nameless = QPushButton()
    qtbot.addWidget(nameless)

    mb.restate_fold_button(nameless, "zz_nameless")

    # The sentence and the stage arrive; the accessible name has nothing to
    # be set from and is left as Qt made it.
    assert nameless.toolTip() == "Ranked hits with effect size and FDR"
    assert nameless.property("stage") == "alpha"
    assert nameless.accessibleName() == ""

    # And the empty name above is a property of that record, not of the
    # function: a record that HAS a name puts it on the button.
    named = QPushButton()
    qtbot.addWidget(named)
    mb.restate_fold_button(named, "hit_list")
    assert named.accessibleName() == mb.FOLD_FALLBACK["hit_list"][0]
    assert named.toolTip().startswith(mb.FOLD_FALLBACK["hit_list"][0])


class _StyleRecorder:
    """A stand-in for ``QStyle`` that records the repolish it was asked for."""

    def __init__(self) -> None:
        self.calls: list = []

    def unpolish(self, widget) -> None:
        self.calls.append(("unpolish", widget))

    def polish(self, widget) -> None:
        self.calls.append(("polish", widget))


def test_a_button_already_wearing_the_stage_is_not_repolished(qtbot,
                                                              monkeypatch):
    """Restating a stage the button already has costs no style pass.

    A repolish re-runs the whole QSS cascade on the widget and is visible
    on a switch -- the fill blinks. ``restate_fold_button`` is called for
    every button on every strip build, so a repolish that changes nothing
    would be a flicker on every host screen that carries folds.

    ``style()`` is shadowed on the instance (the pattern
    ``test_profiler_uncovered_paths`` uses for the same reason): the real
    ``QStyle`` is shared by the whole application, so there is nothing on
    it to count.
    """
    monkeypatch.setitem(mb.FOLD_FALLBACK, "zz_staged",
                        ("Staged", "A folded module", "alpha"))

    already = QPushButton()
    qtbot.addWidget(already)
    already.setProperty("stage", "alpha")
    recorder = _StyleRecorder()
    already.style = lambda: recorder
    try:
        mb.restate_fold_button(already, "zz_staged")
    finally:
        del already.style

    assert recorder.calls == [], "the button was repolished for no change"
    assert already.property("stage") == "alpha"
    assert already.toolTip() == "Staged\nA folded module"

    # The same call on a button wearing the WRONG stage does re-polish it,
    # which is what makes the silence above a decision and not a dead path.
    wrong = QPushButton()
    qtbot.addWidget(wrong)
    wrong.setProperty("stage", "beta")
    moved = _StyleRecorder()
    wrong.style = lambda: moved
    try:
        mb.restate_fold_button(wrong, "zz_staged")
    finally:
        del wrong.style

    assert [name for name, _widget in moved.calls] == ["unpolish", "polish"]
    assert wrong.property("stage") == "alpha"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@pytest.fixture()
def host(qtbot):
    """A host screen whose body can be turned into a page strip."""
    made = QWidget()
    qtbot.addWidget(made)
    layout = QVBoxLayout(made)
    body = QLabel("the host's own page")
    layout.addWidget(body, 1)
    made._fold_page_title = "Host"
    made.resize(400, 300)
    return made


def test_a_page_the_registry_ships_no_art_for_still_opens(host, qtbot,
                                                          monkeypatch):
    """A null icon is a page without a mark, not a page that fails to open.

    ``iconset.app_icon`` falls back twice -- bundled PNG, then a qtawesome
    glyph, then bundled fallback art -- and the last of those can still
    come back null on an installation missing the art. The mark is
    decoration; the page is the point.
    """
    from spacr.qt import iconset

    # The ordinary half first, while the real art is still in place.
    marked = QWidget()
    qtbot.addWidget(marked)
    marked.app_key = "agreement"
    assert mb.show_as_page(marked, host, "Agreement") is marked
    pages = host._fold_pages
    assert not pages.tabIcon(pages.indexOf(marked)).isNull()

    monkeypatch.setattr(iconset, "app_icon", lambda *a, **k: QIcon())
    unmarked = QWidget()
    qtbot.addWidget(unmarked)
    unmarked.app_key = "agreement"

    assert mb.show_as_page(unmarked, host, "Agreement again") is unmarked

    index = pages.indexOf(unmarked)
    assert index >= 0
    assert pages.tabIcon(index).isNull()
    assert pages.tabText(index) == "Agreement again"
    assert pages.currentIndex() == index, "the page was not selected"


def test_closing_a_page_that_is_not_there_leaves_the_strip_alone(qtbot):
    """``tabCloseRequested`` can name an index the strip no longer holds.

    The signal carries an index, and an index is only good for as long as
    nobody else edited the strip. Removing "the widget at index 4" when
    there is no widget at index 4 must not disturb the pages that ARE
    there.
    """
    pages = QTabWidget()
    qtbot.addWidget(pages)
    body = QLabel("host")
    folded = QLabel("folded")
    pages.addTab(body, "Host")
    pages.addTab(folded, "Folded")

    mb._close_fold_page(pages, 7)

    assert pages.count() == 2
    assert pages.widget(1) is folded
    assert folded.parent() is not None

    # The same call on an index that IS there takes the page off and hands
    # the screen back to its caller, still alive.
    mb._close_fold_page(pages, 1)

    assert pages.count() == 1
    assert pages.indexOf(folded) < 0
    assert folded.parent() is None
    assert folded.text() == "folded"


def test_reopening_a_fold_that_is_up_raises_it_without_showing_it_again(
        host, qtbot):
    """The second press selects the page; it does not re-show the screen.

    ``FoldOpener`` keeps ONE screen per fold so the module does not open a
    second database handle or job runner, and the page it is already on
    keeps whatever it had loaded. The visible half of that promise is that
    pressing the button again is a selection, not a rebuild.
    """
    from PySide6.QtWidgets import QApplication

    host.show()
    QApplication.processEvents()

    built: list = []

    def build(_window):
        page = QLabel("the folded module")
        qtbot.addWidget(page)
        built.append(page)
        return page

    opener = mb.FoldOpener(host, "hit_list", build)

    first = opener.open()
    QApplication.processEvents()
    assert first is built[0]
    assert first.isVisible()

    second = opener.open()

    assert second is first, "a second press built a second screen"
    assert len(built) == 1
    assert first.isVisible()
    pages = host._fold_pages
    # One host page plus one folded page, and the folded one is selected.
    assert pages.count() == 2
    assert pages.currentIndex() == pages.indexOf(first)


# ---------------------------------------------------------------------------
# The import-time QSS registration
# ---------------------------------------------------------------------------

def _load_fresh_map_barcodes():
    """Execute the module's source again, as a separate module object.

    ``importlib.reload`` would replace the live module every other test in
    the session is holding classes from; this runs the same file under a
    fresh namespace and leaves ``sys.modules`` alone.
    """
    spec = importlib.util.find_spec("spacr.qt.screens.map_barcodes")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    return fresh


def test_a_theme_that_cannot_take_the_page_block_still_loads_the_module(
        monkeypatch, caplog):
    """The import-time registration is best-effort; the module is not.

    The block is registered twice on purpose -- once here, so a stylesheet
    composed before any fold page exists already carries the rule, and once
    from :func:`_ensure_pages_qss` for a page made afterwards. Neither is
    allowed to stop the module loading: every host screen in the
    application imports it for ``install_folds_on``, so a raise here would
    take the fold strips off every screen rather than leave one strip
    unstyled.
    """
    from spacr.qt import theme

    def refuse(*_args, **_kwargs):
        raise RuntimeError("the stylesheet is not accepting blocks")

    monkeypatch.setattr(theme, "register_widget_qss", refuse)
    caplog.set_level(logging.DEBUG, logger=mb.__name__)

    fresh = _load_fresh_map_barcodes()

    assert fresh.PAGES_NAME == mb.PAGES_NAME
    assert callable(fresh.host_pages)
    assert any("at import" in record.getMessage()
               for record in caplog.records), (
        "the refused registration was not recorded")

    # And the refusal is the theme's, not the module's: the same load with a
    # registrar that accepts does register the block under its own name.
    caplog.clear()
    accepted: list = []

    def accept(name, builder, replace=False):
        accepted.append((name, replace))

    monkeypatch.setattr(theme, "register_widget_qss", accept)
    again = _load_fresh_map_barcodes()

    assert (again.PAGES_NAME, True) in accepted
    assert not [record for record in caplog.records
                if "at import" in record.getMessage()]


# ---------------------------------------------------------------------------
# Categories mounted on the host's form
# ---------------------------------------------------------------------------

class _HostModel:
    """The two attributes ``CategoryFold.mount`` reads off a host model."""

    def __init__(self) -> None:
        self._widgets: dict = {}
        self._defaults: dict = {}


def test_a_category_the_host_already_holds_is_not_mounted_twice(qtbot,
                                                               monkeypatch):
    """A control the host acquired while the fold was building is skipped.

    ``mount`` asks the host which keys it holds TWICE: once before building
    the folded form, to skip those keys outright, and once after, at line
    810. The second read is not redundant --
    ``SettingsWidgets.build_sections`` hands the event loop a turn every
    25 ms so the interface does not freeze, and the host's own deferred
    panel work can land in one of those turns. A category whose every
    control the host has acquired by then must not be mounted: ``collect()``
    is keyed on the setting name, so a second control for one key gives the
    run two answers and the host's own value silently loses.

    The settings model is stubbed rather than built: the real one
    materialises about 1,500 widgets for a module, and none of them are
    what this is about. The stub adds a key to the host between the two
    reads, which is exactly the race the second read exists for.
    """
    from spacr.qt.screens import settings_model

    host_model = _HostModel()
    screen = QWidget()
    qtbot.addWidget(screen)
    screen._settings_model = host_model
    content = QWidget(screen)
    QVBoxLayout(content)
    content.layout().addStretch(1)
    screen._settings_content = content

    shared = QLineEdit()
    private = QLineEdit()
    skipped_with: list = []

    class _StubWidgets:
        """Enough of ``SettingsWidgets`` for one fold to be mounted."""

        def __init__(self, app_key, parent=None, *, skip_keys=(),
                     current=None):
            self.app_key = app_key
            self._parent = parent
            self._widgets = {"src": shared, "frame_gap": private}
            self._defaults = {"timelapse": True, "frame_gap": 3}
            skipped_with.append(frozenset(skip_keys))

        def build_sections(self):
            # The host's own panel finishing during one of the build's
            # breathing turns: it now holds `src` itself.
            host_model._widgets["src"] = QLineEdit()
            return [("Paths", [("Source folder", shared)]),
                    ("Timelapse", [("Frame gap", private)])]

    monkeypatch.setattr(settings_model, "SettingsWidgets", _StubWidgets)

    fold = mb.CategoryFold(screen, "timelapse", gates=("timelapse",))
    assert fold.mount() is True

    # Nothing was skipped up front -- the host held nothing when asked --
    # so "Paths" was dropped by the second read and not by the first.
    assert skipped_with == [frozenset()]
    mounted_titles = [section.property("settingsCategorySource")
                      for section in fold.sections]
    assert mounted_titles == ["Timelapse"], (
        "the category the host had acquired was mounted anyway")
    assert fold.settings_keys == ("frame_gap",)

    # The fold's own control reached the host's collection; the duplicate
    # did not displace the host's.
    assert host_model._widgets["frame_gap"] is private
    assert host_model._widgets["src"] is not shared
    # The gate is the switch's to decide, so it does not ride along as a
    # default; everything else the module's own screen would have does.
    assert host_model._defaults == {"frame_gap": 3}


# ---------------------------------------------------------------------------
# Dependencies between folds
# ---------------------------------------------------------------------------

def test_turning_a_prerequisite_off_leaves_a_dependent_that_was_never_on(
        qtbot):
    """Only the dependents that are ON are switched off with it.

    Disabling a prerequisite walks every declared dependency, and a
    dependent that is already off has nothing to be turned off -- switching
    it anyway would fire its button and write its gate again for no change.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    folds = mb.CategoryFoldSet(
        screen,
        folds={"timelapse": ("timelapse",), "motility": ("motility",)},
        implies={"motility": ("timelapse",)})

    folds.set_active("timelapse", False)

    assert not folds.is_active("timelapse")
    assert not folds.is_active("motility")
    assert folds.apply_gates() == {"timelapse": False, "motility": False}

    # Drive the other half: with the dependent ON, the same call takes it
    # off, so the loop above really did decide rather than never run.
    folds.set_active("motility", True)
    assert folds.is_active("motility") and folds.is_active("timelapse")

    folds.set_active("timelapse", False)

    assert not folds.is_active("motility"), (
        "an active dependent survived its prerequisite")
    assert folds.apply_gates() == {"timelapse": False, "motility": False}


def test_a_dependency_this_host_does_not_carry_is_a_no_op(qtbot):
    """A prerequisite whose categories mounted nothing is simply not there.

    ``CategoryFoldSet.mount`` drops any fold that folds nothing new into
    the host, so a dependency table written for the general case can name a
    key that is no longer in ``folds``. That must leave the switch that
    named it working rather than raise out of a button press.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    folds = mb.CategoryFoldSet(
        screen,
        folds={"motility": ("motility",)},
        implies={"motility": ("timelapse",)})

    folds.set_active("motility", True)

    assert folds.is_active("motility")
    assert "timelapse" not in folds.folds
    assert folds.apply_gates() == {"motility": True}

    # And a prerequisite that IS carried is switched on by the same press,
    # so the absence above is the missing fold and not a dead branch.
    carried = mb.CategoryFoldSet(
        screen,
        folds={"motility": ("motility",), "timelapse": ("timelapse",)},
        implies={"motility": ("timelapse",)})

    carried.set_active("motility", True)

    assert carried.is_active("timelapse")
    assert carried.apply_gates() == {"motility": True, "timelapse": True}
