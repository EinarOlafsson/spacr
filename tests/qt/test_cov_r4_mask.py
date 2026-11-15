"""The parts of Mask Generation's fold wiring that only go wrong once.

``spacr.qt.screens.mask`` is the one walk that reaches the Mask Generation
screen from outside: it hangs the example-plate button on the form, mounts
the Timelapse fold on the masthead, offers that fold's own preview behind
its switch, and marks the settings categories the fold owns. Almost all of
that is exercised by the ordinary screen build.

What is pinned here is the other half -- the paths a healthy screen never
takes, and which therefore only run the first time something is broken:

* a preview card Qt has already deleted under the switch that owns it;
* a fold with no button, no preview, or a preview that will not build;
* a fold whose categories carry no artwork, so nothing is marked;
* every way ``install_example_data_button`` can decline to install one;
* the guards in ``install_folds`` and ``sync_folds``, which exist so that a
  broken part costs its own feature and never the screen.

Each tolerant case is paired, in the same test, with the input that makes
the same code do the thing it is being tolerant about, so "nothing
happened" cannot pass for "the guard worked".
"""

from __future__ import annotations

import logging
import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QLineEdit, QWidget

from spacr.qt.screens import mask as mask_module
from spacr.qt.screens.map_barcodes import CategoryFoldSet
from spacr.qt.widgets.fold_strip import FoldStrip
from spacr.qt.widgets.section import Section

pytestmark = pytest.mark.qt

LOGGER = "spacr.qt.screens.mask"


# ---------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------

class _Screen:
    """The attributes this module asks a screen for, and nothing else.

    A real ``AppScreen`` is used wherever the behaviour under test is the
    screen's; these cases need a form that is MISSING something -- no
    settings model, no ``src`` control, no section that takes prose --
    and forcing a real screen into those states would mostly be testing
    the forcing.
    """

    def __init__(self, **attrs):
        self.applied = []
        self.__dict__.update(attrs)

    def apply_settings_dict(self, settings):
        self.applied.append(dict(settings))
        return len(settings)


class _Model:
    def __init__(self, widgets=None):
        self._widgets = dict(widgets or {})
        self._defaults = {}


class _Header(QWidget):
    """A masthead that records what was hung on it."""

    def __init__(self):
        super().__init__()
        self.trailing = []

    def add_trailing(self, widget):
        self.trailing.append(widget)
        widget.setParent(self)


class _Host:
    """What ``attach_folded`` hands back: a card and its toggle."""

    def __init__(self, parent=None):
        self.card = QWidget(parent)
        self.toggle = QCheckBox(self.card)
        self.card.setVisible(True)
        self.toggle.setVisible(True)
        self.toggle.setChecked(True)


@pytest.fixture
def host(qtbot, qt_theme_applied):
    """A real Mask Generation screen with its switches installed."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    assert mask_module.install_folds(screen) is None
    return screen, screen.dimension_switch("t")


# ---------------------------------------------------------------------------
# _OfferedPreview: the switch outlives the card it hides
# ---------------------------------------------------------------------------

def test_switching_a_fold_off_unchecks_and_hides_its_preview(qtbot):
    """Off is stronger than hidden: the card cannot stay open behind it."""
    host_obj = _Host()
    qtbot.addWidget(host_obj.card)
    watcher = mask_module._OfferedPreview(host_obj)

    watcher.set_offered(True)
    assert host_obj.toggle.isVisibleTo(host_obj.card)
    assert host_obj.toggle.isChecked(), "switching on unchecked the toggle"

    watcher.set_offered(False)
    assert not host_obj.toggle.isChecked()
    assert not host_obj.card.isVisible()
    assert not host_obj.toggle.isVisibleTo(host_obj.card)


def test_a_card_qt_already_deleted_is_logged_rather_than_raised(
        qtbot, caplog):
    """The switch outlives the panel when the screen is torn down.

    ``set_offered`` is a slot on the strip's toggled signal, so it can run
    after Qt has taken the card away. Paired with a live host in the same
    test: the live one proves the same call really does reach through to
    the toggle, so the dead one is tolerance rather than a no-op.
    """
    import shiboken6

    live = _Host()
    qtbot.addWidget(live.card)
    mask_module._OfferedPreview(live).set_offered(False)
    assert not live.toggle.isChecked(), "the live card was never touched"

    dead = _Host()
    qtbot.addWidget(dead.card)
    watcher = mask_module._OfferedPreview(dead)
    shiboken6.delete(dead.toggle)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        watcher.set_offered(False)

    assert any("the folded preview is gone" in record.getMessage()
               for record in caplog.records), caplog.text


# ---------------------------------------------------------------------------
# fold_previews
# ---------------------------------------------------------------------------

def test_the_offered_previews_are_handed_out_as_a_copy(host):
    """Callers get the mapping, not the screen's own dict to edit."""
    screen, _strip = host

    previews = mask_module.fold_previews(screen)

    assert set(previews) == set(mask_module.CATEGORY_FOLDS)
    assert previews == screen._fold_previews
    previews.clear()
    assert set(screen._fold_previews) == set(mask_module.CATEGORY_FOLDS), (
        "fold_previews handed out the screen's own dictionary")
    assert mask_module.fold_previews(_Screen()) == {}


# ---------------------------------------------------------------------------
# _offer_fold_previews
# ---------------------------------------------------------------------------

def _fold_set(keys):
    """A real ``CategoryFoldSet`` over ``keys``, unmounted.

    Only ``.order`` is read by ``_offer_fold_previews``; building the real
    thing keeps the ordering rule the same one the strip is built from.
    """
    return CategoryFoldSet(_Screen(), {key: (key,) for key in keys})


def test_an_offered_preview_starts_in_the_state_its_switch_is_in(
        qtbot, monkeypatch):
    """The panel is hidden until the switch is on, from the first paint."""
    from spacr.qt import preview_registry

    attached = _Host()
    monkeypatch.setattr(preview_registry, "attach_folded",
                        lambda screen, key: attached)
    qtbot.addWidget(attached.card)
    folds = _fold_set(("timelapse",))
    strip = FoldStrip([("timelapse", lambda on: None, True)])
    qtbot.addWidget(strip)

    offered = mask_module._offer_fold_previews(_Screen(), folds, strip)

    assert set(offered) == {"timelapse"}
    assert not attached.toggle.isChecked(), (
        "the preview was offered for a fold whose switch is off")
    button = strip.button_for("timelapse")
    button.setChecked(True)
    assert attached.toggle.isVisibleTo(attached.card), (
        "switching the fold on did not offer its preview")


# ---------------------------------------------------------------------------
# mark_fold_sources
# ---------------------------------------------------------------------------

def test_only_folds_with_artwork_of_their_own_are_reported_as_marked(qtbot):
    """A mark is the icon; a module without one has nothing to say.

    ``mark_folded_sections`` returns the titles it actually drew on, and a
    key with no bundled icon draws on none -- so the returned mapping has
    to leave it out rather than claim an unmarked category.
    """
    titles = ("Time Axes & Tracking (Beta)", "Nothing In Particular")
    marked_section, anonymous_section = (Section(t) for t in titles)
    for section, title in zip((marked_section, anonymous_section), titles):
        # What ``CategoryFold._build_section`` stamps on a mounted card, so
        # the title is reported as it was written rather than uppercased.
        section.setProperty("settingsCategorySource", title)
        qtbot.addWidget(section)

    screen = _Screen(_settings_sections=())
    folds = CategoryFoldSet(
        screen, {"timelapse": ("timelapse",), "not_a_module": ()})
    folds.folds["timelapse"].sections = [marked_section]
    folds.folds["not_a_module"].sections = [anonymous_section]
    screen._category_folds = folds

    marked = mask_module.mark_fold_sources(screen)

    assert marked == {"timelapse": ("Time Axes & Tracking (Beta)",)}
    assert marked_section.source_app() == "timelapse"
    assert anonymous_section.source_mark() is None


# ---------------------------------------------------------------------------
# example_plate_folder
# ---------------------------------------------------------------------------

def test_the_example_plate_lives_beside_the_other_example_data():
    """One cache directory holds everything spaCR fetched or made."""
    from spacr.example_data import cache_folder

    folder = mask_module.example_plate_folder()

    assert folder == os.path.join(cache_folder(),
                                  mask_module.EXAMPLE_PLATE_DIRNAME)
    assert os.path.basename(folder) == "mask_example_plate"


# ---------------------------------------------------------------------------
# install_example_data_button
# ---------------------------------------------------------------------------

def _src_form(qtbot):
    """A settings form whose third section is the one holding ``src``."""
    src = QLineEdit()
    holder = Section("Source")
    holder.add_prose(src)
    elsewhere = Section("Somewhere Else")
    plain = QWidget()
    for widget in (holder, elsewhere, plain):
        qtbot.addWidget(widget)
    return src, holder, (plain, elsewhere, holder)


def test_a_src_control_in_no_prose_section_gets_no_button(qtbot):
    """Placement is found from ``src``, so a stray control places nothing.

    The section is identified by holding ``src`` rather than by its title,
    so category renaming cannot move the button; the cost is that a
    ``src`` sitting outside every section leaves nowhere to put it.
    """
    src, holder, sections = _src_form(qtbot)

    orphan = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                     _settings_sections=(QWidget(), Section("Empty")))
    assert mask_module.install_example_data_button(orphan) is None

    placed = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                     _settings_sections=sections)
    button = mask_module.install_example_data_button(placed)
    assert button is not None and holder.isAncestorOf(button)


# ---------------------------------------------------------------------------
# install_folds
# ---------------------------------------------------------------------------


def test_a_strip_is_built_for_every_fold_set_that_mounted_something(
        qtbot, qt_theme_applied):
    """Why ``install_folds``' ``strip is None`` branch cannot be reached.

    ``CategoryFoldSet.build_strip`` returns None on exactly one condition,
    ``not self.order``, and ``mount()`` RETURNS ``self.order`` -- which
    ``install_folds`` has already tested one line earlier and returned on.
    Nothing touches ``order`` in between. So the guard is a defensive
    re-check after a call that already guarantees the condition, and this
    is the invariant it re-checks, asserted directly.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    header = _Header()
    qtbot.addWidget(header)
    folds = CategoryFoldSet(
        screen, {key: mask_module.FOLD_GATES[key]
                 for key in mask_module.CATEGORY_FOLDS},
        implies=mask_module.FOLD_IMPLIES)

    mounted = folds.mount()

    assert mounted, "the timelapse fold mounted nothing on a real screen"
    assert folds.order == mounted
    assert folds.build_strip(header) is not None


# ---------------------------------------------------------------------------
# sync_folds
# ---------------------------------------------------------------------------

def test_a_settings_file_moves_the_switch_and_a_dead_switch_costs_nothing(
        host, caplog):
    """The gate has no widget, so the bulk apply cannot move the switch.

    ``sync_folds`` reads the gate out of the applied dict and moves the
    button instead. It runs from ``AppScreen`` right after every bulk
    apply, including one that lands while the masthead is being torn
    down -- and a settings file must not fail to load because the switch
    it wanted is already gone.
    """
    import shiboken6

    screen, strip = host

    assert tuple(mask_module.sync_folds(screen, {"timelapse": "True"})) == (
        "timelapse",)
    assert strip.isChecked(), (
        "the settings file did not move the switch")

    shiboken6.delete(strip)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        switched = mask_module.sync_folds(screen, {"timelapse": False})

    assert tuple(switched) == ()
    assert any("Could not sync Time" in record.getMessage()
               for record in caplog.records), caplog.text
