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

from PySide6.QtWidgets import QCheckBox, QLineEdit, QPushButton, QWidget

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


class _Console:
    def __init__(self):
        self.text = []

    def append_stdout(self, text):
        self.text.append(text)


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
    strip = mask_module.install_folds(screen)
    assert strip is not None, "Mask Generation got no fold strip"
    return screen, strip


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

    assert set(previews) == set(mask_module.FOLDED_APPS)
    assert previews == screen._fold_previews
    previews.clear()
    assert set(screen._fold_previews) == set(mask_module.FOLDED_APPS), (
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


def test_a_fold_with_no_button_on_the_strip_gets_no_preview(qtbot,
                                                            monkeypatch):
    """The switch is what offers the panel, so no switch means no panel."""
    from spacr.qt import preview_registry

    built = []

    def attach(screen, key):
        built.append(key)
        return _Host()

    monkeypatch.setattr(preview_registry, "attach_folded", attach)
    folds = _fold_set(("timelapse", "motility"))
    strip = FoldStrip([("timelapse", lambda on: None, True)])
    qtbot.addWidget(strip)

    offered = mask_module._offer_fold_previews(_Screen(), folds, strip)

    assert set(offered) == {"timelapse"}
    assert built == ["timelapse"], (
        "a fold with no button still had its preview built")


def test_a_preview_that_will_not_build_costs_only_that_panel(qtbot,
                                                             monkeypatch,
                                                             caplog):
    """One fold's broken preview must not take the other's away."""
    from spacr.qt import preview_registry

    def attach(screen, key):
        if key == "timelapse":
            raise RuntimeError("the preview builder is broken")
        return _Host()

    monkeypatch.setattr(preview_registry, "attach_folded", attach)
    folds = _fold_set(("timelapse", "motility"))
    strip = FoldStrip([("timelapse", lambda on: None, True),
                       ("motility", lambda on: None, True)])
    qtbot.addWidget(strip)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        offered = mask_module._offer_fold_previews(_Screen(), folds, strip)

    assert set(offered) == {"motility"}
    assert any("Could not attach" in record.getMessage()
               for record in caplog.records), caplog.text


def test_a_fold_that_declares_no_preview_is_simply_skipped(qtbot,
                                                           monkeypatch):
    """Most folds are settings only; ``attach_folded`` answers None."""
    from spacr.qt import preview_registry

    monkeypatch.setattr(
        preview_registry, "attach_folded",
        lambda screen, key: None if key == "timelapse" else _Host())
    folds = _fold_set(("timelapse", "motility"))
    strip = FoldStrip([("timelapse", lambda on: None, True),
                       ("motility", lambda on: None, True)])
    qtbot.addWidget(strip)

    offered = mask_module._offer_fold_previews(_Screen(), folds, strip)

    assert set(offered) == {"motility"}


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


def test_the_button_is_only_offered_on_mask_generation(qtbot):
    """Its whole job is filling this form; another screen's is not it."""
    src, _holder, sections = _src_form(qtbot)

    guest = _Screen(app_key="measure", _settings_model=_Model({"src": src}),
                    _settings_sections=sections)
    assert mask_module.install_example_data_button(guest) is None
    assert getattr(guest, "_example_images_button", None) is None

    owner = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                    _settings_sections=sections)
    assert isinstance(mask_module.install_example_data_button(owner),
                      QPushButton)


def test_a_form_with_no_src_control_gets_no_button(qtbot):
    """The button fills ``src``; a form without one has nothing to fill."""
    src, _holder, sections = _src_form(qtbot)

    blank = _Screen(app_key="mask", _settings_model=_Model(),
                    _settings_sections=sections)
    assert mask_module.install_example_data_button(blank) is None

    filled = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                     _settings_sections=sections)
    assert mask_module.install_example_data_button(filled) is not None


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


def test_the_button_lands_above_src_says_what_it_does_and_loads_a_plate(
        qtbot, monkeypatch, tmp_path):
    """Installed once, in the section holding ``src``, and wired to run."""
    from spacr.qt import synthetic

    src, holder, sections = _src_form(qtbot)
    console = _Console()
    screen = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                     _settings_sections=sections, _console=console)

    button = mask_module.install_example_data_button(screen)

    assert button.text() == mask_module.EXAMPLE_BUTTON_TEXT
    assert button.toolTip() == mask_module.EXAMPLE_BUTTON_TOOLTIP
    assert holder.isAncestorOf(button)
    assert screen._example_images_button is button
    assert mask_module.install_example_data_button(screen) is button, (
        "a second call installed a second button")

    layout = type("_L", (), {"image_files": [str(tmp_path / "a.tif")],
                             "src": str(tmp_path),
                             "notes": {"channels": (0,), "n_fields": 1}})()
    monkeypatch.setattr(synthetic, "generate_mask_demo", lambda dst: layout)
    monkeypatch.setattr(synthetic, "demo_settings",
                        lambda app, source: {"src": source})
    monkeypatch.setattr(mask_module, "example_plate_folder",
                        lambda: str(tmp_path))

    button.click()

    assert screen.applied == [{"src": str(tmp_path)}], (
        "pressing the button applied nothing to the form")
    assert "Example plate ready" in "".join(console.text)


# ---------------------------------------------------------------------------
# install_folds
# ---------------------------------------------------------------------------

def test_a_button_that_cannot_be_installed_does_not_cost_the_switches(
        qtbot, qt_theme_applied, monkeypatch, caplog):
    """The example plate is installed first and outside the folds' guard.

    It has its own guard for the same reason: a screen with switches and
    no example button is a smaller screen, and one with neither is a
    regression twice over.
    """
    from spacr.qt.screens.app_screen import AppScreen

    def refuse(screen):
        raise RuntimeError("the src section moved")

    monkeypatch.setattr(mask_module, "install_example_data_button", refuse)
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        strip = mask_module.install_folds(screen)

    assert strip is not None, "a broken button took the fold strip with it"
    assert list(strip.keys()) == list(mask_module.FOLDED_APPS)
    assert any("Could not install the example-plate button"
               in record.getMessage() for record in caplog.records), (
        caplog.text)


def test_a_screen_with_no_settings_form_carries_no_strip(qtbot,
                                                         qt_theme_applied):
    """A fold that mounts nothing is not a switch with nothing behind it.

    ``CategoryFold.mount`` returns False when the host has no settings
    model to mount onto, and a strip built anyway would put a switch on
    the masthead that reveals no categories at all.
    """
    from spacr.qt.screens.app_screen import AppScreen

    header = _Header()
    qtbot.addWidget(header)
    bare = _Screen(app_key="mask", _header=header)

    assert mask_module.install_folds(bare) is None
    assert header.trailing == [], "an empty strip was hung on the masthead"
    assert mask_module.fold_set(bare) is None

    real = AppScreen(app_key="mask")
    qtbot.addWidget(real)
    assert mask_module.install_folds(real) is not None
    assert mask_module.fold_set(real) is not None


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
                 for key in mask_module.FOLDED_APPS},
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
    assert strip.button_for("timelapse").isChecked(), (
        "the settings file did not move the switch")

    shiboken6.delete(strip.button_for("timelapse"))

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        switched = mask_module.sync_folds(screen, {"timelapse": False})

    assert tuple(switched) == ()
    assert any("Could not sync the folds" in record.getMessage()
               for record in caplog.records), caplog.text
