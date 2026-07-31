"""The chosen icon set, the tabbed Home screen, the edge reveal, and pause.

Three separate contracts live here, in this order:

1. **The installed icons are the ones that were chosen.** Seventeen were
   replaced from ``backup_icons``; six were deliberately left alone.
   Both halves are asserted by name, because "the user reviewed these"
   is exactly the kind of decision that gets silently undone.
2. **The Home layout claims are true.** Every app is in exactly one tab,
   the tab names are the section names, a running job shows up, the
   queue segment appears only when there is a queue, and the app drawer
   reveals on dwell — not on a passing pointer — and is reachable from
   the keyboard.
3. **Pause is honest.** The gate genuinely blocks a worker thread; the
   control is genuinely disabled for every pipeline spaCR actually
   ships, because none of them poll it.
"""
from __future__ import annotations

import hashlib
import os
import threading

import pytest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QLabel, QPushButton

from spacr.qt import bridge, iconset
from spacr.qt.app import (APPS, SECTIONS, _FORCE_GLYPH, _ICON_OVERRIDES,
                          make_home_page, section_members)
from spacr.qt.widgets.home import AppTile, HomePage, PAUSE_UNAVAILABLE

BACKUP_DIR = os.path.join(iconset.RESOURCE_DIR, "backup_icons")

#: The seventeen the user picked, canonical name -> chosen candidate.
CHOSEN = {
    "mask":         "mask_01.png",
    "abort":        "abort_01.png",
    "activation":   "activation_01.png",
    "annotate":     "annotate_01.png",
    "cellpose_all": "cellpose_all_01.png",
    "classify":     "classify_05.png",
    "convert":      "convert_04.png",
    "default":      "default_01.png",
    "download":     "download_05.png",
    "make_masks":   "make_masks_09.png",
    "map_barcodes": "map_barcodes_05.png",
    "regression":   "regression_03.png",
    "run":          "run_03.png",
    "sequencing":   "sequencing_06.png",
    "settings":     "settings_02.png",
    "umap":         "umap_01.png",
    # Drawn later, for the four apps that had nothing worth keeping:
    # queue and batch both aliased sequencing.png, invasion rendered a
    # Font Awesome glyph, replication did not exist.
    "queue":        "queue_01.png",
    "batch":        "batch_07.png",
    "invasion":     "invasion_01.png",
    # Re-picked later: classify was classify_05, replication was
    # replication_02. `recruitment` left this table entirely — its new
    # artwork comes from another family, so it lives in CROSS_FAMILY.
    "classify":     "classify_06.png",
    "replication":  "replication_04.png",
}

#: Installed name -> (backup family, candidate) for the picks that do
#: NOT come from a folder named after the app.
#:
#: :data:`CHOSEN` derives the source folder from the installed name, so
#: it can only express "the user picked one of *this* app's candidates".
#: Half of the later round were cross-picks — Align & Stitch got a
#: cellpose_all drawing, Import Project a download one — and without a
#: table that can say so, those files would be installed and asserted by
#: nothing.
#:
#: ``motility`` is the odd one: it is not a new pick at all. It rendered
#: ``recruitment.png`` through an override, and when Recruitment was
#: given new artwork Motility Assay would have silently followed it —
#: two apps, one picture, neither of them asked for it. Installing the
#: artwork Motility was already showing under its own name keeps the
#: pixels identical and breaks the link.
CROSS_FAMILY = {
    "align":         ("cellpose_all",   "cellpose_all_01.png"),
    "recruitment":   ("cellpose_masks", "cellpose_masks_06.png"),
    "foreign":       ("download",       "download_03.png"),
    "train_compare": ("settings",       "settings_07.png"),
    "db_browser":    ("batch",          "batch_01.png"),
    "timelapse":     ("convert",        "convert_09.png"),
    "motility":      ("recruitment",    "recruitment_02.png"),
}

#: Candidates were generated for these too and the user picked none, so
#: the shipped artwork must still be the shipped artwork.
KEPT = ("cellpose_masks", "measure", "plaque", "ml_analyze",
        "train_cellpose", "logo_spacr")


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@pytest.fixture(autouse=True)
def _empty_registry():
    """The run registry is process-wide; never leak a job between tests."""
    bridge.registry().clear()
    yield
    bridge.registry().clear()


@pytest.fixture
def _empty_journal(tmp_path, monkeypatch):
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def home(qtbot, qt_theme_applied, _empty_journal):
    page = make_home_page()  # the page MainWindow ships
    qtbot.addWidget(page)
    page.resize(1200, 860)
    page.show()
    qtbot.waitExposed(page)
    return page


# ===========================================================================
# 1. The installed icon set
# ===========================================================================

@pytest.mark.parametrize("name,candidate", sorted(CHOSEN.items()))
def test_the_chosen_icon_is_the_one_installed(name, candidate):
    installed = os.path.join(iconset.RESOURCE_DIR, f"{name}.png")
    source = os.path.join(BACKUP_DIR, name, candidate)
    assert os.path.isfile(installed), f"{name}.png is missing"
    assert os.path.isfile(source), f"{candidate} vanished from backup_icons"
    assert _digest(installed) == _digest(source), (
        f"{name}.png is not {candidate} — the chosen artwork was replaced")


@pytest.mark.parametrize("name,family,candidate",
                         sorted((n, f, c) for n, (f, c)
                                in CROSS_FAMILY.items()))
def test_a_cross_family_pick_is_installed_under_its_own_name(
        name, family, candidate):
    """The artwork is the chosen one, and it is filed under the app.

    The second half matters as much as the first: leaving these as
    ``cellpose_all.png`` and reaching them through ``_ICON_OVERRIDES``
    is how ``timelapse`` ended up pointing at ``run.png`` with a comment
    explaining that ``convert.png`` belonged to somebody else. An app
    with its own picture should have its own file."""
    installed = os.path.join(iconset.RESOURCE_DIR, f"{name}.png")
    source = os.path.join(BACKUP_DIR, family, candidate)
    assert os.path.isfile(source), f"{family}/{candidate} vanished"
    assert os.path.isfile(installed), f"{name}.png is missing"
    assert _digest(installed) == _digest(source), (
        f"{name}.png is not {family}/{candidate}")

    from spacr.qt.app import _FORCE_GLYPH, _ICON_OVERRIDES
    assert name not in _ICON_OVERRIDES, (
        f"{name} has its own artwork; the override is stale")
    assert name not in _FORCE_GLYPH, (
        f"{name} has its own artwork; the glyph would hide it")


def test_the_glyph_escape_hatch_is_kept_even_though_it_is_empty():
    """``align`` was the last entry, and the user overruled it.

    The set stays: it is the documented route for an app whose meaning
    no bundled artwork carries, and a mechanism deleted the moment it is
    unused is a mechanism the next person re-invents."""
    from spacr.qt.app import _FORCE_GLYPH
    assert _FORCE_GLYPH == set()


def test_no_two_apps_render_the_same_picture_by_accident(qapp):
    """Every duplicate is a deliberate, documented borrowing.

    Four apps share another app's artwork on purpose (an override with a
    reason next to it). Anything else is two tiles a user cannot tell
    apart — which is what ``queue`` and ``batch`` both aliasing
    sequencing.png looked like, and what Motility Assay would have
    become when Recruitment was re-skinned."""
    from spacr.qt.app import APPS, _icon_for_app
    by_pixels: dict = {}
    for key, name, *_rest in APPS:
        icon = _icon_for_app(key)
        assert icon is not None and not icon.isNull(), f"{name} has no icon"
        blob = bytes(icon.pixmap(48, 48).toImage().constBits())
        by_pixels.setdefault(blob, []).append(name)
    shared = {frozenset(v) for v in by_pixels.values() if len(v) > 1}
    assert shared == {
        frozenset({"Mask", "Model Compare"}),
        frozenset({"Annotate", "Annotator Agreement"}),
        frozenset({"Map Barcodes", "Plate Viewer"}),
        frozenset({"Cellpose Masks", "Train Cellpose"}),
    }, f"unexpected identical artwork: {shared}"


@pytest.mark.parametrize("name", KEPT)
def test_an_icon_nobody_asked_to_change_was_not_changed(name):
    """"If a module icon is not mentioned, I like its current icon."

    Asserted as: the shipped file is none of the candidates that were
    drawn for it. That is a stronger claim than a hash of the current
    bytes, and it does not have to be updated when the artwork is
    legitimately revised later.
    """
    installed = os.path.join(iconset.RESOURCE_DIR, f"{name}.png")
    assert os.path.isfile(installed)
    folder = os.path.join(BACKUP_DIR, name)
    if not os.path.isdir(folder):
        pytest.skip(f"no candidate set was generated for {name}")
    shipped = _digest(installed)
    matches = [f for f in sorted(os.listdir(folder))
               if f.lower().endswith(".png") and not f.startswith("_")
               and _digest(os.path.join(folder, f)) == shipped]
    assert not matches, (
        f"{name}.png was swapped for {matches} — it was left off the "
        "list on purpose")


def test_cellpose_all_and_cellpose_masks_are_no_longer_the_same_file():
    """They shipped byte-identical. Splitting them was the point."""
    a = _digest(os.path.join(iconset.RESOURCE_DIR, "cellpose_all.png"))
    b = _digest(os.path.join(iconset.RESOURCE_DIR, "cellpose_masks.png"))
    assert a != b


def test_every_app_key_resolves_to_a_visible_icon(qapp):
    from spacr.qt.app import _icon_for_app
    blank = [k for k, *_r in APPS if _icon_for_app(k).isNull()
             or _icon_for_app(k).pixmap(24, 24).isNull()]
    assert not blank, f"apps with no icon: {blank}"


@pytest.mark.parametrize("theme_name", ("dark", "light", "space"))
def test_the_new_artwork_still_clears_the_contrast_bar(theme_name, qapp):
    """The re-inking is theme-blind by design, so 1024 px artwork with a
    different alpha mask must not change the answer."""
    weak = []
    for path in iconset.bundled_icon_paths():
        ratio = iconset.icon_contrast(path, theme_name)
        if ratio < iconset.MIN_ICON_CONTRAST:
            weak.append(f"{os.path.basename(path)}: {ratio:.2f}:1")
    assert not weak, f"{theme_name} hides icons: {weak}"


@pytest.mark.parametrize("theme_name", ("dark", "light", "space"))
@pytest.mark.parametrize("name", sorted(CHOSEN))
def test_each_new_icon_is_a_flat_mask_painted_in_the_theme_ink(
        name, theme_name, qapp):
    """Every chosen icon is monochrome-on-transparent, so re-inking must
    paint it flat rather than inventing shading from exporter noise."""
    import numpy as np
    from spacr.qt import theme as theme_mod

    path = os.path.join(iconset.RESOURCE_DIR, f"{name}.png")
    arr = iconset.themed_array(path, theme_name)
    assert arr is not None
    opaque = arr[:, :, 3] > 200
    assert opaque.any(), f"{name}.png has no solid pixels"
    painted = np.unique(arr[opaque][:, :3].reshape(-1, 3), axis=0)
    assert len(painted) == 1, f"{name}.png was not painted flat"
    expected = iconset._hex_to_array(theme_mod.palette_for(theme_name)["fg"])
    assert np.array_equal(painted[0], expected.astype(np.uint8))


def _shaded_mask(bright: bool):
    """A monochrome icon that genuinely carries shading in RGB.

    Every icon spaCR bundles today is a flat alpha mask, so the branch
    of :func:`iconset.reink` that maps a *tonal* monochrome image onto
    the veil→ink band has no artwork left to exercise it. It is still
    the branch that runs the day someone draws a shaded icon, so it is
    tested on a synthetic ramp instead of on whatever happens to ship.

    :param bright: draw light-on-transparent (as the old ``umap.png``
        was) rather than dark-on-transparent.
    """
    import numpy as np
    ramp = np.linspace(0.0, 255.0, 64)
    if not bright:
        ramp = 255.0 - ramp
    rgba = np.zeros((64, 64, 4), dtype=np.float64)
    rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = ramp[None, :]
    rgba[:, :, 3] = 255.0
    return rgba


@pytest.mark.parametrize("bright", (True, False))
@pytest.mark.parametrize("theme_name", ("dark", "light", "space"))
def test_shaded_monochrome_artwork_is_mapped_onto_the_ink_band(
        bright, theme_name, qapp):
    import numpy as np
    from spacr.qt import theme as theme_mod

    rgba = _shaded_mask(bright)
    assert iconset.carries_tonal_structure(rgba)
    out = iconset.reink(rgba, theme_name)

    lum = (0.2126 * out[:, :, 0] + 0.7152 * out[:, :, 1]
           + 0.0722 * out[:, :, 2]).astype(float)
    # Shading survived rather than being flattened to a silhouette…
    assert len(np.unique(out[:, :, :3].reshape(-1, 3), axis=0)) > 8
    # …and it is a genuine ramp, brightest where the drawing is densest.
    row = lum[0]
    assert (np.all(np.diff(row) >= -0.5) or np.all(np.diff(row) <= 0.5))
    # The band runs between the veil and the theme's ink, nothing outside.
    ink = iconset._hex_to_array(theme_mod.palette_for(theme_name)["fg"])
    veil = iconset._hex_to_array(iconset.veil_color(theme_name))
    lo = min(veil.min(), ink.min()) - 1.0
    hi = max(veil.max(), ink.max()) + 1.0
    assert out[:, :, :3].min() >= lo and out[:, :, :3].max() <= hi
    # Alpha is untouched: the shape always lives there.
    assert np.array_equal(out[:, :, 3], rgba[:, :, 3].astype(np.uint8))


def test_the_1024px_artwork_is_worked_on_downscaled(qapp):
    """1024 px is four times the largest slot; the loader caps it."""
    from PIL import Image
    for name in CHOSEN:
        path = os.path.join(iconset.RESOURCE_DIR, f"{name}.png")
        with Image.open(path) as im:
            assert max(im.size) == 1024, f"{name}.png is not the new artwork"
        rgba = iconset._load_rgba(path)
        assert max(rgba.shape[:2]) <= iconset.MAX_WORK_SIZE


# -- the two aliases that had to move ---------------------------------------

def test_model_compare_no_longer_borrows_the_batch_icon():
    """cellpose_all is now "a whole batch of frames"; Model Compare is
    one field segmented twice, which is what mask.png draws."""
    assert _ICON_OVERRIDES["model_compare"] == "mask.png"
    assert "cellpose_all.png" not in _ICON_OVERRIDES.values()


def test_align_has_real_artwork_now_and_no_longer_needs_a_glyph(qapp):
    """Was ``test_align_renders_a_glyph_because_no_png_says_stitched_
    mosaic``, asserting ``"align" in _FORCE_GLYPH``.

    That was a judgement — no bundled PNG read as "tiles registered into
    ONE canvas", so Align & Stitch drew ``fa5s.border-all``. The user has
    since chosen ``cellpose_all_01`` for it, which overrules the
    judgement, and ``align.png`` is installed. The glyph mapping stays in
    the iconset (it is still the right glyph if the PNG ever goes), so
    that is asserted separately from what actually renders."""
    from spacr.qt.app import _icon_for_app
    assert "align" not in _FORCE_GLYPH
    assert "align" not in _ICON_OVERRIDES
    assert os.path.isfile(os.path.join(iconset.RESOURCE_DIR, "align.png"))
    assert not _icon_for_app("align").pixmap(24, 24).isNull()
    # The glyph it used to draw is still mapped, and is still a real one
    # rather than the puzzle-piece fallback — that is what makes
    # _FORCE_GLYPH a usable escape hatch if the PNG is ever withdrawn.
    assert iconset._NAME_TO_GLYPH["align"] == "fa5s.border-all"
    assert (iconset._NAME_TO_GLYPH["align"]
            != iconset._NAME_TO_GLYPH.get("__missing__",
                                          "fa5s.puzzle-piece"))
    # …and the PNG is what is on screen, not that glyph.
    assert (_icon_for_app("align").pixmap(24, 24).toImage()
            != iconset.icon("align").pixmap(24, 24).toImage())


def test_invasion_has_real_artwork_now_and_no_longer_needs_a_glyph(qapp):
    from spacr.qt.app import _icon_for_app
    assert "invasion" not in _FORCE_GLYPH
    assert "invasion" not in _ICON_OVERRIDES
    assert os.path.isfile(os.path.join(iconset.RESOURCE_DIR, "invasion.png"))
    assert not _icon_for_app("invasion").pixmap(24, 24).isNull()


@pytest.mark.parametrize("key", ("queue", "batch", "invasion", "replication"))
def test_the_four_redrawn_apps_own_their_icon_outright(key, qapp):
    """No override, no alias — the file is named after the app."""
    assert key not in _ICON_OVERRIDES
    assert key not in _FORCE_GLYPH
    assert iconset.bundled_icon_path(key) == os.path.join(
        iconset.RESOURCE_DIR, f"{key}.png")


def test_queue_and_batch_no_longer_render_as_the_same_picture(qapp):
    """They both aliased sequencing.png — one helix standing in for two
    apps that do different things. The distinction is the whole reason
    they were redrawn: queue is the same settings over many plates,
    batch is arbitrary module+plate combinations in sequence."""
    from spacr.qt.app import _icon_for_app

    def raster(key):
        return _icon_for_app(key).pixmap(48, 48).toImage()

    assert raster("queue") != raster("batch")
    assert raster("queue") != raster("map_barcodes")
    assert _digest(os.path.join(iconset.RESOURCE_DIR, "queue.png")) != _digest(
        os.path.join(iconset.RESOURCE_DIR, "sequencing.png"))
    assert _digest(os.path.join(iconset.RESOURCE_DIR, "batch.png")) != _digest(
        os.path.join(iconset.RESOURCE_DIR, "sequencing.png"))


def test_invasion_and_replication_are_not_the_same_picture(qapp):
    """Both are "parasite plus host". Invasion is about the membrane,
    replication about the vacuole, and the icons have to say which."""
    from spacr.qt.app import _icon_for_app
    assert (_icon_for_app("invasion").pixmap(48, 48).toImage()
            != _icon_for_app("replication").pixmap(48, 48).toImage())


# ---------------------------------------------------------------------------
# The replication assay, which had never been wired up
# ---------------------------------------------------------------------------

def test_the_replication_assay_is_a_first_class_module():
    """The parasites-per-vacuole assay has complete GUI wiring.

    It briefly asserted ``row[3] == SECTION_BETA`` while #16i had staging
    as a section. #16j put every app back under what it is *about* and
    made maturity a colour, so this is a Toxoplasma assay again — and
    separately a beta one, which is now a lookup in ``APP_STAGE``."""
    from spacr.qt.app import SECTION_TOXO, app_stage
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    from spacr.qt.screens.settings_model import resolve_default_settings

    row = next((a for a in APPS if a[0] == "replication"), None)
    assert row is not None, "replication is not in the app registry"
    assert row[3] == SECTION_TOXO
    assert app_stage("replication") == "beta"
    assert row[3] in SECTIONS
    assert row[1] and row[2]
    assert APP_TITLES.get("replication")
    assert APP_INTROS.get("replication")

    entry = bridge.resolve_pipeline_entry("replication")
    assert entry is not None
    inner = getattr(entry, "__wrapped__", entry)
    assert getattr(inner, "__name__", "") == "analyze_replication"

    settings = resolve_default_settings("replication")
    assert isinstance(settings, dict) and "src" in settings
    assert settings["vacuole_key"] == "auto"


def test_the_replication_screen_opens(qtbot, qt_theme_applied,
                                      _empty_journal):
    from PySide6.QtWidgets import QWidget
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    screen = win._build_screen("replication")
    assert isinstance(screen, QWidget)
    assert getattr(screen, "app_key", None) == "replication"
    screen.deleteLater()


# ===========================================================================
# 2. The Home layout
# ===========================================================================

def test_the_categories_are_the_ones_that_were_asked_for():
    """The section vocabulary, recorded so a rename is deliberate.

    Five names and six tabs counting Home.

    It briefly asserted seven and eight. #16i added "Alpha modules" and
    "Beta modules" as CATEGORIES, which took every app out of Data,
    Segmentation models and Results & QC and gave "where is the format
    converter" two answers. #16j deleted those two tabs and kept the
    classification, as a hover colour — see
    ``test_the_alpha_and_beta_lists_are_the_ones_that_were_asked_for``.
    """
    from spacr.qt import app as app_mod
    assert app_mod.SECTION_CORE == "Core"
    assert app_mod.SECTION_DATA == "Data"
    assert app_mod.SECTION_MODELS == "Segmentation models"
    assert app_mod.SECTION_RESULTS == "Results & QC"
    assert app_mod.SECTION_TOXO == "Toxoplasma"
    assert app_mod.SECTIONS == (
        "Core", "Data", "Segmentation models", "Results & QC", "Toxoplasma")
    # The staging categories are gone as *places*. Named here so that
    # re-adding one has to argue with this line first.
    assert not hasattr(app_mod, "SECTION_ALPHA")
    assert not hasattr(app_mod, "SECTION_BETA")
    assert not hasattr(app_mod, "MATURITY_SECTIONS")
    assert not hasattr(app_mod, "STAGED_FROM")
    assert len(app_mod.SECTIONS) + 1 == 6


def test_every_app_has_a_stage_and_it_is_written_down_once():
    """``APP_STAGE`` is the only record of how finished an app is.

    It replaced ``STAGED_FROM``, which recorded the opposite thing: what
    subject an app had been taken *out of*. Nothing is taken out of
    anything now, so the table only has to say what is not signed off —
    and signing an app off is deleting its line.
    """
    from spacr.qt.app import APP_STAGE, STAGES, app_stage

    assert set(APP_STAGE.values()) == {"alpha", "beta"}, (
        "a stable app has a line in APP_STAGE — 'stable' is the absence "
        "of one, or the table grows a second way to say the same thing")
    keys = {row[0] for row in APPS}
    assert set(APP_STAGE) <= keys, (
        f"APP_STAGE names apps that do not exist: {set(APP_STAGE) - keys}")
    for key in keys:
        assert app_stage(key) in STAGES
    assert app_stage("no such app") == "stable"


def test_home_is_the_first_tab_and_holds_everything(home):
    """The first tab is not a category — it is all of them.

    The categories-only version is the one the user rejected as "too
    empty": a handful of tiles and a blank half-page, with no view that
    showed what spaCR can do."""
    assert home._tabs.count() == len(SECTIONS) + 1
    assert home._tabs.tabText(0) == f"Home  ({len(APPS)})"
    assert home._tabs.currentIndex() == 0
    drawn = {t.text_label for t in home._tabs.widget(0).findChildren(AppTile)}
    assert drawn == {name for _k, name, *_r in APPS}


def test_the_category_tabs_follow_the_workflow_order(home):
    """Six tabs, and the count in each label is the tab's own size.

    Was seven while Alpha and Beta had tabs of their own. ``section_members``
    is what the tab draws, so it is what the label has to count.
    """
    labels = [home._tabs.tabText(i) for i in range(1, home._tabs.count())]
    assert len(labels) == len(SECTIONS) == 5
    for label, section in zip(labels, SECTIONS):
        # "&&" is how Qt is told to draw a literal ampersand.
        assert label.startswith(section.replace("&", "&&"))
        assert label.endswith(f"({len(section_members(section))})")


def test_a_tab_label_draws_its_ampersand_instead_of_eating_it(home):
    """A lone & is a mnemonic: "Results & QC" rendered as "Results  QC".

    Exactly one of the six labels is affected, which is exactly the
    kind of thing that ships unnoticed. Both halves are asserted: the
    helper does the doubling, and no label on screen carries a lone
    ampersand."""
    from spacr.qt.widgets.home import _escape_amp
    assert _escape_amp("Results & QC") == "Results && QC"

    labels = [home._tabs.tabText(i) for i in range(home._tabs.count())]
    ampersanded = [s for s in SECTIONS if "&" in s]
    assert ampersanded, (
        "no section name has an ampersand any more — this test is the "
        "only thing keeping the escaping honest, so say so here")
    for section in ampersanded:
        assert any(section.replace("&", "&&") in label
                   for label in labels)
    assert not any(label.count("&") == 1 for label in labels)


def test_the_home_tab_bands_are_the_categories_themselves(home):
    """Home bands by *section*, with no second membership table.

    It used to band into Prepare / Run / Review through
    ``HomePage._BAND_FOR_SECTION`` plus a three-app ``_BAND_OVERRIDE``,
    which meant Plate Queue was "Data" on its own tab and "Run" on Home,
    and a renamed section dropped its apps into a fallback band. The
    bands are now handed to the page by the registry, so this asserts
    what is on screen rather than that two tables agree.

    All five bands are drawn again. While staging was a section, three
    of them were empty and Home drew four bands out of seven headings;
    the guard for an empty band stays, because a heading with nothing
    under it is worse than no heading."""
    from spacr.qt.app import home_bands
    assert not hasattr(HomePage, "_BAND_FOR_SECTION")
    assert not hasattr(HomePage, "_BAND_OVERRIDE")
    assert not hasattr(HomePage, "BANDS")

    # Every band heading on the Home tab is a section name, in order…
    page = home._tabs.widget(0)
    headings = [lbl.text() for lbl in page.findChildren(QLabel)
                if lbl.text() in {s.upper() for s in SECTIONS}]
    assert headings == [s.upper() for s, _rows in home_bands()]
    assert headings == [s.upper() for s in SECTIONS
                        if any(a[3] == s for a in APPS)]

    # …and every app really does land in exactly one band on screen.
    tiles = page.findChildren(AppTile)
    assert len(tiles) == len(APPS)
    assert len({t.text_label for t in tiles}) == len(APPS)


def test_every_app_is_on_home_and_on_exactly_one_category_tab(home):
    """Back to *exactly* one.

    It was relaxed to "at least one" for #16i, where a staged app was on
    its subject tab and again on its staging tab. With the staging tabs
    gone there is one place per app, which is the property that makes
    the tabs a filter of Home rather than a second index of it.
    """
    placement: dict = {}
    for index in range(1, home._tabs.count()):
        for tile in home._tabs.widget(index).findChildren(AppTile):
            placement.setdefault(tile.text_label, []).append(index)
    expected = {name for _k, name, *_r in APPS}
    assert set(placement) == expected, (
        f"missing: {expected - set(placement)}; "
        f"unexpected: {set(placement) - expected}")
    assert not {n: t for n, t in placement.items() if len(t) != 1}, (
        f"apps on more than one tab: {placement}")


def test_each_tab_holds_exactly_its_own_members(home):
    """``section_members`` is the registry's answer to "what does this
    tab show", so it is what the drawn tiles are checked against."""
    for index, section in enumerate(SECTIONS, start=1):
        page = home._tabs.widget(index)
        drawn = {t.text_label for t in page.findChildren(AppTile)}
        expected = {row[1] for row in section_members(section)}
        assert drawn == expected, f"{section} tab is wrong"


#: The two maturity lists, in the user's own words. This is the record of
#: what was asked for, so a later move out of alpha or beta has to be
#: made here as well as in the registry.
#:
#: They were briefly SECTIONS. They are stages now, and the difference is
#: that an app in one of these lists is still filed under what it does —
#: it just lights a different colour on hover.
ALPHA_MODULES = {
    "align", "model_zoo", "convert", "foreign", "external_masks",
    "model_compare", "queue", "batch", "invasion", "db_browser",
    "distributed_jobs",
    "plate_view", "agreement", "train_compare", "classifier_evaluation",
    "run_history", "report",
}
BETA_MODULES = {
    "make_masks", "train_cellpose", "cellpose_masks", "timelapse",
    "motility", "analyze_plaques", "replication", "umap", "activation",
}


def test_the_alpha_and_beta_lists_are_the_ones_that_were_asked_for():
    """17 alpha, 9 beta, named one at a time.

    Spelling the lists out means a quiet drift fails here rather than
    being noticed in a screenshot."""
    from spacr.qt.app import app_stage
    by_stage: dict = {}
    for key, _name, _desc, _section in APPS:
        by_stage.setdefault(app_stage(key), set()).add(key)
    assert by_stage["alpha"] == ALPHA_MODULES
    assert by_stage["beta"] == BETA_MODULES
    assert len(ALPHA_MODULES) == 17 and len(BETA_MODULES) == 9
    assert by_stage["stable"] == (
        {row[0] for row in APPS} - ALPHA_MODULES - BETA_MODULES)


def test_every_category_carries_its_one_line_note(window):
    """A category with a handful of apps in it looks thin until it says
    what it is for.

    It used to also assert that the Toxoplasma note named the three
    assays staged out of it. There is nothing staged out of anything
    now — all four assays are on the Toxoplasma tab — so the note is
    back to being a description of the subject."""
    from spacr.qt.app import SECTION_NOTES

    assert set(SECTION_NOTES) == set(SECTIONS), (
        "a category has no note — a thin tab would explain nothing")
    assert all(SECTION_NOTES.values())

    tabs = window._startup._tabs
    for index, section in enumerate(SECTIONS, start=1):
        notes = [lbl.text() for lbl in tabs.widget(index).findChildren(QLabel)
                 if lbl.objectName() == "HomeSectionNote"]
        assert notes == [SECTION_NOTES[section]], (
            f"{section} tab drew {notes!r}")


def test_no_category_tab_needs_a_scrollbar_on_a_laptop(window, qapp):
    """1440x900 is the size this layout is dimensioned for.

    **The Home tab is exempt, and that is a decision, not an oversight.**
    Thirty tiles at the size the user asked for — icon over name, packed
    tight — is eight rows plus five headings, about 1200 px of content.
    No arrangement of thirty large tiles fits 670 px of pane; the only
    way to make Home not scroll is to make its tiles small again, which
    is the thing #16j exists to undo. Every *category* tab still fits,
    and that is what this checks."""
    window.resize(1440, 900)
    window.show()
    qapp.processEvents()
    tabs = window._startup._tabs
    overflowing = {}
    for index in range(1, tabs.count()):
        tabs.setCurrentIndex(index)
        qapp.processEvents()
        bar = tabs.widget(index).verticalScrollBar()
        if bar.maximum() > 0:
            overflowing[tabs.tabText(index)] = bar.maximum()
    tabs.setCurrentIndex(0)
    assert not overflowing, (
        f"tabs that scroll at 1440x900: {overflowing}. Take the room out "
        "of the tile or the gaps, not out of the app list.")


def test_every_tab_uses_the_same_large_tile(home):
    """One tile class, one size, every tab including Home.

    Was ``..._carries_the_large_cards`` plus
    ``..._home_tab_tiles_are_the_dense_ones``: a 62 px icon-beside-name
    row on Home and a 164 px card with a three-line description on the
    category tabs. The user asked for one size — "all buttons, including
    the home screen, should look like the buttons in the other tabs" —
    and for the description to go, since the tooltip and the hint bar
    already carry it."""
    from spacr.qt.preferences import scaled_px
    assert home._tabs.tabText(1).startswith("Core")
    core = home._tabs.widget(1).findChildren(AppTile)
    assert len(core) == len(section_members("Core")) == 9
    for index in range(home._tabs.count()):
        for tile in home._tabs.widget(index).findChildren(AppTile):
            assert tile.sizeHint().height() >= scaled_px(HomePage.TILE_H)
            assert tile.sizeHint().width() >= scaled_px(HomePage.TILE_MIN_W)
            # Icon + name, and nothing else. Two labels: the pixmap and
            # the name. A third would be the description coming back.
            labels = tile.findChildren(QLabel)
            assert len(labels) == 2, (
                f"{tile.text_label} draws {[l.text() for l in labels]}")
            assert sum(1 for lbl in labels if lbl.pixmap()) == 1
            assert tile.name_label.full_text() == tile.text_label


def test_the_dense_and_tall_tiles_are_gone():
    """Both old tile classes, named so that re-adding one is deliberate."""
    from spacr.qt.widgets import home as home_mod
    assert not hasattr(home_mod, "DenseTile")
    assert not hasattr(home_mod, "TallTile")
    # `elide_to_lines` went with them: it shortened the tile blurb, and
    # there is no fixed-height wrapped label left on the page.
    assert not hasattr(home_mod, "elide_to_lines")


def test_no_tile_name_is_clipped_on_any_tab(home, qtbot, qapp):
    clipped = []
    for index in range(home._tabs.count()):
        home._tabs.setCurrentIndex(index)
        qapp.processEvents()
        page = home._tabs.widget(index)
        for tile in page.findChildren(AppTile):
            if tile.is_name_elided():
                clipped.append(tile.text_label)
    home._tabs.setCurrentIndex(0)
    assert not clipped, f"clipped tile names: {clipped}"


def test_the_aside_carries_recent_runs_system_and_news(home):
    headers = {lbl.text() for lbl in home.findChildren(QLabel)
               if lbl.objectName() == "HomePanelHeader"}
    assert "RECENT RUNS" in headers
    assert "SYSTEM" in headers
    assert any(h.startswith("NEWS") for h in headers)


def test_the_unfinished_aside_panels_say_so(home):
    """Only provisional News is marked ``(beta)``.

    Recent Runs and Totals became complete when every GUI and CLI pipeline
    began writing an automatic manifest.

    Lower case on purpose: the header is upper-cased and letter-spaced,
    so "(BETA)" would read as another word in the heading."""
    from spacr.qt.widgets.home import BETA_PANEL_TOOLTIP, BETA_SUFFIX, Panel

    marked, plain = {}, {}
    for panel in home.findChildren(Panel):
        (marked if panel.is_beta else plain)[panel.header.text()] = panel
    assert all(h.endswith(BETA_SUFFIX) for h in marked)
    assert not any(h.endswith(BETA_SUFFIX) for h in plain)
    assert BETA_SUFFIX == " (beta)" and BETA_SUFFIX.islower()

    assert {h.replace(BETA_SUFFIX, "").split(" ·")[0] for h in marked} == {
        "NEWS"}
    # MODULE STATE joined the unmarked set with #16j. It is not a panel
    # of numbers at all — it is the legend for the tile hover colours —
    # so there is nothing about it that could be provisional.
    assert set(plain) == {
        "QUEUED", "SYSTEM", "RECENT RUNS", "TOTALS", "MODULE STATE",
    }
    # The mark explains itself rather than just labelling.
    for panel in marked.values():
        assert panel.header.toolTip() == BETA_PANEL_TOOLTIP


def test_the_news_surface_is_the_reserved_slot(home):
    labels = [lbl.text() for lbl in home.findChildren(QLabel)]
    assert any("Reserved for featured" in lbl for lbl in labels)
    marker = QLabel("REPLACED")
    home.set_reserved_content(marker)
    assert home._reserved_content is marker
    assert "REPLACED" in [lbl.text() for lbl in home.findChildren(QLabel)]


def test_hovering_a_tile_explains_it_in_the_hint_bar(home):
    tile = next(t for t in home.findChildren(AppTile)
                if t.text_label == "Mask")
    desc = next(d for k, _n, d, _s in APPS if k == "mask")
    home.eventFilter(tile, QEvent(QEvent.Enter))
    assert home._hint_bar.text() == desc
    home.eventFilter(tile, QEvent(QEvent.Leave))
    assert home._hint_bar.text() != desc


# -- the queue segment -------------------------------------------------------

def _queue_at(tmp_path, monkeypatch, items):
    from spacr.qt import plate_queue as pq
    path = tmp_path / "queue.json"
    monkeypatch.setattr(pq, "_queue_path", lambda: path)
    queue = pq.PlateQueue()
    for label, key, status in items:
        item = pq.QueueItem.build(key, {"src": f"/data/{label}"}, label=label)
        item.status = status
        queue.add(item)
    return queue


def test_the_queue_segment_is_absent_when_the_queue_is_empty(
        home, tmp_path, monkeypatch):
    _queue_at(tmp_path, monkeypatch, [])
    home._queued.refresh()
    assert not home._queued.isVisible()


def test_the_queue_segment_appears_when_there_is_a_queue(
        home, tmp_path, monkeypatch, qapp):
    from spacr.qt.plate_queue import Status
    _queue_at(tmp_path, monkeypatch, [
        ("plate_08", "mask", Status.RUNNING),
        ("plate_09", "mask", Status.QUEUED),
    ])
    home._queued.refresh()
    qapp.processEvents()
    assert home._queued.isVisible()
    texts = [lbl.text() for lbl in home._queued.findChildren(QLabel)]
    assert "plate_08" in texts and "plate_09" in texts
    assert "running" in texts and "queued" in texts


def test_a_finished_queue_does_not_keep_the_segment_on_screen(
        home, tmp_path, monkeypatch):
    from spacr.qt.plate_queue import Status
    _queue_at(tmp_path, monkeypatch, [
        ("plate_01", "mask", Status.SUCCESS),
        ("plate_02", "mask", Status.FAILED),
    ])
    home._queued.refresh()
    assert not home._queued.isVisible()


# -- a running module --------------------------------------------------------

def _fake_run(app_key="mask", pausable=False):
    """A registered RunHandle without starting a thread.

    ``PipelineWorker`` is the real class — it is what carries the gate
    and the ``supports_pause`` answer, so faking it would fake the very
    thing under test.
    """
    def job(_settings):
        return None
    if pausable:
        bridge.pausable(job)
    setattr(job, bridge.APP_KEY_ATTR, app_key)
    worker = bridge.PipelineWorker(job, {})
    handle = bridge.RunHandle(app_key, worker, None,
                              parent=bridge.registry())
    bridge.registry().register(handle)
    return handle, worker


def test_nothing_running_means_no_banner(home):
    assert not home._banner.isVisible()


def test_a_running_module_is_reflected_on_home(home, qapp):
    handle, _worker = _fake_run("mask")
    qapp.processEvents()
    assert home._banner.isVisible()
    assert "Mask" in home._banner._title.text()
    assert "running" in home._banner._title.text()
    bridge.registry().unregister(handle)
    qapp.processEvents()
    assert not home._banner.isVisible()


def test_concurrent_runs_are_stacked_oldest_first_on_home(home, qapp):
    first, _ = _fake_run("mask")
    second, _ = _fake_run("measure")
    third, _ = _fake_run("classify")
    qapp.processEvents()

    visible = [banner for banner in home._banners if banner.isVisible()]
    assert len(visible) == 3
    assert ["Mask", "Measure", "Classify (CV)"] == [
        banner._title.text().split(" ·", 1)[0] for banner in visible
    ]

    bridge.registry().unregister(second)
    qapp.processEvents()
    visible = [banner for banner in home._banners if banner.isVisible()]
    assert ["Mask", "Classify (CV)"] == [
        banner._title.text().split(" ·", 1)[0] for banner in visible
    ]
    bridge.registry().unregister(first)
    bridge.registry().unregister(third)


def test_the_banner_shows_progress_scraped_from_the_pipeline(home, qapp):
    handle, worker = _fake_run("measure")
    worker.line_ready.emit(
        "Progress: 41/96, operation_type: measure, 2.1s/field\n")
    qapp.processEvents()
    assert handle.progress == (41, 96)
    assert abs(handle.fraction() - 41 / 96) < 1e-9
    home._banner.refresh()
    assert "41 of 96" in home._banner._sub.text()
    # …and does not then repeat the same line back at the user.
    assert home._banner._sub.text().count("41") == 1


def test_clicking_open_navigates_to_the_running_app(home, qtbot):
    _fake_run("classify")
    with qtbot.waitSignal(home.tile_clicked, timeout=1000) as blocker:
        home._banner._btn_open.click()
    assert blocker.args == ["classify"]


# ===========================================================================
# 3. Pause — the part that is not layout
# ===========================================================================

def test_the_gate_really_holds_a_thread_and_really_lets_it_go():
    """Not a flag that is written and never read: a thread is parked."""
    gate = bridge.PauseGate()
    reached = threading.Event()
    released = threading.Event()

    def body():
        reached.set()
        gate.wait_if_paused()
        released.set()

    gate.pause()
    assert gate.is_paused()
    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    assert reached.wait(5), "the thread never started"
    assert not released.wait(0.2), "the gate did not hold the thread"
    gate.resume()
    assert released.wait(5), "the gate did not release the thread"
    worker.join(5)
    assert not gate.is_paused()


def test_checkpoint_is_a_no_op_outside_a_worker():
    """Pipeline code must be able to call it from a plain script."""
    assert bridge.current_gate() is None
    bridge.checkpoint()          # must not raise, must not block


def test_a_job_that_polls_the_gate_can_genuinely_be_paused(qtbot):
    """The mechanism end to end, through a real PipelineWorker.

    Handshakes, not sleeps: the gate is closed *before* the thread
    starts, so the job must park at its very first checkpoint. The
    negative assertion is then unfalsifiable by a slow machine — a job
    that got past the gate would have advanced the counter, and it is
    still zero.
    """
    state = {"n": 0}
    running = threading.Event()
    finished = threading.Event()
    total = 25

    @bridge.pausable
    def job(_settings):
        running.set()                 # in fn, before the first checkpoint
        while state["n"] < total:
            bridge.checkpoint()
            state["n"] += 1
        finished.set()

    thread, worker = bridge.make_thread(job, {}, app_key="mask")
    try:
        assert worker.supports_pause
        worker.gate.pause()           # closed before the job can run
        thread.start()

        assert running.wait(10), "the job never reached its first checkpoint"
        assert not finished.wait(0.3), "a paused job ran to completion"
        assert state["n"] == 0, (
            f"a paused job did {state['n']} units of work past the gate")

        worker.gate.resume()
        assert finished.wait(10), "resume did not release the job"
        assert state["n"] == total
    finally:
        worker.gate.resume()
        if thread.isRunning():
            with qtbot.waitSignal(thread.finished, timeout=10000):
                pass


def test_a_paused_job_still_shows_as_paused_on_home(home, qapp):
    handle, _worker = _fake_run("mask", pausable=True)
    handle.gate.pause()
    home._banner.refresh()
    qapp.processEvents()
    assert "paused" in home._banner._sub.text()
    assert home._banner.pause_button.text() == "Resume"


def test_no_shipped_pipeline_claims_to_be_pausable():
    """The reason the control is disabled — asserted, not assumed.

    If a pipeline ever starts calling ``bridge.checkpoint()`` at a safe
    boundary, mark it ``bridge.pausable`` and this test is the one that
    tells you the Home control just came alive.
    """
    claimed = []
    for key, *_rest in APPS:
        entry = bridge.resolve_pipeline_entry(key)
        if entry is not None and getattr(entry, bridge.PAUSABLE_ATTR, False):
            claimed.append(key)
    assert not claimed, (
        f"{claimed} declare themselves pausable — make sure they really "
        "poll bridge.checkpoint() between units of work")


def test_the_pause_control_is_disabled_for_a_job_that_cannot_pause(
        home, qapp):
    _fake_run("mask", pausable=False)
    qapp.processEvents()
    button = home._banner.pause_button
    assert button.isVisible()
    assert not button.isEnabled()
    assert button.toolTip() == PAUSE_UNAVAILABLE
    assert "half-written" in button.accessibleDescription()


def test_the_pause_control_is_live_for_a_job_that_can_pause(home, qapp):
    handle, _worker = _fake_run("mask", pausable=True)
    qapp.processEvents()
    button = home._banner.pause_button
    assert button.isEnabled()
    assert button.text() == "Pause"

    button.click()
    assert handle.gate.is_paused()
    assert button.text() == "Resume"

    button.click()
    assert not handle.gate.is_paused()
    assert button.text() == "Pause"


def test_pressing_a_disabled_pause_cannot_pause_anything(home, qapp):
    """Belt and braces: the slot refuses even if something calls it."""
    handle, _worker = _fake_run("mask", pausable=False)
    qapp.processEvents()
    home._banner._on_pause()
    assert not handle.gate.is_paused()


def test_a_finished_worker_never_leaves_its_gate_latched():
    """A job that ends while paused must not strand the gate."""
    def job(_settings):
        return None

    worker = bridge.PipelineWorker(job, {})
    worker.gate.pause()
    worker.run()
    assert not worker.gate.is_paused()
    assert bridge.current_gate() is None


# ===========================================================================
# 4. The slide-in app list
# ===========================================================================

@pytest.fixture
def window(qtbot, qt_theme_applied, _empty_journal, monkeypatch):
    # MainWindow intentionally restores the user's persisted dock mode.  That
    # is application state, not test input: a developer who prefers a locked
    # or hidden dock must not change the outcome of the auto-drawer tests.
    monkeypatch.setattr("spacr.qt.preferences.get_dock_mode", lambda: "auto")
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1440, 900)
    win.show()
    qtbot.waitExposed(win)
    return win


def _hover(widget, enter=True):
    """Send a real enter/leave event to ``widget``."""
    if enter:
        pos = widget.rect().center()
        widget.enterEvent(QEnterEvent(pos, pos, widget.mapToGlobal(pos)))
    else:
        widget.leaveEvent(QEvent(QEvent.Leave))


def test_the_app_list_starts_hidden(window):
    drawer = window._app_drawer
    assert not drawer.is_open()
    assert not drawer.isVisible()
    # …and the sidebar is still the same object everything else names.
    assert drawer._panel is window._sidebar


def test_hovering_the_edge_strip_reveals_the_app_list(window, qtbot, qapp):
    drawer = window._app_drawer
    _hover(drawer._trigger)
    assert drawer._open_timer.isActive(), "the dwell timer did not arm"
    qtbot.waitUntil(drawer.is_open, timeout=2000)
    assert drawer.isVisible()
    names = {b.accessibleName() for b in drawer._panel.findChildren(QPushButton)}
    assert {n for _k, n, *_r in APPS} <= names


def test_a_pointer_merely_passing_the_edge_does_not_open_it(window, qapp):
    """The dwell delay is the whole reason a hot edge is usable: a
    pointer crossing on its way elsewhere must not summon the panel."""
    drawer = window._app_drawer
    _hover(drawer._trigger)
    assert drawer._open_timer.isActive()
    _hover(drawer._trigger, enter=False)         # left before it fired
    assert not drawer._open_timer.isActive()
    qapp.processEvents()
    assert not drawer.is_open()


def test_leaving_the_panel_closes_it_again(window, qtbot):
    drawer = window._app_drawer
    drawer.open()
    assert drawer.is_open()
    drawer._close_timer.setInterval(0)
    drawer.schedule_close()
    qtbot.waitUntil(lambda: not drawer.is_open(), timeout=2000)


def test_a_click_inside_the_panel_pins_it_against_the_close_timer(window):
    drawer = window._app_drawer
    drawer.open()
    drawer.hold(True)
    drawer.schedule_close()
    assert not drawer._close_timer.isActive(), "held drawer scheduled a close"
    assert drawer.is_open()


def test_the_app_list_is_reachable_without_a_mouse(window, qapp):
    """A reveal you can only hover is a reveal a keyboard user does not
    have — every app but the ones on the open tab would be unreachable."""
    drawer = window._app_drawer
    window.toggle_app_drawer()
    qapp.processEvents()
    assert drawer.is_open()
    assert drawer.is_held(), "keyboard open must pin, or focus races the close"
    focused = qapp.focusWidget()
    assert focused is not None
    assert drawer._panel.isAncestorOf(focused), (
        "focus did not land inside the app list")
    window.toggle_app_drawer()
    assert not drawer.is_open()


def test_escape_closes_the_drawer(window, qapp):
    from PySide6.QtGui import QKeyEvent
    drawer = window._app_drawer
    window.toggle_app_drawer()
    qapp.processEvents()
    assert drawer.is_open()
    drawer.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert not drawer.is_open()


def test_choosing_an_app_from_the_drawer_navigates_and_closes_it(
        window, qapp):
    drawer = window._app_drawer
    window.toggle_app_drawer()
    qapp.processEvents()
    button = next(b for b in window._sidebar.findChildren(QPushButton)
                  if b.property("navKey") == "measure")
    button.click()
    qapp.processEvents()
    assert window._status_app_label.text() == "Measure"
    assert not drawer.is_open()


def test_the_drawer_is_not_the_only_way_to_reach_every_app(window, qapp):
    """The reveal is now a convenience, not the only route: the Home tab
    lists all thirty apps, and so does the spaCR menu."""
    home_tab = window._startup._tabs.widget(0)
    assert {t.text_label for t in home_tab.findChildren(AppTile)} == {
        name for _k, name, *_r in APPS}
    menu_labels = set()
    for top in window.menuBar().actions():
        if top.text().replace("&", "") != "spaCR":
            continue
        for act in top.menu().actions():
            if not act.isSeparator():
                menu_labels.add(act.text())
        break
    assert {name for _k, name, *_r in APPS} <= menu_labels
    assert "All apps" in menu_labels


def test_the_sidebar_draws_an_ampersand_instead_of_a_mnemonic(window):
    """"Align & Stitch" rendered as "Align _Stitch" in the nav column."""
    button = next(b for b in window._sidebar.findChildren(QPushButton)
                  if b.property("navKey") == "align")
    assert "&&" in button.full_text()
    assert button.accessibleName() == "Align & Stitch"
