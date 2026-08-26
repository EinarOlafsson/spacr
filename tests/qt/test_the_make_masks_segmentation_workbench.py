"""Make Masks as the segmentation workbench: the modules folded into it.

Everything a person does to a segmentation is one loop — segment, look,
correct, train, segment again — so the modules that used to be tiles of
their own are buttons on this screen's masthead. Two things have to hold
for that to be a fold rather than a deletion:

* a folded module arrives whole, still pointed at the field this screen
  already has open, and still able to do what its tile could;
* its button goes on saying what the tile said, including the maturity
  colour it lit up in, after the registry row it read that from is gone.

The Curate fold additionally closes a hole: Curate paints, records, and
never wrote a pixel of its own, so its ledger asserted corrections beside
a mask the pipeline had produced.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.curation import is_curated
from spacr.mask_io import load_mask
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import (
    FOLD_FALLBACK,
    FOLD_ORDER,
    MASK_FOLDER_KEY,
    MakeMasksScreen,
    fold_description,
)


@pytest.fixture
def field_folder(tmp_path: Path) -> Path:
    """Three fields, one of which already has a two-object mask."""
    folder = tmp_path / "workbench"
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(3):
        imageio.imwrite(folder / f"f_{i:02d}.tif",
                        rng.integers(0, 4000, size=(48, 48), dtype=np.uint16))
    mask = np.zeros((48, 48), dtype=np.uint16)
    mask[4:14, 4:14] = 3
    mask[30:40, 30:40] = 9
    imageio.imwrite(folder / "masks" / "f_00.tif", mask)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A Make Masks screen whose folded windows are closed afterwards."""
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made.close_folded()


# ---------------------------------------------------------------------------
# The strip itself
# ---------------------------------------------------------------------------

def test_every_folded_module_is_an_icon_button_on_the_masthead(screen):
    """Each fold is its own icon, with the module's sentence as its tooltip.

    No text on the button: the icon is the module, exactly as it was on the
    tile the button replaced.
    """
    assert list(screen._folds.keys()) == list(FOLD_ORDER)
    for key in FOLD_ORDER:
        button = screen._folds.button_for(key)
        name, description, _stage = fold_description(key)
        assert not button.icon().isNull(), f"{key} has no icon"
        assert button.text() == ""
        assert description and description in button.toolTip()
        assert name in button.toolTip()


def test_the_fold_fallback_is_the_only_answer_for_every_fold(screen):
    """Every fold here has lost its row, so this table is all there is.

    It used to be checkable against the registry, because Napari Bridge
    still registered one and the two could be compared. That fold has since
    given its row up like the rest, and with nothing registered there is
    nothing to compare -- so what is asserted is the thing that now matters:
    each key has a COMPLETE entry, because a missing one is not an error
    anybody sees. The registry answers a key it no longer holds exactly as
    it answers a typo, so the button would simply show a bare title and
    light stable-blue for a module assessed as something else.
    """
    from spacr.qt import app as app_module

    registered = {row[0] for row in app_module.APPS}
    still_a_tile = sorted(set(FOLD_ORDER) & registered)
    assert still_a_tile == [], (
        f"these are folded AND registered, so one of the two is wrong: "
        f"{still_a_tile}")

    for key in FOLD_ORDER:
        assert key in FOLD_FALLBACK, (
            f"{key} is folded onto this screen and kept no record; nothing "
            f"can answer for it")
        name, description, stage = FOLD_FALLBACK[key]
        assert name and description and stage, (
            f"{key} has no row and an incomplete kept description; its "
            f"button would show a bare title and light stable-blue")
        assert stage in {"alpha", "beta", "stable"}, f"{key}: {stage!r}"


def test_a_folded_record_reaches_the_button(screen):
    """The record is only worth keeping if the button actually reads it."""
    from spacr.qt.widgets.fold_strip import folded_fallback

    for key, (name, description, stage) in FOLD_FALLBACK.items():
        got_name, got_description, got_stage = folded_fallback(key)
        assert got_name == name, key
        assert got_description == description, key
        assert got_stage == stage, key


def test_a_folded_button_keeps_its_stage_after_the_row_is_dropped(
        qtbot, qt_theme_applied, monkeypatch):
    """Folding a module ends in dropping its registry row; the colour stays.

    The strip reads a button's hover colour from the app registry, which
    answers "stable" for a key it no longer holds — so an alpha module's
    button would light blue where its tile lit green-cyan.
    """
    from spacr.qt import app as app_module

    monkeypatch.setattr(
        app_module, "APPS",
        [row for row in app_module.APPS if row[0] not in FOLD_ORDER])
    monkeypatch.setattr(
        app_module, "APP_STAGE",
        {key: value for key, value in app_module.APP_STAGE.items()
         if key not in FOLD_ORDER})

    made = MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        for key, (name, description, stage) in FOLD_FALLBACK.items():
            button = made._folds.button_for(key)
            assert button.property("stage") == stage, key
            assert description in button.toolTip(), key
            assert name in button.toolTip(), key
    finally:
        made.close_folded()


# ---------------------------------------------------------------------------
# Opening a fold
# ---------------------------------------------------------------------------

def test_the_two_cellpose_halves_share_one_workbench(screen):
    """Train and Mask-the-folder are two tabs of one screen, not two copies.

    A checkpoint trained on one tab is what the other segments with, and a
    second copy of the workbench would not have it.
    """
    assert (screen.folded_screen("train_cellpose")
            is screen.folded_screen(MASK_FOLDER_KEY))


def test_opening_a_fold_twice_raises_the_page_it_already_made(
        screen, field_folder: Path):
    """The second press finds the paths and results the first one left."""
    screen._open_folder(str(field_folder))
    first = screen.open_folded("model_compare")
    second = screen.open_folded("model_compare")
    assert first is second
    assert first.screen is screen.folded_screen("model_compare")


def test_a_folded_module_is_a_page_beside_the_editor_not_a_window(
        screen, field_folder: Path):
    """A window is the last resort for a fold, and this host needs none.

    The editor keeps the first page and cannot be closed off the strip;
    the module arrives beside it as the whole screen it was.
    """
    screen._open_folder(str(field_folder))
    panel = screen.open_folded("curate")
    pages = screen._fold_pages

    assert not panel.isWindow()
    assert pages.tabText(0) == mm.HEADER_TITLE
    assert pages.tabText(pages.indexOf(panel)) == "Curate"
    assert pages.currentWidget() is panel

    pages.tabCloseRequested.emit(pages.indexOf(panel))
    assert pages.indexOf(panel) < 0
    assert screen.open_folded("curate") is panel
    assert pages.indexOf(panel) > 0


def test_the_editor_page_cannot_be_closed_off_the_strip(
        screen, field_folder: Path):
    """There is nothing behind the host's own page to fall back to."""
    screen._open_folder(str(field_folder))
    screen.open_folded("curate")
    pages = screen._fold_pages

    pages.tabCloseRequested.emit(0)

    assert pages.tabText(0) == mm.HEADER_TITLE


def test_a_key_this_screen_does_not_fold_opens_nothing(screen):
    """An unknown key answers None rather than building a stray window."""
    assert screen.folded_screen("regression") is None
    assert screen.open_folded("regression") is None


def test_a_folded_module_is_pointed_at_the_open_folder(
        qtbot, screen, field_folder: Path):
    """The folder is already chosen here; a fold must not ask for it again."""
    screen._open_folder(str(field_folder))
    assert screen.seed_folded("model_compare") == {"folder": str(field_folder)}
    compare = screen.folded_screen("model_compare")
    qtbot.waitUntil(lambda: compare.source_folder() == str(field_folder),
                    timeout=10000)
    assert len(compare.field_names()) == 3


def test_a_fold_opened_with_no_folder_seeds_nothing(screen):
    """Not a failure: the module opens on its own picker, as its tile did."""
    assert screen.seed_folded("model_compare") == {}


def test_the_mask_editors_are_handed_the_field_on_screen(
        screen, field_folder: Path):
    """Curate gets the mask, the napari bridge gets the mask and the image."""
    screen._open_folder(str(field_folder))
    mask_path = str(field_folder / "masks" / "f_00.tif")

    curate = screen.seed_folded("curate")
    assert curate == {"mask": mask_path}
    assert screen.folded_screen("curate")._mask_edit.text() == mask_path

    bridge = screen.seed_folded("napari_bridge")
    assert bridge == {"mask": mask_path,
                      "image": str(field_folder / "f_00.tif")}
    assert (screen.folded_screen("napari_bridge")._image_edit.text()
            == str(field_folder / "f_00.tif"))


def test_a_module_with_no_screen_of_its_own_gets_its_settings_page(
        screen, field_folder: Path, monkeypatch):
    """A settings-only fold arrives as the page its tile opened, src filled.

    Driven through a monkeypatched fold list because every module this
    screen folds today reaches a screen class of its own. The generic
    settings page is the shape the next settings-only fold takes, and this
    is what says it still works.
    """
    monkeypatch.setattr(mm, "FOLD_ORDER",
                        tuple(FOLD_ORDER) + ("cellpose_masks",))
    screen._open_folder(str(field_folder))
    assert screen.seed_folded("cellpose_masks") == {"src": str(field_folder)}
    page = screen.folded_screen("cellpose_masks")
    assert page.app_key == "cellpose_masks"
    assert page._settings_model.collect().get("src") == str(field_folder)


def test_the_series_modules_do_not_fold_into_hand_curation(screen):
    """Timelapse and Motility are not on this masthead, and cannot be.

    Rewritten from a test that folded Timelapse in here. They fold into
    MASK GENERATION, whose settings they overlap: this screen corrects
    masks that already exist, while tracking a series and measuring how it
    moved are things mask generation does over one. Asserted on the strip
    the user sees as well as on the list, because a key removed from the
    list and left wired to a button is the failure this guards.
    """
    for key in ("timelapse", "motility"):
        assert key not in FOLD_ORDER
        assert key not in mm.FOLD_FALLBACK
        assert screen._folds.button_for(key) is None
        assert screen.folded_screen(key) is None
        assert screen.open_folded(key) is None


def test_the_zoo_hands_its_comparison_to_the_folded_model_compare(
        qtbot, screen, field_folder: Path):
    """Folded, this screen is the zoo's host, so it is what opens Compare.

    The hand-off is wired by whoever hosts the zoo; unwired, the zoo's
    "compare these two" button selects two models and opens nothing.
    """
    screen._open_folder(str(field_folder))
    zoo = screen.folded_screen("model_zoo")
    zoo.compare_requested.emit({"model_a": "cpsam", "model_b": "cyto3",
                                "folder": str(field_folder), "n_fields": 2})
    compare = screen.folded_screen("model_compare")
    assert "model_compare" in screen._fold_dialogs
    qtbot.waitUntil(lambda: compare.source_folder() == str(field_folder),
                    timeout=10000)
    assert compare.model_configs()[0].model == "cpsam"
    assert compare.model_configs()[1].model == "cyto3"


# ---------------------------------------------------------------------------
# Mask the whole folder
# ---------------------------------------------------------------------------

def test_masking_the_whole_folder_runs_the_applying_half_on_that_folder(
        screen, field_folder: Path, monkeypatch):
    """The button starts a real run, on the folder already open here."""
    started = []
    monkeypatch.setattr(MakeMasksScreen, "_start_folded_run",
                        staticmethod(lambda page: started.append(page.app_key)))
    monkeypatch.setattr(screen, "_confirm", lambda *args: True)
    screen._open_folder(str(field_folder))

    assert screen.mask_whole_folder() is True

    assert started == ["cellpose_masks"]
    workbench = screen.folded_screen(MASK_FOLDER_KEY)
    assert (workbench.apply_screen._settings_model.collect().get("src")
            == str(field_folder))
    assert str(field_folder) in screen._status_label.text()


def test_masking_the_whole_folder_leaves_the_training_path_alone(
        screen, field_folder: Path, monkeypatch):
    """Training reads <src>/train/images; a folder of fields is not that.

    Seeding both halves from one folder would point training at a layout
    that is not there, which is the one thing a fold must not guess at.
    """
    monkeypatch.setattr(MakeMasksScreen, "_start_folded_run",
                        staticmethod(lambda page: None))
    monkeypatch.setattr(screen, "_confirm", lambda *args: True)
    screen._open_folder(str(field_folder))
    screen.mask_whole_folder()

    workbench = screen.folded_screen(MASK_FOLDER_KEY)
    assert not workbench.train_screen._settings_model.collect().get("src")


def test_masking_the_whole_folder_needs_a_folder(screen, monkeypatch):
    """With nothing open there is nothing to mask, and it says so."""
    monkeypatch.setattr(
        MakeMasksScreen, "_start_folded_run",
        staticmethod(lambda page: pytest.fail("started a run with no folder")))

    assert screen.mask_whole_folder() is False
    assert "Open a folder" in screen._status_label.text()


def test_a_declined_confirmation_masks_nothing(
        screen, field_folder: Path, monkeypatch):
    """Segmenting a whole folder is minutes of GPU; "no" has to mean no."""
    monkeypatch.setattr(
        MakeMasksScreen, "_start_folded_run",
        staticmethod(lambda page: pytest.fail("ran a declined job")))
    monkeypatch.setattr(screen, "_confirm", lambda *args: False)
    screen._open_folder(str(field_folder))

    assert screen.mask_whole_folder() is False


# ---------------------------------------------------------------------------
# Curate writes a mask
# ---------------------------------------------------------------------------

def test_curate_writes_the_mask_it_painted(screen, field_folder: Path):
    """The corrections reach the pixels, and the ledger goes with them.

    On its own Curate wrote the ledger and nothing else, so
    :func:`spacr.curation.is_curated` reported a file the pipeline had made
    as hand-corrected. The write arrives with the fold, because this is the
    screen that writes masks.
    """
    screen._open_folder(str(field_folder))
    dialog = screen.open_folded("curate")
    assert "Save mask" in dialog.actions

    panel = dialog.screen.open_mask()
    assert panel is not None
    panel.session.label = 7
    painted = panel.session.paint({"y": 24.0, "x": 24.0}, radius=3.0)
    assert painted > 0

    mask_path = str(field_folder / "masks" / "f_00.tif")
    assert not is_curated(mask_path)

    assert screen.save_curated_mask() == mask_path

    on_disk = load_mask(mask_path)
    assert int((on_disk == 7).sum()) == painted
    assert int((on_disk == 3).sum()) == 100
    assert is_curated(mask_path)


def test_saving_from_curate_with_no_mask_open_says_so(screen):
    """Nothing is written, and the status line explains why."""
    screen.open_folded("curate")

    assert screen.save_curated_mask() == ""
    assert "Open a mask in Curate" in screen._status_label.text()


def test_a_curate_save_that_fails_is_reported_not_swallowed(
        screen, field_folder: Path, monkeypatch):
    """A read-only masks folder must not look like a successful save."""
    screen._open_folder(str(field_folder))
    dialog = screen.open_folded("curate")
    panel = dialog.screen.open_mask()
    panel.session.label = 7
    panel.session.paint({"y": 24.0, "x": 24.0}, radius=3.0)

    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(panel.session, "save_mask", refuse)

    assert screen.save_curated_mask() == ""
    assert "read-only file system" in screen._status_label.text()


def test_closing_the_screen_closes_the_modules_it_opened(
        qtbot, qt_theme_applied, field_folder: Path):
    """A folded module polls the GPU on a worker while it is alive.

    Qt aborts the process when a running QThread is destroyed, so leaving a
    folded module running behind a closed host is not a tidiness question.

    Rewritten twice: the module opened is Curate rather than Timelapse,
    which no longer folds in here, and it is asserted through the page it
    now occupies rather than through a window's visibility -- a page is
    only visible when its host is on screen, and this host never is.
    """
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    made._open_folder(str(field_folder))
    panel = made.open_folded("curate")
    pages = made._fold_pages
    assert pages.currentWidget() is panel

    made.close()

    assert not panel.isVisible()
    assert not made.folded_screen("curate").isVisible()


def test_closing_the_screen_closes_a_module_that_was_never_opened(
        qtbot, qt_theme_applied, field_folder: Path):
    """Pointing a module at the folder builds it, and building starts work.

    Model Compare loads the folder on a worker thread the moment it is
    given one, which happens in `seed_folded` -- before any button has
    been pressed and before it has a page. Walking only the panels left
    that thread running with nothing holding it, and the process died of
    it in whatever ran next.
    """
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    made._open_folder(str(field_folder))
    made.seed_folded("model_compare")
    compare = made.folded_screen("model_compare")
    assert compare is not None
    assert "model_compare" not in made._fold_dialogs

    made.close_folded()

    assert not compare.isVisible()
    assert all(not thread.isRunning()
               for thread, _worker in list(compare._jobs))


def test_fold_description_falls_back_only_when_the_row_is_gone(monkeypatch):
    """The registry wins while it has the row; the kept copy answers after.

    Proved by GIVING the registry an answer that differs from the kept one: if
    the fallback were preferred, the row would not show through. The row is put
    back rather than edited in place, because Model Zoo has none any more —
    which is precisely the state the second half of this test describes.
    """
    from spacr.qt import app as app_module

    restored = list(app_module.APPS) + [
        ("model_zoo", "Zoological Gardens", "a different line", "Models")]
    monkeypatch.setattr(app_module, "APPS", restored)
    monkeypatch.setitem(app_module.APP_STAGE, "model_zoo", "beta")

    name, description, stage = fold_description("model_zoo")
    assert name == "Zoological Gardens"
    assert description == "a different line"
    assert stage == "beta"

    monkeypatch.setattr(
        app_module, "APPS",
        [row for row in restored if row[0] != "model_zoo"])
    assert fold_description("model_zoo") == FOLD_FALLBACK["model_zoo"]


def test_the_masthead_carries_the_module_name_and_its_api_link(screen):
    """The strip sits on the module's own masthead, not on a bare title."""
    assert screen._header.title_label.text() == mm.HEADER_TITLE
    assert screen._header.description_label.text() == mm.HEADER_DESCRIPTION
    assert screen._header.info_link is not None
    assert screen._folds.parent() is screen._header


# ---------------------------------------------------------------------------
# The rows are dropped
# ---------------------------------------------------------------------------
# Folding a module ends in it losing its registry row, which is what takes the
# tile off Home. The row is load-bearing, though: `spacr-run`, pre-flight, the
# settings form, the drop handlers and the fold button's own colour all key off
# it. What follows is the contract each of those has to go on honouring for the
# four modules this screen absorbed, with the row gone.

#: The registry rows this screen's folds replaced. `cellpose_all` (the
#: "Mask the whole folder" button) and `napari_bridge` are not here: the
#: first never had a row, and the second is not part of this drop.
DROPPED_HERE = ("train_cellpose", "model_compare", "model_zoo", "curate")


def test_the_rows_this_screen_folded_have_left_the_registry():
    """The tile is gone from Home, which is the whole point of the fold."""
    import spacr.qt
    from spacr.qt import app as app_module

    spacr.qt.register_self_registering_modules()
    live = {row[0] for row in app_module.APPS}
    still_there = sorted(key for key in DROPPED_HERE if key in live)
    assert not still_there, (
        f"these modules are buttons on Make Masks and tiles on Home at the "
        f"same time: {still_there}")


def test_every_module_this_screen_folded_is_still_reachable_from_it(screen):
    """A fold that cannot be opened is a deletion wearing a button.

    Pressed rather than called: the strip is the only way in once the row is
    gone, so the assertion has to go through the button a hand would find.
    """
    for key in DROPPED_HERE:
        button = screen._folds.button_for(key)
        assert button is not None, f"{key} has no button on the masthead"
        button.click()
        assert screen.folded_screen(key) is not None, (
            f"{key}'s button opened nothing")


def test_the_gui_only_sentence_is_written_in_cli_not_borrowed_from_a_row():
    """`spacr-run curate` has to explain itself on a machine with no PySide6.

    ``cli._absorb_registered_gui_only`` lifts the sentence out of a registered
    app's ``cli_note=``, so while the row existed the table answered whether
    or not anybody had written the sentence down. The row is gone and that
    source with it — and the process that needs the answer most, a cluster
    node with no Qt registry loaded at all, never had it. Read out of the
    module's own literal, because that is the only copy left.
    """
    import ast
    import pathlib

    import spacr.cli as cli_module

    source = pathlib.Path(cli_module.__file__).read_text()
    written = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = getattr(node.target, "id", "")
        if target != "INTERACTIVE_ONLY" or not isinstance(node.value, ast.Dict):
            continue
        written = {key.value for key in node.value.keys
                   if isinstance(key, ast.Constant)}
    assert written, "cli.INTERACTIVE_ONLY is no longer a literal dict"
    for key in ("model_compare", "model_zoo", "curate"):
        assert key in written, (
            f"{key} has no GUI-only sentence in cli.py; with its registry row "
            f"dropped, `spacr-run {key}` answers 'unknown module' instead")
        assert key in cli_module.INTERACTIVE_ONLY
        assert "unknown module" not in cli_module._unknown_module_message(key)


def test_the_training_half_still_runs_headless_with_no_row():
    """`spacr-run train_cellpose` and pre-flight both key off the dropped name.

    Unlike the other three this one is not GUI-only: it has an entry point, a
    settings vocabulary and a validation rule set, and a cluster job written
    before the fold names it. All four registries have to keep answering.
    """
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.qt.screens.settings_model import resolve_default_settings
    from spacr.validate import APP_FUNCTIONS
    import spacr.cli as cli_module

    assert cli_module.MODULES["train_cellpose"].func_name == "train_cellpose"
    assert APP_FUNCTIONS["train_cellpose"] == "spacr.submodules.train_cellpose"
    entry = resolve_pipeline_entry("train_cellpose")
    assert entry is not None, (
        "the bridge lost train_cellpose's entry point with its registry row; "
        "the Run button on the folded page does nothing")
    inner = getattr(entry, "__wrapped__", entry)
    assert inner.__name__ == "train_cellpose"
    defaults = resolve_default_settings("train_cellpose")
    assert {"n_epochs", "learning_rate"} <= set(defaults), (
        f"the training form collapsed to {sorted(defaults)}")


def test_a_settings_file_written_before_the_fold_still_fills_the_form(
        screen, tmp_path, qtbot):
    """A CSV names the module by key, and the key outlived the tile.

    Round-tripped through the folded workbench rather than asserted against
    the defaults table, because what a user has on disk is a file the training
    page has to read.
    """
    workbench = screen.folded_screen("train_cellpose")
    train = workbench.train_screen
    assert train.app_key == "train_cellpose"
    before = train._settings_model.collect()
    assert {"n_epochs", "learning_rate", "model_type", "batch_size"} <= set(
        before), (
        f"the training form collapsed to {sorted(before)}; a settings file "
        f"written before the fold names keys it no longer has a control for")

    applied = train.apply_settings_dict({"n_epochs": 42, "learning_rate": 0.05})
    assert applied == 2
    collected = train._settings_model.collect()
    assert int(collected["n_epochs"]) == 42
    assert float(collected["learning_rate"]) == pytest.approx(0.05)


def test_a_dropped_file_still_finds_a_handler_for_every_fold():
    """The drop handlers are keyed by app key and never consult the registry."""
    from spacr.qt.dnd_handlers import get_handler

    for key in DROPPED_HERE:
        assert get_handler(key) is not None, (
            f"a file dropped on {key}'s page lands nowhere")


def test_every_folded_button_lights_the_colour_its_tile_lit(screen):
    """The stage the tile was showing the day its row went, and no other.

    ``app_stage`` answers "stable" for a key the registry no longer holds, so
    with nothing kept an alpha module's button would light blue. The kept
    table is the only source left, and this asserts each button actually wears
    what it says.
    """
    from spacr.qt.theme import STAGE_HOVER

    for key in FOLD_ORDER:
        _name, _description, stage = fold_description(key)
        assert stage in STAGE_HOVER, f"{key} lights in no known colour"
        assert screen._folds.button_for(key).property("stage") == stage, key
    assert fold_description("model_compare")[2] == "stable"
    assert fold_description("curate")[2] == "alpha"
