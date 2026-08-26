"""Cellpose Masks is a tab of the Cellpose workbench, not a registry row.

"Train Cellpose — Train custom Cellpose models" and "Cellpose Masks —
Cellpose mask generation" used to sit two rows apart under a third called
"Mask — Generate cellpose masks for cells, nuclei and pathogens", and
nothing in any of the three lines told a user which to open. The two
Cellpose rows are one loop: fine-tune a model on a handful of labelled
fields, run it over the rest of the folder, look at what came out, label
the failures, train again.

What this file pins is the part of the merge that is easy to get wrong:

* the two ``src`` fields stay separate, because ``src`` names a folder of
  image/mask PAIRS on one half and a folder of plain images on the other;
* the segmentation knobs cross, and are read from the propagation map
  :mod:`spacr.qt.preview_registry` already declares rather than from a
  second list that could disagree with it;
* the model crosses as the checkpoint that was actually written, never as
  the ``model_name`` string — which means "the model to save as" on one
  half and "the model to segment with" on the other;
* and both module keys still run headless, because merging two screens may
  not merge two pipelines.
"""
from __future__ import annotations

import os

import pytest

from spacr.qt.screens.train_cellpose import (
    APPLY_KEY,
    TRAIN_KEY,
    WORKBENCH_INTRO,
    WORKBENCH_TITLE,
    CellposeWorkbenchScreen,
    carried_setting_keys,
)


@pytest.fixture
def train_src_offered(monkeypatch):
    """Give the training half the ``src`` its entry point reads.

    ``spacr.submodules.train_cellpose`` opens ``<src>/train/images``, and
    ``spacr.cli`` documents ``src`` as required, but
    ``get_train_cellpose_default_settings`` does not offer the key — so the
    generated form has no path field for it. The tests below are about how
    the two path fields behave once both exist, which is a property of the
    screen rather than of that defaults helper, so the helper is completed
    here instead of the screen being written around the gap.
    """
    import spacr.settings as settings_module

    original = settings_module.get_train_cellpose_default_settings

    def with_src(settings):
        filled = original(settings)
        filled.setdefault("src", "path")
        return filled

    monkeypatch.setattr(
        settings_module, "get_train_cellpose_default_settings", with_src)
    return with_src


@pytest.fixture
def workbench(qtbot):
    """The merged screen, with both module pages live."""
    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    return screen


def _set(screen, key, value):
    """Write one setting into a module page's form."""
    assert screen._settings_model._widgets.get(key) is not None, (
        f"{screen.app_key} has no {key} field to write to")
    screen.apply_settings_dict({key: value})


def _read(screen, key):
    """Read one setting back out of a module page's form."""
    return screen._settings_model.collect().get(key)


# ---------------------------------------------------------------------------
# One row, both halves
# ---------------------------------------------------------------------------

def test_one_module_page_carries_both_halves_of_the_loop(workbench):
    """Two tabs, running the two modules, training first."""
    assert workbench._tabs.count() == 2
    assert workbench.train_screen.app_key == TRAIN_KEY
    assert workbench.apply_screen.app_key == APPLY_KEY
    assert workbench.screen_for(APPLY_KEY) is workbench.apply_screen
    assert workbench._tabs.currentIndex() == 0
    assert workbench.active_app_key() == TRAIN_KEY
    labels = [workbench._tabs.tabText(i) for i in range(2)]
    assert all(labels) and labels[0] != labels[1]
    # The page answers to the row it was opened under, whichever tab shows.
    workbench._tabs.setCurrentIndex(1)
    assert workbench.app_key == TRAIN_KEY
    assert workbench.active_app_key() == APPLY_KEY


def test_the_screen_is_named_for_the_loop_not_for_the_training_half():
    """Somebody after "segment this folder with a stock model" has to find it.

    Named for the training half, the applying half becomes unreachable by
    search: it no longer has a row of its own to be found under.
    """
    assert "Train" not in WORKBENCH_TITLE
    assert "Cellpose" in WORKBENCH_TITLE
    intro = WORKBENCH_INTRO.lower()
    assert "segment" in intro and "stock" in intro
    assert "fine-tune" in intro


def test_the_line_under_the_title_says_what_src_means_on_this_tab(workbench):
    """One box meaning two folders is what the second tab exists to prevent."""
    training = workbench._header.instruction_label.text()
    assert "train/images" in training and "train/masks" in training
    workbench._tabs.setCurrentIndex(1)
    applying = workbench._header.instruction_label.text()
    assert applying != training
    assert ".tif" in applying and "masks" in applying


# ---------------------------------------------------------------------------
# The two path fields
# ---------------------------------------------------------------------------

def test_each_half_keeps_its_own_path_field(train_src_offered, qtbot):
    """Two fields, two widgets, two values — the point of two tabs."""
    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    train_field = screen.train_screen._settings_model._widgets.get("src")
    apply_field = screen.apply_screen._settings_model._widgets.get("src")
    assert train_field is not None and apply_field is not None
    assert train_field is not apply_field

    _set(screen.train_screen, "src", "/data/labelled")
    _set(screen.apply_screen, "src", "/data/plate1")
    screen._tabs.setCurrentIndex(1)
    screen._tabs.setCurrentIndex(0)
    screen._tabs.setCurrentIndex(1)
    assert _read(screen.train_screen, "src") == "/data/labelled"
    assert _read(screen.apply_screen, "src") == "/data/plate1"


def test_the_path_is_never_one_of_the_carried_settings():
    """The one key that must not cross, named as such rather than by luck."""
    assert "src" not in carried_setting_keys()
    assert "model_name" not in carried_setting_keys()


# ---------------------------------------------------------------------------
# The knobs that do cross
# ---------------------------------------------------------------------------

def test_the_carried_knobs_are_read_from_the_preview_propagation_map():
    """A second copy of the list is a copy that can disagree with the map."""
    from spacr.qt import preview_registry

    declared = set(preview_registry.PREVIEWS[APPLY_KEY].propagation.values())
    assert set(carried_setting_keys()) == declared - {"model_name", "src"}
    assert "diameter" in carried_setting_keys()


def test_extending_the_propagation_map_extends_what_is_carried(monkeypatch):
    """Read, not copied: a knob added to the map is carried without a change
    here."""
    from dataclasses import replace

    from spacr.qt import preview_registry

    spec = preview_registry.PREVIEWS[APPLY_KEY]
    grown = dict(spec.propagation)
    grown["cell_background"] = "background"
    monkeypatch.setitem(preview_registry.PREVIEWS, APPLY_KEY,
                        replace(spec, propagation=grown))
    assert "background" in carried_setting_keys()


def test_a_diameter_chosen_for_training_is_the_one_the_masks_are_made_at(
        workbench):
    """The loop is one run's settings, so the shared knobs follow the eye."""
    _set(workbench.train_screen, "diameter", 55)
    workbench._tabs.setCurrentIndex(1)
    assert _read(workbench.apply_screen, "diameter") == 55

    _set(workbench.apply_screen, "diameter", 80)
    workbench._tabs.setCurrentIndex(0)
    assert _read(workbench.train_screen, "diameter") == 80


def test_a_knob_the_other_half_does_not_have_is_not_smuggled_into_it(
        workbench):
    """Training has no flow threshold; writing one into it would hide a
    setting in a form that never shows it again."""
    _set(workbench.apply_screen, "flow_threshold", 0.9)
    workbench._tabs.setCurrentIndex(1)
    workbench._tabs.setCurrentIndex(0)
    collected = workbench.train_screen._settings_model.collect()
    assert "flow_threshold" not in collected


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

#: The name the training form starts on, and the only one it can currently
#: be set to: ``model_name`` is drawn as a closed dropdown of stock Cellpose
#: models, so the tests name the checkpoint the way a real run would.
TRAINED_MODEL = "new_model"


def _write_checkpoint(root, name, *, suffix=""):
    """Put a checkpoint where ``train_cellpose`` would have left one."""
    folder = os.path.join(root, "models", "cellpose_model", "models")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_cpsam_e2_X1000_Y1000{suffix}.CP_model")
    with open(path, "wb") as handle:
        handle.write(b"weights")
    return path


def test_the_model_crosses_as_the_checkpoint_that_was_actually_written(
        train_src_offered, qtbot, tmp_path):
    """The Apply half segments with what the Train half produced."""
    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    expected = _write_checkpoint(str(tmp_path), TRAINED_MODEL)
    _set(screen.train_screen, "src", str(tmp_path))
    assert _read(screen.train_screen, "model_name") == TRAINED_MODEL

    screen._tabs.setCurrentIndex(1)
    assert _read(screen.apply_screen, "custom_model") == expected
    assert not screen._carry_note.isHidden()
    assert os.path.basename(expected) in screen._carry_note.text()


def test_a_model_name_string_is_never_written_over_the_apply_model(workbench):
    """``model_name`` means two different things, so the string may not cross.

    Copied across, the training half's default would point the applying half
    at a stock Cellpose model called ``new_model``, which does not exist —
    and "segment this folder with cpsam" would stop working for everyone who
    has never trained anything.
    """
    before = _read(workbench.apply_screen, "model_name")
    assert _read(workbench.train_screen, "model_name") != before
    workbench._tabs.setCurrentIndex(1)
    assert _read(workbench.apply_screen, "model_name") == before
    assert _read(workbench.apply_screen, "custom_model") in (None, "")
    assert workbench._carry_note.isHidden()


def test_a_finished_run_wins_over_the_snapshots_it_made_on_the_way(
        train_src_offered, qtbot, tmp_path):
    """Cellpose saves periodically during training under an ``_epoch_`` name;
    the run's own final save is the model, even when it is older."""
    final = _write_checkpoint(str(tmp_path), TRAINED_MODEL)
    snapshot = _write_checkpoint(str(tmp_path), TRAINED_MODEL,
                                 suffix="_epoch_0100")
    os.utime(final, (1, 1))
    os.utime(snapshot, (10_000, 10_000))

    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    _set(screen.train_screen, "src", str(tmp_path))
    assert screen.trained_checkpoint() == final


def test_another_models_checkpoint_is_not_picked_up(
        train_src_offered, qtbot, tmp_path):
    """Only the model this run is named for; the folder holds every run's."""
    _write_checkpoint(str(tmp_path), "somebody_elses")
    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    _set(screen.train_screen, "src", str(tmp_path))
    assert _read(screen.train_screen, "model_name") == TRAINED_MODEL
    assert screen.trained_checkpoint() == ""


# ---------------------------------------------------------------------------
# Settings arriving from elsewhere
# ---------------------------------------------------------------------------

def test_settings_land_on_the_half_they_belong_to(workbench):
    """A restored session or a recipe belongs to one module, not to both."""
    workbench._tabs.setCurrentIndex(1)
    assert workbench.apply_settings_dict({"n_epochs": 3}) >= 1
    assert workbench._tabs.currentIndex() == 0
    assert _read(workbench.train_screen, "n_epochs") == 3

    assert workbench.apply_settings_dict({"flow_threshold": 0.75}) >= 1
    assert workbench._tabs.currentIndex() == 1
    assert _read(workbench.apply_screen, "flow_threshold") == 0.75


def test_a_dict_that_names_neither_half_stays_on_the_tab_in_front_of_you(
        workbench):
    """``diameter`` is both modules'; guessing would move the page for
    nothing."""
    workbench._tabs.setCurrentIndex(1)
    workbench.apply_settings_dict({"diameter": 42})
    assert workbench._tabs.currentIndex() == 1
    assert _read(workbench.apply_screen, "diameter") == 42


# ---------------------------------------------------------------------------
# What the merge may not break
# ---------------------------------------------------------------------------

def test_a_failure_is_reported_under_the_key_of_the_half_that_failed(
        workbench, qtbot):
    """One page, two modules: the AI console must be asked about the right
    one."""
    seen = []
    workbench.error_explain_requested.connect(
        lambda text, key: seen.append((text, key)))
    workbench.apply_screen.error_explain_requested.emit("boom", APPLY_KEY)
    workbench.train_screen.error_explain_requested.emit("bang", TRAIN_KEY)
    assert seen == [("boom", APPLY_KEY), ("bang", TRAIN_KEY)]


def test_the_apply_half_keeps_the_live_preview_declared_for_it(workbench):
    """The preview is attached by the window's stack watcher, which never
    sees a tab — so the screen installs it, or Cellpose Masks silently loses
    the panel whose entire job is "did the mask come out right"."""
    assert getattr(workbench.apply_screen, "_registry_preview", None) is not None
    assert getattr(workbench.train_screen, "_registry_preview", None) is None
    assert getattr(workbench.apply_screen, "_settings_search", None) is not None
    assert getattr(workbench.train_screen, "_settings_search", None) is not None


def test_both_halves_still_run_headless_under_their_own_keys():
    """Merging two screens may not merge two pipelines: a cluster job and a
    settings CSV written for either module have to keep working."""
    from spacr import cli
    from spacr.validate import APP_FUNCTIONS

    assert cli.MODULES[TRAIN_KEY].func_name == "train_cellpose"
    assert cli.MODULES[APPLY_KEY].func_name == "identify_masks_finetune"
    assert APP_FUNCTIONS[TRAIN_KEY].endswith("train_cellpose")
    assert APP_FUNCTIONS[APPLY_KEY].endswith("identify_masks_finetune")


def test_each_tab_runs_the_entry_point_of_its_own_module():
    """The Run button on a tab is the module's, not the page's."""
    from spacr.qt.bridge import resolve_pipeline_entry

    def name(key):
        entry = resolve_pipeline_entry(key)
        return getattr(getattr(entry, "__wrapped__", entry), "__name__", "")

    assert name(TRAIN_KEY) == "train_cellpose"
    assert name(APPLY_KEY) == "identify_masks_finetune"


def test_the_factory_wires_the_host_the_way_a_module_page_is_wired(qtbot):
    """A screen built by the registry gets the window's two connections, or
    "Explain error" and "Submit remotely" reach nobody."""
    from spacr.qt.screens.train_cellpose import build_screen

    class Host:
        def __init__(self):
            self.errors = []
            self.submissions = []

        def _on_explain_error(self, text, key):
            self.errors.append((text, key))

        def _on_remote_submit_requested(self, key, settings):
            self.submissions.append((key, settings))

    host = Host()
    screen = build_screen(host=host)
    qtbot.addWidget(screen)
    screen.apply_screen.error_explain_requested.emit("boom", APPLY_KEY)
    screen.train_screen.remote_submit_requested.emit(TRAIN_KEY, {"src": "/x"})
    assert host.errors == [("boom", APPLY_KEY)]
    assert host.submissions == [(TRAIN_KEY, {"src": "/x"})]


def test_the_api_link_follows_the_visible_half(workbench):
    """One masthead serves two modules, so its API link has to reach the one
    on screen — the embedded pages' own mastheads are hidden.

    There is no dot to read it off any more: the link is the last line of
    the description's hover help, so that is where it is read from.
    """
    from spacr.qt.screens.settings_model import api_docs_url
    from spacr.qt.widgets.info_link import InfoLink

    assert not workbench.findChildren(InfoLink), (
        "the masthead grew an information dot back")
    help_label = workbench._header.api_help
    assert help_label is not None
    assert help_label.url() == api_docs_url(TRAIN_KEY)
    assert TRAIN_KEY in str(help_label.property("moduleApiAppKey"))
    workbench._tabs.setCurrentIndex(1)
    assert help_label.url() == api_docs_url(APPLY_KEY)
    assert workbench.train_screen._header.isHidden()
    assert workbench.apply_screen._header.isHidden()


def test_the_help_the_hover_shows_carries_the_visible_half_s_link(
        workbench, qtbot):
    """Measured through the popup, which is what the reader actually gets.

    The dot could be pointed at the right page and still be the wrong
    answer if nothing showed it; this drives the hover and reads the link
    out of the tooltip that appears.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    from spacr.qt.screens.settings_model import api_docs_url
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    workbench.show()
    qtbot.waitExposed(workbench)
    help_label = workbench._header.api_help
    tooltip = HoverTooltip.instance()

    QApplication.sendEvent(help_label, QEvent(QEvent.Type.Enter))
    assert tooltip.isVisible()
    assert WORKBENCH_INTRO in tooltip._label.text()
    assert tooltip.api_url() == api_docs_url(TRAIN_KEY)
    QApplication.sendEvent(help_label, QEvent(QEvent.Type.Leave))

    workbench._tabs.setCurrentIndex(1)
    QApplication.sendEvent(help_label, QEvent(QEvent.Type.Enter))
    assert tooltip.api_url() == api_docs_url(APPLY_KEY)
    QApplication.sendEvent(help_label, QEvent(QEvent.Type.Leave))


def test_a_language_change_repoints_the_masthead_help(workbench):
    """The API pages are per language, and the help has to follow.

    The language pass finds the label by its ``moduleApiAppKey`` property,
    exactly as it found the dot, and rebuilds the prose with it.
    """
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.screens.settings_model import api_docs_url

    help_label = workbench._header.api_help
    try:
        retranslate_widget_tree(workbench, "sv")
        assert help_label.url() == api_docs_url(TRAIN_KEY, language="sv")
        assert "?lang=sv" in help_label.help_html()

        # The two repointings compose. The masthead was built for the
        # training half, so a language pass that read only the key it was
        # built with would send a reader on the Apply tab to the other
        # module's page.
        workbench._tabs.setCurrentIndex(1)
        retranslate_widget_tree(workbench, "sv")
        assert help_label.url() == api_docs_url(APPLY_KEY, language="sv")
    finally:
        workbench._tabs.setCurrentIndex(0)
        retranslate_widget_tree(workbench, "en")
    assert help_label.url() == api_docs_url(TRAIN_KEY)
