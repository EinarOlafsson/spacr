"""333: the live preview segments with the model the RUN is configured with.

The preview built its Cellpose model from a combo the panel seeded from
``model_name`` and nothing else. Mask does not declare ``model_name`` -- a
built Mask screen's ``_settings_model.collect()`` carries ``cell_model_name``,
``nucleus_model_name``, ``organelle_model_name``, ``pathogen_model_name`` and
``pathogen_model``, and no bare ``model_name`` at all -- so the combo was never seeded from Mask. A user who
chose a zoo checkpoint for the pathogens got stock cpsam in the preview, and
the preview is the thing they are looking at while deciding whether the
diameter and the thresholds are right.

That is the worst shape this defect can take. The preview does not fail; it
answers a different question and looks authoritative doing it. So the three
properties tested here are:

  * the preview reads the SAME key the run reads, INCLUDING the pathogen
    override (``spacr/object.py`` 696-697: ``pathogen_model`` beats
    ``pathogen_model_name``);
  * the picture says which model made it, and says it about the pass that ran
    rather than about whatever the combo holds now;
  * a checkpoint that is not on this machine previews with a STATED fallback
    rather than stalling or being silently substituted.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def panel(qapp):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    widget = LivePreviewPanel(threaded=False)
    yield widget
    widget.deleteLater()


@pytest.fixture
def checkpoint(tmp_path):
    """A file that exists, so it is a loadable checkpoint rather than a typo."""
    path = tmp_path / "cpsam_v2_toxo_r2.pth"
    path.write_bytes(b"weights")
    return str(path)


def _mask_arrived(panel, model_box_text=None):
    """Push one mask through the documented direct-call hatch (token -1)."""
    panel._image = np.zeros((8, 8), dtype=np.uint16)
    if model_box_text is not None:
        _offer(panel, model_box_text)
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[0:4, 0:4] = 1
    panel._build_request()          # this is what decides the model for a pass
    panel._on_worker_done({"cell": mask}, "", -1)


def _offer(panel, text):
    """Select ``text`` in the model combo, adding it if it is not offered."""
    index = panel._model_box.findText(text)
    if index < 0:
        panel._model_box.addItem(text)
        index = panel._model_box.count() - 1
    panel._model_box.setCurrentIndex(index)


# --------------------------------------------------------------------------
# 1. the preview reads the key the run reads
# --------------------------------------------------------------------------

def test_a_pathogen_checkpoint_reaches_the_preview(panel, checkpoint):
    """The reported defect, in one assertion.

    ``pathogen_model_name`` is a real Mask row today -- the panel builds a
    widget for it -- so this is the path a user actually walks.
    """
    panel._object_box.setCurrentText("pathogen")
    panel.apply_settings({"pathogen_model_name": checkpoint})

    assert panel._model_box.currentText() == checkpoint, (
        "the preview would have segmented with cpsam while the run used "
        f"{checkpoint}")


def test_pathogen_model_wins_the_way_it_wins_in_the_run(panel, tmp_path):
    """``spacr/object.py`` 696-697 overrides the name key with this one.

    A preview that preferred the other key would disagree with the run in
    exactly the cases where the user has gone to the trouble of setting both.
    """
    override = tmp_path / "override.pth"
    override.write_bytes(b"w")
    named = tmp_path / "named.pth"
    named.write_bytes(b"w")

    panel._object_box.setCurrentText("pathogen")
    panel.apply_settings({"pathogen_model": str(override),
                          "pathogen_model_name": str(named)})

    assert panel._model_box.currentText() == str(override)


def test_an_unset_override_does_not_win(panel, checkpoint):
    """Mask ALWAYS carries ``pathogen_model``; unset is spelled ``None``.

    The run tests it with ``is not None`` for that reason. A preview that
    read the key's presence would have previewed a model called "None".
    """
    panel._object_box.setCurrentText("pathogen")
    panel.apply_settings({"pathogen_model": None,
                          "pathogen_model_name": checkpoint})

    assert panel._model_box.currentText() == checkpoint


def test_each_object_reads_its_own_model(panel, tmp_path):
    """The model is a per-object setting and this panel has one combo."""
    cell = tmp_path / "cell.pth"
    cell.write_bytes(b"w")
    pathogen = tmp_path / "pathogen.pth"
    pathogen.write_bytes(b"w")
    settings = {"cell_model_name": str(cell),
                "pathogen_model_name": str(pathogen)}

    panel._object_box.setCurrentText("cell")
    panel.apply_settings(settings)
    assert panel._model_box.currentText() == str(cell)

    panel._object_box.setCurrentText("pathogen")
    assert panel._model_box.currentText() == str(pathogen), (
        "the panel opens on cell, so without following the object selector "
        "the pathogen case -- the one this was reported for -- never fires")


def test_a_model_the_user_picked_survives_an_object_switch(panel, tmp_path,
                                                           monkeypatch):
    """Undoing the user's own choice would be this item's defect, committed
    by the fix for it."""
    import spacr.qt.widgets.model_zoo_picker as picker

    cell = tmp_path / "cell.pth"
    cell.write_bytes(b"w")
    chosen = tmp_path / "chosen_from_the_zoo.pth"
    chosen.write_bytes(b"w")

    panel._object_box.setCurrentText("cell")
    panel.apply_settings({"cell_model_name": str(cell),
                          "pathogen_model_name": "cpsam"})
    monkeypatch.setattr(picker, "choose_model", lambda *a, **k: str(chosen))
    panel._choose_a_preview_model()
    assert panel._model_box.currentText() == str(chosen)

    panel._object_box.setCurrentText("pathogen")
    assert panel._model_box.currentText() == str(chosen), (
        "the panel overwrote a checkpoint the user had just chosen")


def test_the_bare_model_name_still_seeds_the_modules_that_use_it(panel):
    """A panel serving no named module still reads the bare ``model_name``,
    and so does Cellpose Masks while ``custom_model`` is unset. That read
    must not be lost. (Plaque Assay does NOT read it -- section 6.)"""
    panel.apply_settings({"model_name": "cyto2"})

    assert panel._model_box.currentText() == "cyto2"


# --------------------------------------------------------------------------
# 2. a model that is not on this machine
# --------------------------------------------------------------------------

def test_a_checkpoint_that_is_not_there_previews_with_a_stated_fallback(panel):
    """The RUN stops on this and should. The PREVIEW must not: a zoo model
    the user has not downloaded would turn Run preview into a button that
    only ever produces an error."""
    _offer(panel, "/models/not_downloaded_yet.pth")

    model, note = panel._model_for_this_pass()

    assert model == "cpsam", "the preview must still produce a picture"
    assert "/models/not_downloaded_yet.pth" in note, (
        f"the fallback has to name what was asked for, not just happen: {note!r}")


def test_a_model_that_is_there_is_not_second_guessed(panel, checkpoint):
    """The fallback must fire on a MISSING file and on nothing else."""
    _offer(panel, checkpoint)

    model, note = panel._model_for_this_pass()

    assert model == checkpoint
    assert note == ""


def test_a_configured_model_that_was_never_downloaded_shows_and_explains_itself(
        panel):
    """The item's third requirement, end to end.

    The checkpoint the RUN is configured with is offered even though it is
    not here -- hiding it would put the preview back to showing cpsam and
    saying nothing -- and the pass falls back with the substitution stated.
    """
    panel._object_box.setCurrentText("pathogen")
    panel.apply_settings({"pathogen_model": "/models/not_downloaded_yet.pth"})

    assert panel._model_box.currentText() == "/models/not_downloaded_yet.pth"
    model, note = panel._model_for_this_pass()
    assert model == "cpsam"
    assert "/models/not_downloaded_yet.pth" in note


def test_the_request_carries_the_model_that_can_actually_load(panel):
    """The worker is handed the fallback, so the pass runs rather than
    raising ``FileNotFoundError`` out of ``_resolve_cellpose_pretrained``."""
    panel._image = np.zeros((8, 8), dtype=np.uint16)
    _offer(panel, "/models/not_downloaded_yet.pth")

    assert panel._build_request().model == "cpsam"


# --------------------------------------------------------------------------
# 3. the picture says which model made it
# --------------------------------------------------------------------------

def test_the_status_line_names_the_model_that_made_the_masks(panel,
                                                             checkpoint):
    _mask_arrived(panel, checkpoint)

    assert checkpoint in panel._status.text(), (
        "a preview whose provenance is unstated is the same defect wearing "
        f"a different hat: {panel._status.text()!r}")


def test_the_status_line_states_the_fallback_it_made(panel):
    _mask_arrived(panel, "/models/not_downloaded_yet.pth")

    text = panel._status.text()
    assert "cpsam" in text and "/models/not_downloaded_yet.pth" in text, (
        f"the substitution was made in silence: {text!r}")


def test_the_status_names_the_model_that_ran_not_the_one_now_selected(
        panel, checkpoint):
    """Changing the combo does not re-segment. Captioning the masks with the
    current selection would name a model that never touched them."""
    _mask_arrived(panel, checkpoint)
    _offer(panel, "cpsam")
    panel._recompute_masks()

    assert checkpoint in panel._status.text()


def test_the_run_history_records_the_model_that_ran(panel):
    """The history is scrubbed back to compare one pass against another, so
    a pass has to be labelled with the model that made it.

    Driven through a FALLBACK pass, which is the case where the model asked
    for and the model that ran differ at all. Asserting this on an ordinary
    pass proves nothing: the combo still holds the model that ran, so
    reading either gives the same answer -- a mutation that put the combo
    back left the assertion green.
    """
    _mask_arrived(panel, "/models/not_downloaded_yet.pth")

    assert panel._history[-1]["model"] == "cpsam", (
        "the history labelled the picture with a checkpoint that is not on "
        "this machine and therefore made none of it: "
        f"{panel._history[-1]['model']!r}")


# --------------------------------------------------------------------------
# 4. the round trip
# --------------------------------------------------------------------------

def test_propagation_writes_back_the_key_it_read(panel, tmp_path):
    """Seeding and propagation are one contract. Writing only
    ``pathogen_model_name`` back left ``pathogen_model`` overriding it, so
    the run kept the old checkpoint -- the same disagreement, reversed."""
    override = tmp_path / "override.pth"
    override.write_bytes(b"w")

    panel._object_box.setCurrentText("pathogen")
    panel.apply_settings({"pathogen_model": str(override),
                          "pathogen_model_name": "cpsam"})
    _offer(panel, "cpsam")

    out = panel.settings_for_propagation()
    assert out["pathogen_model"] == "cpsam", (
        "the run would have gone on using "
        f"{override}: {out.get('pathogen_model')!r}")


def test_propagation_does_not_switch_the_override_on(panel, checkpoint):
    """``pathogen_model`` is validated harder than the name key -- a run
    stops on a path that is not there. Turning it on for a user who never
    set it is not the preview's to do."""
    panel._object_box.setCurrentText("pathogen")
    panel.apply_settings({"pathogen_model": None,
                          "pathogen_model_name": checkpoint})

    assert "pathogen_model" not in panel.settings_for_propagation()


# --------------------------------------------------------------------------
# 5. the zoo is the same zoo
# --------------------------------------------------------------------------

def test_the_preview_offers_only_cellpose_models(panel, monkeypatch):
    """``kinds`` is a rule rather than a parameter: the zoo also carries the
    YOLO well detector and ``CellposeModel`` cannot load it, so offering it
    here produces a preview that fails the moment it is selected."""
    import spacr.qt.widgets.model_zoo_picker as picker

    seen = {}
    monkeypatch.setattr(
        picker, "choose_model",
        lambda parent=None, kinds=None: seen.update(kinds=kinds))
    panel._choose_a_preview_model()

    assert seen["kinds"] == ("cellpose",)


# --------------------------------------------------------------------------
# 6. the other two modules this panel serves
# --------------------------------------------------------------------------
#
# ``cellpose_masks`` and ``analyze_plaques`` reach this panel through
# :mod:`spacr.qt.preview_registry`, and NEITHER RUN SEGMENTS WITH
# ``model_name`` the way the first pass assumed. Measured on screens built
# from this tree (2026-09-14):
#
#   analyze_plaques  collect() -> {'plaque_model': 'bundled', 'model_name': 'cpsam'}
#   cellpose_masks   collect() -> {'model_name': 'cpsam', 'custom_model': None}
#
# The plaque run resolves ``plaque_model`` through
# ``spacr.submodules._resolve_plaque_model`` and hands the result to Cellpose
# as ``custom_model``; ``model_name`` never reaches Cellpose. So the plaque
# preview showed cpsam's masks against a run on the plaque model -- the
# defect this file exists for, one module over. The Cellpose Masks run loads
# ``custom_model`` over ``model_name`` whenever it is set
# (``spacr.spacr_cellpose.identify_masks_finetune``).


@pytest.fixture
def module_panel(qapp):
    """A synchronous panel serving ``module``, as the registry mounts it."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    made = []

    def build(module):
        widget = LivePreviewPanel(threaded=False, module=module)
        made.append(widget)
        return widget

    yield build
    for widget in made:
        widget.deleteLater()


@pytest.fixture
def no_download(monkeypatch):
    """Fail the test if anything starts a download.

    A preview refresh that fetched 1.2 GB would be the surprise the item's
    WATCH OUT forbids, and both of the resolver's download paths are named
    here so neither can be reached quietly.
    """
    from spacr import model_zoo, utils

    def refuse(*_a, **_k):
        raise AssertionError("the preview started a download")

    monkeypatch.setattr(model_zoo, "fetch", refuse)
    monkeypatch.setattr(utils, "download_models", refuse)


def _zoo_entry(monkeypatch, key, name):
    from types import SimpleNamespace

    from spacr import model_zoo

    entry = SimpleNamespace(key=key, name=name, path="")
    monkeypatch.setattr(model_zoo, "catalogue",
                        lambda *a, **k: [entry])
    return entry


def _registry_built(qtbot, module):
    """The card and panel exactly as ``preview_registry`` builds them."""
    from types import SimpleNamespace

    from spacr.qt.screens.app_screen import _build_live_preview_card

    panel, card = _build_live_preview_card(SimpleNamespace(app_key=module))
    qtbot.addWidget(card)
    # Returned and HELD by the caller: qtbot keeps no strong reference, and
    # the panel is the card's child, so dropping the card deletes the combo.
    return panel, card


def _wait_for_model(qtbot, panel, expected, timeout=20000):
    """Wait for an off-thread resolution, then assert with what was SEEN."""
    try:
        qtbot.waitUntil(
            lambda: panel._model_box.currentText() == expected,
            timeout=timeout)
    except Exception:                                    # noqa: BLE001
        pass
    assert panel._model_box.currentText() == expected, (
        f"the run segments with {expected!r}; the preview holds "
        f"{panel._model_box.currentText()!r}")


def test_the_plaque_preview_uses_the_model_the_plaque_run_resolves(
        qtbot, checkpoint):
    """The reported shape, for Plaque Assay, through the registry's builder.

    The expectation is the RUN's own resolver's answer, not a copy of its
    rules, so the two cannot drift apart again.
    """
    from spacr.submodules import _resolve_plaque_model

    settings = {"plaque_model": checkpoint, "model_name": "cpsam"}
    expected = _resolve_plaque_model(dict(settings))

    panel, _card = _registry_built(qtbot, "analyze_plaques")
    panel.apply_settings(settings)

    _wait_for_model(qtbot, panel, expected)


def test_the_plaque_preview_follows_the_real_screens_default(
        qtbot, monkeypatch):
    """End to end on a built Plaque Assay screen, whose default is the v2 model.

    Resolved WITHOUT fetching, as the preview does: the default is a 1.2 GB
    zoo checkpoint since 2026-09-21, and a test must not download it.

    Since item 468 the screen's own Plaque preview (not the Mask panel)
    holds the form's ``plaque_model`` and resolves it with the run's
    resolver when a pass starts, so what it shows is the setting itself.
    Before item 333 the preview seeded ``model_name`` ('cpsam') from the
    same form, so the two most visible values on the screen disagreed.
    """
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets.plaque_preview import PlaquePreviewPanel

    window = MainWindow()
    qtbot.addWidget(window)
    window._on_nav_selected("analyze_plaques")
    qtbot.wait(50)
    screen = window._screens["analyze_plaques"]
    collected = screen._settings_model.collect()
    from spacr.submodules import DEFAULT_PLAQUE_MODEL

    assert collected["plaque_model"] == DEFAULT_PLAQUE_MODEL
    panel = screen._live_preview
    assert isinstance(panel, PlaquePreviewPanel)

    screen._on_preview_switch(True)

    assert panel._model_box.currentText() == str(
        collected.get("plaque_model") or DEFAULT_PLAQUE_MODEL)
    assert panel.current_settings()["plaque_model"] == \
        panel._model_box.currentText()


def test_an_unset_plaque_model_means_what_the_run_takes_it_to_mean(
        module_panel, tmp_path, monkeypatch, no_download):
    """``None`` is the run's default, the v2 zoo model since 2026-09-21. A
    copy already on this machine -- under a HOME the test controls -- is what
    the preview previews, and nothing is downloaded to find it."""
    import spacr.submodules as sm

    monkeypatch.setenv("HOME", str(tmp_path))
    local = tmp_path / ".spacr" / "models" / "cpsam_plaque_r5"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"w")
    assert sm._requested_plaque_model({"plaque_model": None}) == (
        sm.DEFAULT_PLAQUE_MODEL)

    panel = module_panel("analyze_plaques")
    panel.apply_settings({"plaque_model": None, "model_name": "cyto2"})

    assert panel._model_box.currentText() == str(local)
    assert panel._model_for_this_pass() == (str(local), "")


def test_a_downloaded_plaque_zoo_key_previews_its_local_copy(
        module_panel, tmp_path, monkeypatch, no_download):
    monkeypatch.setenv("HOME", str(tmp_path))
    _zoo_entry(monkeypatch, "toxoplasma_plaque_v1", "cpsam_plaque_r3")
    local = tmp_path / ".spacr" / "models" / "cpsam_plaque_r3"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"w")

    panel = module_panel("analyze_plaques")
    panel.apply_settings({"plaque_model": "toxoplasma_plaque_v1"})

    assert panel._model_box.currentText() == str(local)


def test_a_plaque_zoo_key_not_downloaded_is_stated_and_never_fetched(
        module_panel, tmp_path, monkeypatch, no_download):
    """A zoo KEY is not path-shaped, so the path test that catches a missing
    checkpoint cannot see it. Without the resolver's answer the key would
    either be dropped (cpsam, in silence) or handed to Cellpose, which maps
    an unknown name to cpsam with a log line and nothing on screen."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _zoo_entry(monkeypatch, "toxoplasma_plaque_v1", "cpsam_plaque_r3")

    panel = module_panel("analyze_plaques")
    panel.apply_settings({"plaque_model": "toxoplasma_plaque_v1"})

    assert panel._model_box.currentText() == "toxoplasma_plaque_v1"
    model, note = panel._model_for_this_pass()
    assert model == "cpsam"
    assert "toxoplasma_plaque_v1" in note


def test_a_missing_bundled_plaque_model_is_stated_and_never_downloaded(
        module_panel, tmp_path, monkeypatch, no_download):
    import spacr.submodules as sm

    monkeypatch.setattr(sm, "__file__",
                        str(tmp_path / "empty" / "submodules.py"))

    panel = module_panel("analyze_plaques")
    panel.apply_settings({"plaque_model": "bundled"})

    model, note = panel._model_for_this_pass()
    assert model == "cpsam"
    assert "bundled" in note


def test_a_pass_started_before_the_resolution_lands_still_uses_it(
        qapp, checkpoint):
    """The plaque resolver lives in ``spacr.submodules``, a 3.5 s import the
    app has not paid when the preview opens, so it runs off the GUI thread.
    A pass that starts before the answer is delivered must not segment with
    whatever the combo held in the meantime."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    panel = LivePreviewPanel(threaded=True, module="analyze_plaques")
    try:
        panel.apply_settings({"plaque_model": checkpoint,
                              "model_name": "cpsam"})
        # No event processing in between: the job's answer cannot have been
        # delivered yet.
        assert panel._model_for_this_pass() == (checkpoint, "")
    finally:
        panel._model_jobs.shutdown()
        panel.deleteLater()


def test_a_late_resolution_does_not_undo_a_model_the_user_picked(
        qtbot, checkpoint, tmp_path):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    chosen = tmp_path / "chosen.pth"
    chosen.write_bytes(b"w")
    panel = LivePreviewPanel(threaded=True, module="analyze_plaques")
    qtbot.addWidget(panel)
    panel.apply_settings({"plaque_model": checkpoint})
    _offer(panel, str(chosen))

    qtbot.waitUntil(lambda: panel._model_jobs.pending_jobs() == 0,
                    timeout=20000)
    qtbot.wait(20)
    assert panel._model_box.currentText() == str(chosen)
    panel._model_jobs.shutdown()


def test_the_cellpose_masks_preview_loads_custom_model_over_model_name(
        qtbot, checkpoint):
    panel, _card = _registry_built(qtbot, "cellpose_masks")
    panel.apply_settings({"model_name": "cpsam", "custom_model": checkpoint})

    assert panel._model_box.currentText() == checkpoint, (
        "the Cellpose Masks run loads custom_model; the preview showed "
        f"{panel._model_box.currentText()!r}")


def test_an_unset_custom_model_leaves_model_name_in_charge(module_panel):
    panel = module_panel("cellpose_masks")
    panel.apply_settings({"model_name": "cyto2", "custom_model": None})

    assert panel._model_box.currentText() == "cyto2"


# ---- the round trip, for both ---------------------------------------------

def test_an_untouched_plaque_model_is_not_rewritten_as_a_path(
        module_panel, checkpoint, no_download):
    """'bundled' and a zoo key resolve to paths. Writing the resolved path
    back over the setting the user chose would change what a recorded run
    says it asked for, which the resolver's own docstring warns against."""
    panel = module_panel("analyze_plaques")
    panel.apply_settings({"plaque_model": checkpoint})

    assert "plaque_model" not in panel.settings_for_propagation()


def test_a_plaque_checkpoint_the_user_picks_propagates(
        module_panel, checkpoint, tmp_path, no_download):
    from spacr.qt.preview_registry import PREVIEWS

    chosen = tmp_path / "chosen.pth"
    chosen.write_bytes(b"w")
    panel = module_panel("analyze_plaques")
    panel.apply_settings({"plaque_model": checkpoint})
    _offer(panel, str(chosen))

    out = panel.settings_for_propagation()
    assert out["plaque_model"] == str(chosen)
    assert PREVIEWS["analyze_plaques"].propagation.get(
        "plaque_model") == "plaque_model", (
        "the registry drops every name its map does not carry")


def test_a_stock_name_is_not_written_into_plaque_model(
        module_panel, checkpoint, no_download):
    """The plaque run cannot load a stock name -- the resolver raises
    ValueError on 'cpsam' -- so propagating one would break the run."""
    panel = module_panel("analyze_plaques")
    panel.apply_settings({"plaque_model": checkpoint})
    _offer(panel, "cpsam")

    assert "plaque_model" not in panel.settings_for_propagation()


def test_choosing_a_stock_model_clears_custom_model(module_panel, checkpoint):
    """Writing only ``model_name`` back left ``custom_model`` in charge, and
    writing 'cpsam' INTO ``custom_model`` stops the run outright: it prints
    "Custom model not found" and returns."""
    from spacr.qt.preview_registry import PREVIEWS

    panel = module_panel("cellpose_masks")
    panel.apply_settings({"model_name": "cpsam", "custom_model": checkpoint})
    _offer(panel, "cpsam")

    out = panel.settings_for_propagation()
    assert out["custom_model"] is None
    assert out["model_name"] == "cpsam"
    assert PREVIEWS["cellpose_masks"].propagation.get(
        "custom_model") == "custom_model"


def test_an_untouched_custom_model_round_trips(module_panel, checkpoint):
    panel = module_panel("cellpose_masks")
    panel.apply_settings({"model_name": "cpsam", "custom_model": checkpoint})

    assert panel.settings_for_propagation()["custom_model"] == checkpoint


def test_propagation_does_not_switch_custom_model_on(module_panel, checkpoint):
    panel = module_panel("cellpose_masks")
    panel.apply_settings({"model_name": "cpsam", "custom_model": None})
    _offer(panel, checkpoint)

    assert "custom_model" not in panel.settings_for_propagation()


# --------------------------------------------------------------------------
# 7. organelle slots read the key the run reads
# --------------------------------------------------------------------------

def test_an_organelle_slot_reads_the_key_the_run_reads(panel, tmp_path):
    """The run reads a slot's model through
    ``spacr.object_roles.organelle_settings_view``, which exposes
    ``organelleb_model_name`` as ``organelle_model_name``. Asserted against
    that function rather than against a spelling, so a change to the view
    is a change to this test."""
    from spacr.object_roles import organelle_settings_view
    from spacr.organelle_types import NUMBER_OF_ORGANELLES
    from spacr.qt.widgets.live_preview import object_role

    first = tmp_path / "first.pth"
    first.write_bytes(b"w")
    second = tmp_path / "second.pth"
    second.write_bytes(b"w")
    settings = {NUMBER_OF_ORGANELLES: 2,
                "organelle_model_name": str(first),
                "organelleb_model_name": str(second)}
    panel.apply_settings(settings)

    box = panel._object_box
    index = next(i for i in range(box.count())
                 if object_role(box.itemData(i) or box.itemText(i))
                 == "organelleb")
    box.setCurrentIndex(index)

    run = organelle_settings_view(settings, "organelleb")
    assert panel._model_box.currentText() == run["organelle_model_name"]


# --------------------------------------------------------------------------
# 8. the provenance clause is in the reader's language
# --------------------------------------------------------------------------

def test_the_stated_fallback_is_in_the_readers_language(panel, monkeypatch):
    """Composed only from catalogue sources that already exist -- ``Model``
    and ``missing`` -- so no new caption enters the pinned inventory."""
    from spacr.qt import i18n
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES

    for language in CATALOG_LANGUAGES:
        monkeypatch.setattr(i18n, "current_language",
                            lambda code=language: code)
        _mask_arrived(panel, "/models/not_downloaded_yet.pth")
        text = panel._status.text()
        assert i18n.tr("missing", language) in text, (language, text)
        assert i18n.tr("Model", language) in text, (language, text)
        assert "not on this machine" not in text, (language, text)
        assert "/models/not_downloaded_yet.pth" in text


# --------------------------------------------------------------------------
# 9. the tracking preview Mask folds in for Timelapse
# --------------------------------------------------------------------------

def test_the_track_preview_segments_with_the_tracked_objects_run_model(
        qapp, checkpoint):
    """Same defect, second panel. The Timelapse fold tracks the masks the
    Mask run makes, and those are made with that object's model key -- but
    ``TimelapsePreviewPanel.apply_settings`` seeded the object, channel and
    diameter and never the model, so every frame was segmented with the
    menu's first entry."""
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    track = TimelapsePreviewPanel(threaded=False)
    try:
        track.apply_settings({"timelapse_objects": ["pathogen"],
                              "pathogen_model": checkpoint,
                              "pathogen_model_name": "cpsam"})
        assert track._object_box.currentText() == "pathogen"
        assert track.current_params()["model"] == checkpoint, (
            "the track preview segmented with "
            f"{track.current_params()['model']!r}")
    finally:
        track.deleteLater()
