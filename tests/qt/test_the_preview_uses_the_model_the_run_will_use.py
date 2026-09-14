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
    """``cellpose_masks`` and ``analyze_plaques`` reach this panel through
    the preview registry, have one object type and call its model
    ``model_name``. That read must not be lost."""
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
