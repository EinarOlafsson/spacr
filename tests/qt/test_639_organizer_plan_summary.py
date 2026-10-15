"""The organizer describes the chosen output without mutating its move plan."""
from copy import deepcopy
from pathlib import Path

import pytest

from spacr.qt.widgets import organize_for_measure as ofm


pytestmark = pytest.mark.qt


@pytest.fixture(params=[("measure", "measure"), ("images", "mask"),
                        ("import", "measure"), ("import", "mask")])
def organizer(request, qtbot, qt_theme_applied):
    mode, target = request.param
    dialog = ofm._dialog_for(mode)()
    qtbot.addWidget(dialog)
    dialog.layout_box.setCurrentIndex(dialog.layout_box.findData(target))
    yield dialog, target
    dialog._stop_thumbs()


def _plan(tmp_path, with_masks):
    source = str(tmp_path / "input")
    dest = str(tmp_path / "sorted")
    rows = []
    for channel in (1, 2):
        rows.append(ofm.cs.PlanRow(
            channel, ofm.cs.SetName("plate1", "A01", 1), (("fieldID", "1"),),
            str(Path(source) / f"channel{channel}.tif"),
            str(Path(dest) / f"C{channel:02d}" / f"channel{channel}.tif"),
            source_mask=str(Path(source) / "masks" / f"channel{channel}.tif")
            if with_masks else None,
            target_mask=str(Path(dest) / f"C{channel:02d}" / "masks" /
                            f"channel{channel}.tif") if with_masks else None))
    return ofm.cs.SortPlan(
        source, dest, rows=rows, channels=[1, 2],
        mask_roles={1: "nucleus", 2: "cell"} if with_masks else {},
        warnings=["First warning: preserve its wording.", "Second warning: μ."])


@pytest.mark.parametrize("with_masks", [False, True])
def test_apply_confirmation_matches_selected_layout_and_keeps_plan(
        organizer, tmp_path, monkeypatch, with_masks):
    dialog, target = organizer
    plan = _plan(tmp_path, with_masks)
    before = deepcopy(plan)
    monkeypatch.setattr(dialog, "prepare_plan", lambda **kwargs: plan)
    seen = []
    dialog.ask = lambda title, text: seen.append((title, text)) or True
    dialog._on_apply()
    assert dialog.plan is plan
    assert plan == before
    assert len(seen) == 1
    title, text = seen[0]
    first_line = text.splitlines()[0]
    assert "Move 2 image(s)" in first_line
    assert plan.dest in first_line
    if target == "mask":
        assert title == "Move the images?"
        assert "Do not write stacks or merged arrays." in first_line
        assert str(Path(plan.dest) / "masks") in first_line
        assert "channel folder(s)" not in first_line
    else:
        assert title == "Move and merge?"
        assert "2 channel folder(s)" in first_line
        assert "1 set(s) in Yokogawa naming" in first_line
        assert "Then build channel stacks and any merged arrays supported " \
               "by the available masks." in first_line
    if with_masks:
        assert "Mask roles: channel 1: nucleus, channel 2: cell" in text
    else:
        assert "Mask roles:" not in text
    assert text.splitlines()[-2:] == before.warnings


def test_invalid_plan_keeps_every_warning_and_problem_in_the_status(
        organizer, tmp_path, monkeypatch):
    dialog, target = organizer
    plan = _plan(tmp_path, True)
    plan.problems = ["Unknown external failure: /data/μ.tif", "Last problem."]
    before = deepcopy(plan)
    monkeypatch.setattr(dialog, "prepare_plan", lambda **kwargs: plan)
    questions = []
    dialog.ask = lambda *args: questions.append(args) or True
    dialog._on_apply()
    assert questions == []
    assert dialog.plan is None
    assert plan == before
    text = dialog.status.text()
    assert text.splitlines()[-4:] == before.warnings + before.problems
    if target == "mask":
        assert "Do not write stacks or merged arrays." in text.splitlines()[0]
    else:
        assert "Then build channel stacks" in text.splitlines()[0]
