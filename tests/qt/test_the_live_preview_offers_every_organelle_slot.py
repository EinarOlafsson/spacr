"""One live-preview organelle entry per declared slot.

With ``number_of_organelles`` at 2 the panel offered a single "organelle"
entry: the second slot could not be previewed at all, and anything tuned while
it was notionally selected propagated into the FIRST slot's keys -- silently
re-tuning an organelle the user was not looking at.

Reported 2026-09-01: "in the live settings primary object settings, if in the
main settings number of objects is 2 then there should be 2 organelle options
there other wise we cant propegate the settings properly".

The captions count (``organelle 2``) because that is what the main panel
counts; the settings keys use letters (``organelleb``) because a digit cannot
start a Python identifier. :func:`object_role` is the join between them.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.live_preview import (LivePreviewPanel, object_role,
                                           organelle_label)


@pytest.fixture
def panel(qapp):
    return LivePreviewPanel()


def _captions(panel):
    box = panel._object_box
    return [box.itemData(i) or box.itemText(i) for i in range(box.count())]


def test_one_organelle_is_not_renumbered(panel):
    """The ordinary run must not be relabelled to say something new."""
    panel.apply_settings({"number_of_organelles": 1})
    assert "organelle" in _captions(panel)
    assert "organelle 1" not in _captions(panel)


def test_two_organelles_are_both_offered(panel):
    panel.apply_settings({"number_of_organelles": 2})
    captions = _captions(panel)
    assert "organelle" in captions and "organelle 2" in captions


def test_four_organelles_are_all_offered(panel):
    panel.apply_settings({"number_of_organelles": 4})
    captions = _captions(panel)
    for n in (2, 3, 4):
        assert organelle_label(n) in captions, n


def test_the_second_slot_propagates_to_its_own_keys(panel):
    """The defect this is for: slot 2's tuning must not land on slot 1."""
    panel.apply_settings({"number_of_organelles": 2})
    panel._object_box.setCurrentText("organelle 2")
    panel._organelle_channel.setValue(3)

    out = panel.settings_for_propagation()

    assert out["organelleb_channel"] == 3
    assert out.get("organelle_channel") != 3, (
        "slot 2's channel was written into slot 1")


def test_each_slot_keeps_its_own_channel(panel):
    """Switching between slots must not carry a channel across."""
    panel.apply_settings({"number_of_organelles": 2,
                          "organelle_channel": 1,
                          "organelleb_channel": 3})
    panel._object_box.setCurrentText("organelle")
    assert panel._organelle_channel.value() == 1
    panel._object_box.setCurrentText("organelle 2")
    assert panel._organelle_channel.value() == 3
    panel._object_box.setCurrentText("organelle")
    assert panel._organelle_channel.value() == 1


def test_the_compartment_tuning_follows_the_selected_slot(panel):
    """One set of organelle widgets serves whichever slot is selected, so its
    keys must carry that slot's role."""
    panel.apply_settings({"number_of_organelles": 2})
    panel._object_box.setCurrentText("organelle 2")

    out = panel.settings_for_propagation()

    assert any(k.startswith("organelleb_") for k in out), (
        "no key named slot 2 at all")
    assert "organelleb_min_area" in out


def test_the_model_follows_the_selected_slot(panel):
    panel.apply_settings({"number_of_organelles": 2})
    panel._object_box.setCurrentText("organelle 2")
    panel._model_box.addItem("/models/mito.pth")
    panel._model_box.setCurrentText("/models/mito.pth")
    assert panel.settings_for_propagation()[
        "organelleb_model_name"] == "/models/mito.pth"


def test_raising_the_count_keeps_the_current_selection(panel):
    """A rebuild that threw the user back to "cell" would lose their place
    every time the main panel changed."""
    panel.apply_settings({"number_of_organelles": 2})
    panel._object_box.setCurrentText("organelle 2")
    panel.apply_settings({"number_of_organelles": 4})
    assert panel._object_box.currentText() == "organelle 2"


def test_the_caption_and_the_role_agree():
    assert object_role(organelle_label(1)) == "organelle"
    assert object_role(organelle_label(2)) == "organelleb"
    assert object_role(organelle_label(3)) == "organellec"
