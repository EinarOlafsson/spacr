"""The GPU line on the first slide of the setup screen.

Asked for: say that segmentation and object classification need an NVIDIA
GPU, and tell the reader whether this machine has one -- green when it
does, red when it does not, and NAME the card either way. "No compatible
GPU" on its own leaves the reader wondering whether spaCR looked.

The greeting moved up a row to make space, which is why the two bands are
asserted here as well: they must not overlap, or the word that fades in
lands on top of the sentence.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import setup_slides
from spacr.qt.widgets.setup_slides import (GPU_DOCTOR_HINT, GPU_NO_INK,
                                           GPU_NOTE_BAND, GPU_REQUIREMENT,
                                           GPU_YES_INK, GREETING_BAND,
                                           SetupSlides, graphics_card)


@pytest.fixture
def slides(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    made = SetupSlides()
    qtbot.addWidget(made)
    made.show()
    qtbot.waitExposed(made)
    return made


def test_the_note_says_what_a_gpu_is_needed_for(slides):
    text = slides._gpu_note.text()

    assert "NVIDIA" in text
    assert "egmentation" in text and "lassification" in text
    # And that it is not a gate: everything else runs without one.
    assert "without" in text.lower()


def test_a_usable_card_is_named_in_green(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(setup_slides, "graphics_card",
                        lambda: (True, "NVIDIA GeForce RTX 3090"))
    made = SetupSlides()
    qtbot.addWidget(made)

    text = made._gpu_note.text()
    assert "NVIDIA GeForce RTX 3090" in text
    assert GPU_YES_INK.lower() in text.lower()
    assert GPU_NO_INK.lower() not in text.lower()


def test_an_unusable_card_is_still_named_in_red(qtbot, tmp_path, monkeypatch):
    """The card is named even when it cannot be used: "no compatible GPU"
    beside the name of the card the machine has is a different message
    from "none detected", and the fix differs too."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(setup_slides, "graphics_card",
                        lambda: (False, "AMD Radeon RX 7900"))
    made = SetupSlides()
    qtbot.addWidget(made)

    text = made._gpu_note.text()
    assert "AMD Radeon RX 7900" in text
    assert GPU_NO_INK.lower() in text.lower()


def test_no_card_at_all_says_so(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(setup_slides, "graphics_card", lambda: (False, ""))
    made = SetupSlides()
    qtbot.addWidget(made)

    text = made._gpu_note.text()
    assert "detected" in text.lower()
    assert GPU_NO_INK.lower() in text.lower()


def test_the_greeting_sits_above_the_note(slides):
    """Two bands, not one. The greeting fades in and out over the card, so
    an overlap would drop a word on top of the sentence."""
    assert GREETING_BAND < GPU_NOTE_BAND

    slides._place_the_greeting()
    greeting = slides._greeting.geometry()
    note = slides._gpu_note.geometry()

    assert greeting.bottom() <= note.top()


def test_the_detection_says_both_things(qtbot):
    """`(usable, name)` -- whether it can run, and what it is. They are
    different questions: a card torch cannot reach is still a card, and
    the answer there is a CUDA build rather than new hardware."""
    usable, name = graphics_card()

    assert isinstance(usable, bool)
    assert isinstance(name, str)
    if usable:
        assert name, "a usable card must be named"


def test_the_sentence_is_translatable(slides):
    from spacr.qt import i18n

    for source in (GPU_REQUIREMENT, "Compatible GPU", "No compatible GPU"):
        assert i18n.has_translation(source), f"no catalog entry for {source!r}"
        assert i18n.tr(source, "sv") != source


def test_a_card_torch_cannot_reach_is_sent_to_the_doctor(qtbot, tmp_path,
                                                         monkeypatch):
    """A CUDA problem, not a hardware one, and the reader should not have
    to guess that from a red line.

    `spacr-doctor` exists for exactly this case: a CPU-only torch build
    and a driver older than the CUDA runtime torch was built against both
    present as "cuda not available" and need different fixes.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(setup_slides, "graphics_card",
                        lambda: (False, "NVIDIA GeForce RTX 3090"))
    made = SetupSlides()
    qtbot.addWidget(made)

    text = made._gpu_note.text()
    assert "spacr-doctor" in text
    assert "NVIDIA GeForce RTX 3090" in text


def test_no_card_at_all_is_not_sent_to_the_doctor(qtbot, tmp_path,
                                                  monkeypatch):
    """There is nothing for it to diagnose: the fix is hardware."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(setup_slides, "graphics_card", lambda: (False, ""))
    made = SetupSlides()
    qtbot.addWidget(made)

    assert "spacr-doctor" not in made._gpu_note.text()


def test_a_working_card_is_not_sent_to_the_doctor(qtbot, tmp_path,
                                                  monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(setup_slides, "graphics_card",
                        lambda: (True, "NVIDIA GeForce RTX 3090"))
    made = SetupSlides()
    qtbot.addWidget(made)

    assert "spacr-doctor" not in made._gpu_note.text()


def test_the_command_it_names_is_one_that_exists():
    """The hint is only useful if the command is real -- `spacr-doctor`,
    which setup.py installs as a console script."""
    import pathlib

    setup = pathlib.Path(__file__).resolve().parents[2] / "setup.py"
    assert "spacr-doctor=spacr.doctor:main" in setup.read_text()
    assert "spacr-doctor" in GPU_DOCTOR_HINT
