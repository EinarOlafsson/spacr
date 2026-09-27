"""The setup slides name the graphics card and survive what they ask.

Pinned here, each as what the user reads on the first slide:

* a card torch can see but cannot use is named and called unusable; with
  the resolver unavailable, torch's own answer names the card; with torch
  unable to reach any card, NVML still names it;
* the accelerator library, the verdict colour, the Cellpose version and
  the capability table each fall back to something readable when what
  they ask cannot answer; a neural engine is listed as not used;
* signing in falls back to a terminal when the in-app window cannot be
  built, and closing a sign-in window that has already gone is quiet;
* the GPU note stays on the card when its height, the buttons' position
  or the buttons' size cannot be read, and when there are no buttons.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

from spacr import accelerator as acc
from spacr.qt.widgets import setup_slides as S

pytestmark = pytest.mark.qt


def _broken(*_a, **_k):
    raise RuntimeError("not available here")


# ---------------------------------------------------------------------------
# Naming the card
# ---------------------------------------------------------------------------

def test_a_card_torch_cannot_use_is_named_and_unusable(monkeypatch):
    pytest.importorskip("torch")
    found = acc.Accelerator(kind="cuda", device="cuda", label="GTX (CUDA)",
                            name="GTX 1080", detected=True, usable=False)
    monkeypatch.setattr(acc, "inspect_torch", lambda _torch: found)
    assert S.graphics_card() == (False, "GTX 1080")


def test_torch_names_the_card_when_the_resolver_cannot(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(acc, "inspect_torch", _broken)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda _i: "NVIDIA RTX A6000")
    assert S.graphics_card() == (True, "NVIDIA RTX A6000")


@pytest.mark.parametrize("raw", [b"Tesla T4", "Tesla T4"])
def test_nvml_names_a_card_torch_cannot_reach(monkeypatch, raw):
    torch = pytest.importorskip("torch")
    from spacr.qt.widgets import home

    monkeypatch.setattr(acc, "inspect_torch", _broken)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    nvml = types.SimpleNamespace(
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: ("handle", i),
        nvmlDeviceGetName=lambda handle: raw)
    monkeypatch.setattr(home, "_nvml", lambda: nvml)
    assert S.graphics_card() == (False, "Tesla T4")


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def test_an_unknown_accelerator_library_is_not_named(monkeypatch):
    monkeypatch.setattr(acc, "resolve", _broken)
    assert S._gpu_library() == ""


def test_the_verdict_keeps_its_own_colours_without_a_palette(monkeypatch):
    from spacr.qt import theme

    monkeypatch.setattr(theme, "active_palette", _broken)
    assert S.verdict_ink(True) == S.GPU_YES_INK
    assert S.verdict_ink(False) == S.GPU_NO_INK


def test_a_cellpose_that_cannot_be_read_is_just_cellpose(monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose", None)
    assert S.SetupSlides._cellpose_label() == "Cellpose"


def test_a_capability_table_that_cannot_be_asked_is_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.accelerator", None)
    assert S.SetupSlides._what_this_machine_can_do() == []


def test_a_capability_table_that_fails_midway_is_empty(monkeypatch):
    monkeypatch.setattr(acc, "capabilities", _broken)
    assert S.SetupSlides._what_this_machine_can_do() == []


def test_a_neural_engine_is_listed_as_not_used(monkeypatch):
    monkeypatch.setattr(acc, "neural_engines", lambda: ("Apple ANE",))
    rows = S.SetupSlides._what_this_machine_can_do()
    assert any("Apple ANE" in row and "not used by spaCR" in row
               for row in rows)


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

@pytest.fixture
def slides(qtbot):
    dialog = S.SetupSlides()
    qtbot.addWidget(dialog)
    dialog.resize(900, 700)
    dialog.show()
    qtbot.waitExposed(dialog)
    return dialog


def test_a_sign_in_window_that_cannot_be_built_uses_a_terminal(slides,
                                                               monkeypatch):
    from spacr.qt.ai import pty_sign_in

    ran = []
    monkeypatch.setattr(pty_sign_in, "pty_available", lambda: True)
    monkeypatch.setattr(pty_sign_in, "SignInDialog", _broken)
    monkeypatch.setattr(slides, "_run_in_a_terminal",
                        lambda command: ran.append(command) or True)
    provider = types.SimpleNamespace(label="GitHub")
    assert slides._sign_in_here_or_in_a_terminal(provider,
                                                 "gh auth login") is True
    assert ran == ["gh auth login"]
    assert slides._sign_in_dialog is None


def test_closing_a_sign_in_window_that_has_gone_is_quiet(slides):
    class Gone:
        def close(self):
            raise RuntimeError("Internal C++ object already deleted.")

    slides._sign_in_dialog = Gone()
    slides._stop_sign_in()
    assert slides._sign_in_dialog is None


def _on_card(slides):
    note = slides._gpu_note
    return (note.geometry().top() >= 0
            and note.geometry().bottom() <= slides.card.height())


def test_the_note_stays_on_the_card_when_nothing_can_be_measured(
        slides, monkeypatch):
    note = slides._gpu_note
    if note.isHidden():
        pytest.skip("this machine shows no GPU note")
    back = slides._back
    monkeypatch.setattr(note, "heightForWidth", lambda _w: 0)
    monkeypatch.setattr(back, "mapTo", _broken)
    monkeypatch.setattr(back, "sizeHint", _broken)
    slides._place_the_gpu_note()
    assert _on_card(slides)
    assert note.height() == note.sizeHint().height()


def test_the_note_stays_on_the_card_without_buttons(slides, monkeypatch):
    note = slides._gpu_note
    if note.isHidden():
        pytest.skip("this machine shows no GPU note")
    monkeypatch.setattr(slides, "_back", None)
    slides._place_the_gpu_note()
    assert _on_card(slides)
