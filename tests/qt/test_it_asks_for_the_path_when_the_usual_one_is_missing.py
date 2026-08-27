"""The fallback prompts, and the six things every one of them owes.

Instruction 274. These are BACKUPS: the ordinary resolution runs first
every time and this is reached only once it has failed, so a run that works
today gains no dialog.
"""

from __future__ import annotations

import os

import pytest

from spacr.qt import ask_for_the_path as A


@pytest.fixture(autouse=True)
def forget_between_tests():
    A.forget()
    yield
    A.forget()


def _chooser(*answers):
    """A stand-in for the folder dialog that gives these answers in turn."""
    remaining = list(answers)
    asked = []

    def chooser(title, start=""):
        asked.append(title)
        return remaining.pop(0) if remaining else ""

    chooser.asked = asked
    return chooser


def _present(monkeypatch):
    monkeypatch.setattr(A, "somebody_is_there", lambda: True)


# --- never headless --------------------------------------------------------


def test_it_never_blocks_when_nobody_is_there():
    """The one that would hang a pipeline overnight."""
    assert A.somebody_is_there() is False, "a test process is not a person"
    path, why = A.ask_for_a_folder("k", tried="none found", what="Merged")
    assert path is None
    assert "nobody to ask" in why


def test_an_explicit_opt_out_is_honoured(monkeypatch):
    monkeypatch.setenv("SPACR_NO_PROMPTS", "1")
    assert A.somebody_is_there() is False


def test_the_headless_answer_says_the_run_stops():
    _path, why = A.ask_for_a_folder("k", tried="none found", what="Merged")
    assert "stops" in why


# --- says what was tried ---------------------------------------------------


def test_the_prompt_says_what_was_tried_first(monkeypatch, tmp_path):
    _present(monkeypatch)
    chooser = _chooser(str(tmp_path))
    A.ask_for_a_folder("k", tried="no merged folder under /data/plate1",
                       what="Merged folder", chooser=chooser)
    assert "no merged folder under /data/plate1" in chooser.asked[0]


# --- validate before accepting ---------------------------------------------


def test_an_empty_folder_is_refused_and_it_asks_again(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    good = tmp_path / "good"
    good.mkdir()
    (good / "a.tif").write_bytes(b"")
    _present(monkeypatch)

    chooser = _chooser(str(empty), str(good))
    path, _why = A.ask_for_a_folder(
        "k", tried="none found", what="Images",
        validate=A.a_folder_holding(".tif"), chooser=chooser)
    assert path == str(good)
    assert len(chooser.asked) == 2, "it accepted the empty folder"
    assert "holds no .tif" in chooser.asked[1], (
        "the second ask does not say what was wrong with the first")


def test_a_file_is_not_a_folder(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x")
    assert "not a folder" in A.a_folder_holding(".tif")(str(target))


def test_a_folder_with_the_right_file_passes(tmp_path):
    (tmp_path / "a.TIF").write_bytes(b"")
    assert A.a_folder_holding(".tif")(str(tmp_path)) is None


# --- once per run ----------------------------------------------------------


def test_it_asks_once_and_remembers(monkeypatch, tmp_path):
    _present(monkeypatch)
    chooser = _chooser(str(tmp_path))
    first, _ = A.ask_for_a_folder("merged", tried="x", what="Merged",
                                  chooser=chooser)
    second, why = A.ask_for_a_folder("merged", tried="x", what="Merged",
                                     chooser=chooser)
    assert first == second == str(tmp_path)
    assert len(chooser.asked) == 1, "it asked twice in one run"
    assert "chosen earlier" in why


def test_two_different_things_are_asked_separately(monkeypatch, tmp_path):
    _present(monkeypatch)
    chooser = _chooser(str(tmp_path), str(tmp_path))
    A.ask_for_a_folder("merged", tried="x", what="Merged", chooser=chooser)
    A.ask_for_a_folder("images", tried="y", what="Images", chooser=chooser)
    assert len(chooser.asked) == 2


# --- refusable -------------------------------------------------------------


def test_cancelling_gives_the_original_error(monkeypatch):
    _present(monkeypatch)
    path, why = A.ask_for_a_folder("k", tried="none found", what="Merged",
                                   chooser=_chooser(""))
    assert path is None
    assert "cancelled" in why
    assert "would have given anyway" in why


def test_a_cancel_is_not_remembered(monkeypatch, tmp_path):
    """Cancelling once must not silently answer the next run."""
    _present(monkeypatch)
    A.ask_for_a_folder("k", tried="x", what="Merged", chooser=_chooser(""))
    assert A.remembered("k") is None
