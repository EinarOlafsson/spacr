"""Plaque Assay's drop helpers on the input a tidy drop never has.

``test_517``/``test_518``/``test_526`` drive the policy with ordinary folders
of images and PDFs. These cases are the rest: a folder that cannot be listed,
names already taken, a share that refuses links, a screen whose mode can
only be read off its form, and a PDF that cannot start reading.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr.qt import dnd_handlers as dh
from spacr.qt.widgets.plaque_preview import FIGURE_MODE, PLAQUE_MODE


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((4, 4), np.uint8))
    return path


def _refuse_scandir(monkeypatch, refused: Path):
    real = os.scandir

    def scandir(path="."):
        if Path(path) == refused:
            raise PermissionError(13, "denied", str(path))
        return real(path)

    monkeypatch.setattr(dh.os, "scandir", scandir)


def test_plaque_inputs_sorts_files_folders_and_what_is_neither(tmp_path):
    loose = _image(tmp_path / "loose.tif")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    folder = tmp_path / "plate"
    _image(folder / "b.tif")
    _image(folder / "a.tif")
    (folder / "notes.txt").write_text("x")
    (folder / "sub.tif").mkdir()

    pdfs, images, papers = dh.plaque_inputs(
        [loose, pdf, folder, tmp_path / "gone", tmp_path / "note.txt"])

    assert pdfs == [pdf]
    assert images == [loose, folder / "a.tif", folder / "b.tif"]
    assert papers == []


def test_plaque_inputs_stops_at_the_limit_and_skips_unlistable_folders(
        tmp_path, monkeypatch):
    many = tmp_path / "many"
    for name in ("a", "b", "c"):
        _image(many / f"{name}.tif")
    locked = tmp_path / "locked"
    locked.mkdir()

    _, images, _ = dh.plaque_inputs([many], limit=1)
    assert len(images) == 1

    _refuse_scandir(monkeypatch, locked)
    assert dh.plaque_inputs([locked]) == ([], [], [])
    assert dh.plaque_others([locked]) == 0


def test_a_folder_whose_papers_cannot_be_listed_holds_none(
        tmp_path, monkeypatch):
    import spacr.plaque_papers as papers

    def unreadable(folder):
        raise OSError("share went away")

    monkeypatch.setattr(papers, "figure_folders", unreadable)
    assert dh._holds_papers(tmp_path) is False


def test_plaque_others_counts_only_visible_unusable_files(tmp_path):
    folder = tmp_path / "drop"
    _image(folder / "a.tif")
    (folder / "readme.txt").write_text("x")
    (folder / ".DS_Store").write_text("x")
    (folder / "sub").mkdir()
    stray = tmp_path / "stray.csv"
    stray.write_text("x")

    assert dh.plaque_others([folder, stray, _image(tmp_path / "b.tif"),
                             tmp_path / "gone"]) == 2
    assert dh.plaque_others([folder], limit=0) == 0


def test_pdfs_in_a_file_or_an_unlistable_folder_are_none(
        tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert dh._pdfs_in(pdf) == []
    assert dh._pdfs_in(tmp_path) == [pdf]

    def refuse(self):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(Path, "iterdir", refuse)
    assert dh._pdfs_in(tmp_path) == []


class _Reader:
    def __init__(self, value=None, error=None):
        self._value, self._error = value, error

    def get_value(self):
        if self._error:
            raise self._error
        return self._value

    def text(self):
        return self._value


def _screen(panel=None, widget=None, broken_model=False):
    model = (None if broken_model else
             SimpleNamespace(_widgets={"plaque_mode": widget}))
    return SimpleNamespace(_live_preview=panel, _settings_model=model)


def test_the_mode_is_read_off_the_form_when_the_panel_cannot_say():
    def broken_mode():
        raise RuntimeError("panel deleted")

    panel = SimpleNamespace(mode=broken_mode)
    assert dh._plaque_mode_of(
        _screen(panel, _Reader("Figure"))) == FIGURE_MODE
    assert dh._plaque_mode_of(
        _screen(None, _Reader("figure", error=ValueError("x")))) \
        == FIGURE_MODE
    assert dh._plaque_mode_of(_screen(None, None)) == PLAQUE_MODE
    assert dh._plaque_mode_of(_screen(None, broken_model=True)) == PLAQUE_MODE


def test_a_selection_folder_takes_the_next_free_name_and_keeps_both_twins(
        tmp_path):
    first = _image(tmp_path / "one" / "a.tif")
    twin = _image(tmp_path / "two" / "a.tif")
    (tmp_path / "one" / "plaque_selection_now").mkdir()

    folder = dh.plaque_selection_folder([first, twin, first, twin],
                                        stamp="now")

    assert folder == tmp_path / "one" / "plaque_selection_now_1"
    assert sorted(p.name for p in folder.iterdir()) == [
        "0003_a.tif", "a.tif", "one_a.tif", "two_a.tif"]


def test_a_selection_beside_an_unwritable_folder_goes_to_temp(
        tmp_path, monkeypatch):
    image = _image(tmp_path / "one" / "a.tif")
    real_mkdir = Path.mkdir

    def refuse(self, *args, **kwargs):
        if self.parent == image.parent:
            raise PermissionError(13, "read-only share")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refuse)
    folder = dh.plaque_selection_folder([image], stamp="t")
    try:
        assert folder.parent != image.parent
        assert folder.name.startswith("plaque_selection_t_")
        assert (folder / "a.tif").exists()
    finally:
        for child in folder.iterdir():
            child.unlink()
        folder.rmdir()


def test_links_fall_back_to_hard_links_then_say_what_to_drop(
        tmp_path, monkeypatch):
    image = _image(tmp_path / "one" / "a.tif")

    def no_symlink(*args, **kwargs):
        raise OSError("symlinks not allowed")

    monkeypatch.setattr(dh.os, "symlink", no_symlink)
    folder = dh.plaque_selection_folder([image], stamp="hard")
    linked = folder / "a.tif"
    assert not linked.is_symlink()
    assert os.path.samefile(linked, image)

    def no_link(*args, **kwargs):
        raise OSError("links not allowed")

    monkeypatch.setattr(dh.os, "link", no_link)
    with pytest.raises(OSError, match="Drop the folder that holds"):
        dh.plaque_selection_folder([image], stamp="none")


def test_alternatives_are_sibling_and_child_folders_with_images(tmp_path):
    handler = dh.PlaqueDropHandler()
    dropped = tmp_path / "dropped"
    dropped.mkdir()
    _image(dropped / "inner" / "a.tif")
    _image(tmp_path / "sibling" / "b.tif")
    (tmp_path / "empty").mkdir()

    hits = handler.suggest_alternatives(dropped)

    assert hits == [tmp_path / "sibling", dropped / "inner"]
    assert handler.suggest_alternatives(tmp_path / "sibling" / "b.tif") == []


def test_alternatives_skip_a_place_that_cannot_be_listed(
        tmp_path, monkeypatch):
    handler = dh.PlaqueDropHandler()
    dropped = tmp_path / "dropped"
    _image(dropped / "inner" / "a.tif")
    real = Path.iterdir

    def iterdir(self):
        if self == tmp_path:
            raise PermissionError(13, "denied")
        return real(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert handler.suggest_alternatives(dropped) == [dropped / "inner"]


def test_a_pdf_that_cannot_start_reading_is_reported(tmp_path, monkeypatch):
    import spacr.qt.dnd as dnd

    reported = []
    monkeypatch.setattr(dnd, "_report_drop_problem",
                        lambda screen, path, what, how: reported.append(what))
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]

    busy = SimpleNamespace(mode=lambda: FIGURE_MODE,
                           fetch_papers=lambda *a: False)
    dh.PlaqueDropHandler._take_pdfs(pdfs, _screen(busy))
    assert reported == ["Plaque Assay could not start reading this PDF."]

    reported.clear()
    started = []
    ready = SimpleNamespace(mode=lambda: FIGURE_MODE,
                            fetch_papers=lambda *a: started.append(a) or True)
    dh.PlaqueDropHandler._take_pdfs(pdfs, _screen(ready))
    assert reported == []
    assert started == [(pdfs, str(tmp_path))]


def test_a_pdf_dropped_in_plaque_mode_asks_for_figure_mode(
        tmp_path, monkeypatch):
    import spacr.qt.dnd as dnd

    reported = []
    monkeypatch.setattr(dnd, "_report_drop_problem",
                        lambda screen, path, what, how: reported.append(how))
    started = []
    panel = SimpleNamespace(mode=lambda: PLAQUE_MODE,
                            fetch_paper=lambda *a: started.append(a) or True)

    dh.PlaqueDropHandler._take_pdfs([tmp_path / "a.pdf"], _screen(panel))

    assert started == []
    assert reported == ["Switch Plaque Assay to Figure mode, then drop the "
                        "PDF again."]


def test_a_paper_folder_dropped_in_figure_mode_becomes_the_source(
        tmp_path, monkeypatch):
    import spacr.qt.widgets.plaque_preview as pp
    from spacr.plaque_papers import LEGENDS_FILE

    paper = tmp_path / "smith_2020"
    figure = _image(paper / "figures" / "fig1.tif")
    (paper / LEGENDS_FILE).write_text("file,legend\nfig1.tif,Plaques\n")
    monkeypatch.setattr(pp, "follow_the_input",
                        lambda screen, pdfs, images, papers, name: FIGURE_MODE)
    taken = []
    monkeypatch.setattr(dh.PlaqueDropHandler, "_take_images",
                        staticmethod(lambda paths, screen: taken.append(paths)))

    dh.PlaqueDropHandler().apply(paper, _screen())

    assert taken == [[paper]]
    assert dh._plaque_images_in(paper) == []
    assert dh._plaque_images_in(figure) == []


def test_the_cellpose_screens_take_one_folder_of_images(tmp_path):
    handler = dh.CellposeFolderDropHandler()
    folder = tmp_path / "train"
    image = _image(folder / "a.tif")

    assert handler.accepts_multiple() is False
    assert handler.can_accept(folder)
    assert not handler.can_accept(image)
    assert handler.apply_all([folder], _screen()) is False
    assert "folder of images" in handler.error_message(image)


def test_make_masks_opens_a_dropped_path_in_its_queue(tmp_path):
    opened, logged = [], []
    screen = SimpleNamespace(
        open_paths=opened.append,
        _console=SimpleNamespace(append_stdout=logged.append))

    dh.MakeMasksDropHandler().apply(tmp_path, screen)

    assert opened == [[str(tmp_path)]]
    assert logged == [f"[drop] make_masks queue = {tmp_path}\n"]


class _Unstatable(type(Path())):
    """A path whose stat fails the way a share that went away fails."""

    fail_on = ("is_dir",)

    def is_dir(self):
        if "is_dir" in self.fail_on:
            raise OSError("share went away")
        return super().is_dir()

    def is_file(self):
        if "is_file" in self.fail_on:
            raise OSError("share went away")
        return super().is_file()


class _FileGone(_Unstatable):
    fail_on = ("is_file",)


def test_a_mask_drop_that_cannot_be_examined_is_refused_quietly(tmp_path):
    nothing = {"is_dir": False, "is_file": False, "accepted": False,
               "alternatives": []}
    assert dh.scan_mask_drop(_Unstatable(tmp_path / "gone")) == nothing
    assert dh.scan_mask_drop(_FileGone(tmp_path / "a.tif")) == nothing


def test_a_mask_drop_of_a_folder_whose_listing_fails(tmp_path, monkeypatch):
    folder = tmp_path / "empty"
    folder.mkdir()

    def unreadable(path):
        raise OSError("listing failed")

    monkeypatch.setattr(dh, "has_images_in", unreadable)
    monkeypatch.setattr(dh, "find_image_folders_nearby", unreadable)
    assert dh.scan_mask_drop(str(folder)) == {
        "is_dir": True, "is_file": False, "accepted": False,
        "alternatives": []}


def test_a_mask_file_is_accepted_only_when_it_can_be_described(
        tmp_path, monkeypatch):
    import spacr.qt.multi_format as multi_format

    image = _image(tmp_path / "a.tif")
    other = tmp_path / "a.txt"
    other.write_text("x")
    monkeypatch.setattr(multi_format, "describe_file", lambda path: object())
    assert dh.scan_mask_drop(image)["accepted"] is True
    assert dh.scan_mask_drop(other)["accepted"] is False
    monkeypatch.setattr(multi_format, "describe_file", lambda path: None)
    assert dh.scan_mask_drop(image)["accepted"] is False

    def broken(path):
        raise ValueError("unreadable header")

    monkeypatch.setattr(multi_format, "describe_file", broken)
    assert dh.scan_mask_drop(image)["accepted"] is False


def test_a_decision_that_raises_off_the_gui_thread_answers_the_default(
        monkeypatch):
    monkeypatch.setattr(dh, "_on_gui_thread", lambda: False)

    def work():
        raise OSError("stat failed")

    key = ("test", "raises")
    assert dh._decide(key, work, default="guess") == "guess"
    assert key not in dh._decisions


def test_the_decision_cache_drops_expired_then_soonest_expiring(monkeypatch):
    monkeypatch.setattr(dh, "_decisions", {})
    monkeypatch.setattr(dh, "_DECISION_CAP", 2)
    dh._decisions[("old",)] = (0.0, "expired", 1)
    dh._decisions[("kept",)] = (10.0 ** 12, "kept", 2)

    dh._remember(("b",), "b", 3)
    assert set(dh._decisions) == {("kept",), ("b",)}

    dh._remember(("c",), "c", 4)
    assert set(dh._decisions) == {("kept",), ("c",)}

    dh._remember(("c",), "late", 1)
    assert dh._decisions[("c",)][1] == "c", "a late answer never overwrites"
