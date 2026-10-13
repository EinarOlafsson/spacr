"""Item 701: Make Masks can skip fields that already have Keep or Discard.

* with Hide judged fields on, Next visits only the unjudged fields and the
  counter counts only them; off, every field;
* a verdict given with the switch on moves to the next unjudged field, and a
  queue with nothing unjudged left says so rather than looping;
* the choice is remembered per folder;
* ``spacr-make-masks --exclude-verdicts`` drops judged fields and reports how
  many, and a queue built that way starts the editor with the switch on.
"""
from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr import cli_make_masks as cli
from spacr.curation_queue import build_queue, mark_state
from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen

KEEP = (0, 2, 5)
DISCARD = (3, 8)
UNJUDGED = [1, 4, 6, 7, 9]


def _judge(folder: Path, index: int, keep: bool) -> None:
    path = str(folder / f"img_{index:02d}.tif")
    engine.record_curation(str(folder), path, path, 1, keep)


def _write_verdicts(folder: Path, rows) -> Path:
    destination = folder / "csv" / "keep_discard.csv"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=engine.CURATION_COLUMNS)
        writer.writeheader()
        for image_path, verdict in rows:
            writer.writerow({"image path": str(image_path), "keep": verdict})
    return destination


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "plaques"
    (root / "masks").mkdir(parents=True)
    image = np.zeros((32, 32), dtype=np.uint16)
    image[8:24, 8:24] = 5000
    for index in range(10):
        imageio.imwrite(root / f"img_{index:02d}.tif", image)
    for index in KEEP:
        _judge(root, index, True)
    for index in DISCARD:
        _judge(root, index, False)
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    return widget


def _walk(screen) -> list:
    visited = [screen._current_index]
    for _ in range(20):
        before = screen._current_index
        screen._on_next()
        if screen._current_index == before:
            break
        visited.append(screen._current_index)
    return visited


def test_the_switch_sits_beside_first_unreviewed(screen):
    row = screen._btn_clear.parentWidget().layout()
    assert screen._btn_hide_judged.isCheckable()
    assert not screen._btn_hide_judged.isChecked()
    assert abs(row.indexOf(screen._btn_hide_judged)
               - row.indexOf(screen._btn_first_unreviewed)) == 1
    assert "csv/keep_discard.csv" in screen._btn_hide_judged.toolTip()


def test_off_next_visits_all_ten(screen):
    assert _walk(screen) == list(range(10))
    assert screen._tally_total.text() == "/10"


def test_on_next_visits_exactly_the_unjudged_five(screen):
    screen._btn_hide_judged.setChecked(True)
    assert screen._tally_total.text() == "/5"
    assert screen._current_index == 1
    assert screen._tally_index.text() == "1"
    assert _walk(screen) == UNJUDGED
    assert screen._tally_index.text() == "5"
    screen._on_prev()
    assert screen._current_index == 7
    assert screen._tally_index.text() == "4"
    screen._tally_index.setText("2")
    screen._on_tally_entered()
    assert screen._current_index == 4


def test_switching_off_restores_every_field_and_counter(screen):
    screen._btn_hide_judged.setChecked(True)
    assert screen._current_index == 1
    screen._btn_hide_judged.setChecked(False)
    assert screen._tally_total.text() == "/10"
    assert screen._tally_index.text() == "2"
    screen._on_prev()
    assert screen._current_index == 0


def test_hiding_cannot_leave_a_field_after_a_failed_save(screen, monkeypatch):
    monkeypatch.setattr(screen, "_save_mask_if_needed", lambda: False)
    screen._btn_hide_judged.setChecked(True)
    assert not screen._btn_hide_judged.isChecked()
    assert not screen._hide_judged
    assert screen._current_index == 0
    assert screen._tally_total.text() == "/10"


def test_hiding_and_navigation_leave_verdict_bytes_unchanged(screen, folder):
    verdicts = folder / "csv" / "keep_discard.csv"
    before = verdicts.read_bytes()
    screen._btn_hide_judged.setChecked(True)
    assert _walk(screen) == UNJUDGED
    screen._on_prev()
    screen._btn_hide_judged.setChecked(False)
    assert verdicts.read_bytes() == before


def test_uncertainty_reorder_keeps_a_judged_first_field_hidden(screen):
    screen._btn_hide_judged.setChecked(True)
    pairs = screen._field_pairs()
    scores = {pair: {"field": 1.0 - index / 10,
                     "n_objects": 0, "n_passes": 1}
              for index, pair in enumerate(pairs)}
    screen._apply_uncertainty_ranking(pairs, scores, model="test")
    assert screen._image_files[0] == "img_00.tif"
    assert screen._image_files[screen._current_index] == "img_01.tif"
    assert screen._tally_total.text() == "/5"


def test_a_verdict_with_the_switch_on_moves_to_the_next_unjudged(screen):
    screen._btn_hide_judged.setChecked(True)
    screen._go_to_index(1)
    screen._on_curate(True)
    assert screen._current_index == 4
    assert screen._tally_total.text() == "/4"


def test_every_field_judged_says_so_and_does_not_loop(screen):
    screen._btn_hide_judged.setChecked(True)
    for index in UNJUDGED[:-1]:
        _judge(Path(screen._folder), index, True)
    screen._go_to_index(9)
    screen._on_curate(False)
    assert screen._current_index == 9
    assert "no unjudged" in screen._status_label.text()
    assert screen._tally_total.text() == "/0"
    screen._on_next()
    screen._on_prev()
    assert screen._current_index == 9


def test_last_field_does_not_claim_earlier_unjudged_fields_are_done(screen):
    screen._btn_hide_judged.setChecked(True)
    screen._go_to_index(9)
    screen._on_curate(False)
    assert screen._current_index == 9
    assert screen._tally_total.text() == "/4"
    assert "no unjudged" not in screen._status_label.text().lower()
    assert "every field" not in screen._status_label.text().lower()
    screen._on_prev()
    assert screen._current_index == 7


def test_the_choice_is_remembered_per_folder(screen, qtbot, folder, tmp_path):
    screen._btn_hide_judged.setChecked(True)
    again = MakeMasksScreen()
    qtbot.addWidget(again)
    assert again._open_folder(str(folder))
    assert again._btn_hide_judged.isChecked()
    assert again._tally_total.text() == "/5"
    assert again._current_index == 1
    assert again._tally_index.text() == "1"
    other = tmp_path / "other"
    other.mkdir()
    imageio.imwrite(other / "a.tif", np.zeros((8, 8), dtype=np.uint16))
    assert again._open_folder(str(other))
    assert not again._btn_hide_judged.isChecked()


def test_exclude_verdicts_dry_run_lists_the_five(folder, capsys):
    assert cli.main(["--folder", str(folder), "--exclude-verdicts",
                     "--dry-run", "--order", "name"]) == 0
    out = capsys.readouterr().out
    assert "5 excluded by Keep/Discard verdict" in out
    for index in UNJUDGED:
        assert f"img_{index:02d}" in out
    for index in KEEP + DISCARD:
        assert f"img_{index:02d}" not in out


def test_without_the_flag_the_queue_keeps_judged_fields(folder):
    queue = build_queue(folder, order="name", announce=lambda _t: None)
    assert len(queue.items) == 10
    assert queue.excluded_by_verdict == 0


def test_a_queue_with_the_flag_starts_the_switch_on(qtbot, qt_theme_applied,
                                                    folder):
    queue = build_queue(folder, order="name", exclude_verdicts=True,
                        announce=lambda _t: None)
    assert [i.stem for i in queue.items] == [f"img_{i:02d}" for i in UNJUDGED]
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget.open_queue(queue)
    assert widget._btn_hide_judged.isChecked()
    assert widget._tally_total.text() == "/5"


def test_sibling_layout_reads_the_images_folder_csv(tmp_path):
    images = tmp_path / "set" / "images"
    (tmp_path / "set" / "masks").mkdir(parents=True)
    images.mkdir()
    for name in ("a", "b"):
        imageio.imwrite(images / f"{name}.tif", np.zeros((8, 8), np.uint16))
    path = str(images / "a.tif")
    engine.record_curation(str(images), path, path, 1, True)
    queue = build_queue(tmp_path / "set", order="name", exclude_verdicts=True,
                        announce=lambda _t: None)
    assert [i.stem for i in queue.items] == ["b"]
    assert queue.excluded_by_verdict == 1


def test_sibling_root_verdicts_are_hidden_on_toggle_and_remembered_reopen(
        qtbot, qt_theme_applied, tmp_path):
    root = tmp_path / "siblings"
    images = root / "images"
    images.mkdir(parents=True)
    (root / "masks").mkdir()
    for name in ("a", "b"):
        imageio.imwrite(images / f"{name}.tif", np.zeros((8, 8), np.uint16))
    engine.record_curation(str(root), str(images / "a.tif"), "", 0, True)
    queue = build_queue(root, order="name", announce=lambda _t: None)
    assert [item.stem for item in queue.items] == ["a", "b"]
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget.open_queue(queue)
    widget._btn_hide_judged.setChecked(True)
    assert widget._current_index == 1
    assert widget._tally_total.text() == "/1"
    again = MakeMasksScreen()
    qtbot.addWidget(again)
    assert again.open_queue(queue)
    assert again._btn_hide_judged.isChecked()
    assert again._current_index == 1
    assert again._tally_index.text() == "1"
    assert again._tally_total.text() == "/1"


def test_mixed_folder_choice_uses_first_folder_after_navigating_into_second(
        qtbot, qt_theme_applied, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        folder.mkdir()
        imageio.imwrite(folder / "img_00.tif", np.zeros((8, 8), np.uint16))
    _judge(first, 0, True)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    paths = [str(first), str(second)]
    assert widget._open_queue(paths)
    widget._on_next()
    assert widget._folder == str(second)
    widget._btn_hide_judged.setChecked(True)
    assert widget._current_index == 1
    again = MakeMasksScreen()
    qtbot.addWidget(again)
    assert again._open_queue(paths)
    assert again._btn_hide_judged.isChecked()
    assert again._current_index == 1
    assert again._tally_total.text() == "/1"
    assert again._open_folder(str(second))
    assert not again._btn_hide_judged.isChecked()


def test_verdict_cache_reuses_unchanged_rows_and_notices_external_rewrite(
        screen, folder, monkeypatch):
    screen._btn_hide_judged.setChecked(True)
    read_curation = engine.read_curation
    reads = []

    def counted_read(owner):
        reads.append(owner)
        return read_curation(owner)

    monkeypatch.setattr(engine, "read_curation", counted_read)
    for _ in range(3):
        screen._sync_queue_tally()
    assert reads == []
    _judge(folder, 1, False)
    reads.clear()
    screen._sync_queue_tally()
    assert reads == [str(folder)]
    assert screen._tally_total.text() == "/4"
    assert screen._visible_indices() == [4, 6, 7, 9]
    for _ in range(3):
        screen._sync_queue_tally()
    assert reads == [str(folder)]


@pytest.mark.parametrize("final_verdict", ["", "pending"])
def test_cli_uses_the_final_row_even_when_it_clears_a_verdict(
        folder, capsys, final_verdict):
    path = folder / "img_00.tif"
    _write_verdicts(folder, [(path, "true"), (path, final_verdict)])
    assert engine.curation_verdict(str(folder), str(path)) is None
    assert cli.main(["--folder", str(folder), "--exclude-verdicts",
                     "--dry-run", "--order", "name"]) == 0
    out = capsys.readouterr().out
    assert "0 excluded by Keep/Discard verdict" in out
    assert "img_00" in out


def test_cli_accepts_the_same_verdict_words_as_the_editor(folder, capsys):
    words = ("true", "1", "yes", "keep", "false", "0", "no", "discard",
             "pending", "")
    _write_verdicts(folder, [(folder / f"img_{i:02d}.tif", f" {word.upper()} ")
                           for i, word in enumerate(words)])
    assert cli.main(["--folder", str(folder), "--exclude-verdicts",
                     "--dry-run", "--order", "name"]) == 0
    out = capsys.readouterr().out
    assert "8 excluded by Keep/Discard verdict" in out
    for index in range(8):
        assert engine.curation_verdict(
            str(folder), str(folder / f"img_{index:02d}.tif")) is not None
        assert f"img_{index:02d}" not in out
    assert "img_08" in out
    assert "img_09" in out


def test_cli_excludes_seg_bundle_stems(tmp_path, capsys):
    root = tmp_path / "bundles"
    root.mkdir()
    image = np.zeros((8, 8), dtype=np.uint16)
    for name in ("a", "b.part", "c"):
        np.save(root / f"{name}_seg.npy", {"img": image, "masks": image})
    _write_verdicts(root, [(root / "a_seg.npy", "keep"),
                           (root / "b.part_seg.npy", "discard")])
    queue = build_queue(root, order="name", exclude_verdicts=True,
                        announce=lambda _t: None)
    assert [item.stem for item in queue.items] == ["c"]
    assert queue.excluded_by_verdict == 2
    assert cli.main(["--folder", str(root), "--exclude-verdicts",
                     "--dry-run", "--order", "name"]) == 0
    out = capsys.readouterr().out
    assert "2 excluded by Keep/Discard verdict" in out
    assert "a_seg.npy" not in out
    assert "b.part_seg.npy" not in out
    assert "1  c" in out.splitlines()


def test_cli_excludes_verdicts_before_limit_without_double_counting_status(
        folder, capsys):
    mark_state(folder, "img_00", "done")
    mark_state(folder, "img_01", "skip")
    queue = build_queue(folder, order="name", exclude_verdicts=True, limit=2,
                        announce=lambda _t: None)
    assert [item.stem for item in queue.items] == ["img_04", "img_06"]
    assert queue.excluded_by_verdict == 4
    assert cli.main(["--folder", str(folder), "--exclude-verdicts",
                     "--dry-run", "--order", "name", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "4 excluded by Keep/Discard verdict" in out
    assert "img_04" in out
    assert "img_06" in out
    assert "img_07" not in out


def test_cli_all_judged_explains_verdicts_without_suggesting_status_deletion(
        folder, capsys):
    for index in UNJUDGED:
        _judge(folder, index, True)
    before = (folder / "csv" / "keep_discard.csv").read_bytes()
    assert cli.main(["--folder", str(folder), "--exclude-verdicts",
                     "--dry-run", "--order", "name"]) == 0
    out = capsys.readouterr().out.lower()
    assert "10 excluded by keep/discard verdict" in out
    nothing_to_open = out.split("nothing to open:", 1)[1]
    assert "verdict" in nothing_to_open
    assert "delete a row" not in nothing_to_open
    assert "already has a state" not in nothing_to_open
    assert (folder / "csv" / "keep_discard.csv").read_bytes() == before
