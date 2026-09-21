"""Plaque figures: detect, read the labels, key the legend, measure (items 424, 468).

The word positions below are what RapidOCR returned for Fig 6 of PMC9744290
(DOI 10.1371/journal.ppat.1011009), panel D: two columns (-MEV, +MEV) over two
rows (-Rapamycin, +Rapamycin, printed rotated), and the four crop boxes the
v2 detector drew. Every label was read correctly on the real figure on
2026-09-21; this pins the geometry that turns positions into conditions.
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import plaque_papers as pp

W = pp.Word
REGIONS = [pp.Region(18, 207, 102, 290), pp.Region(104, 207, 187, 291),
           pp.Region(104, 291, 186, 376), pp.Region(16, 292, 104, 375)]
WORDS = [W("D", 6, 193, 17, 208), W("-MEV", 49, 196, 84, 208),
         W("+MEV", 126, 196, 164, 208), W("-Rapamycin", 6, 216, 22, 284),
         W("+Rapamycin", 5, 297, 21, 368), W("E", 200, 193, 211, 208)]
CAPTION = ("Reconstitution of the MVA pathway.A, reconstitution into the UPRT "
           "locus. B, IFA demonstrating it. C, intracellular replication. D, "
           "plaque assays of the iTPI2compMVA strain with or without MEV and "
           "Rapamycin. E, relative plaque sizes.")


def test_each_crop_gets_its_column_and_row_label():
    got = {(a.row, a.column): a.label_text
           for a in pp.annotate_regions(REGIONS, WORDS, caption=CAPTION)}
    assert got == {(1, 1): "-MEV / -Rapamycin", (1, 2): "+MEV / -Rapamycin",
                   (2, 1): "-MEV / +Rapamycin", (2, 2): "+MEV / +Rapamycin"}


def test_the_panel_letter_keys_into_the_legend():
    for a in pp.annotate_regions(REGIONS, WORDS, caption=CAPTION):
        assert a.panel == "D"
        assert a.legend_text.startswith("plaque assays of the iTPI2compMVA")
        assert a.source == "label+legend" and a.strength == "strong"


def test_a_distant_letter_is_not_taken_as_the_panel():
    words = [w for w in WORDS if w.text != "D"] + [W("C", 6, 20, 17, 35)]
    assert all(a.panel is None for a in pp.annotate_regions(REGIONS, words))


def test_with_nothing_to_read_the_position_is_the_condition_and_weak():
    a = pp.annotate_regions(REGIONS[:1], [], figure_label="Fig 6")[0]
    assert a.source == "position" and a.strength == "weak"
    assert "Fig 6" in a.condition


@pytest.mark.parametrize("caption,expected", [
    ("Title.A, one. B-D, two (B, D) and more. E, five.",
     {"A": "one.", "B": "two (B, D) and more.", "C": "two (B, D) and more.",
      "D": "two (B, D) and more.", "E": "five."}),
    ("Title. (A) first (B) second in P. falciparum.",
     {"A": "first", "B": "second in P. falciparum."}),
])
def test_legends_split_on_letters_in_sequence_only(caption, expected):
    got = pp.split_legend(caption)
    assert {k: v for k, v in got.items() if k} == expected


def test_overrides_round_trip_and_keep_other_figures(tmp_path):
    path = tmp_path / pp.ANNOTATIONS_FILE
    pp.write_annotation_overrides(path, [{"file": "a.png", "region": 1,
                                          "condition": "WT", "approved": True}])
    pp.write_annotation_overrides(path, [{"file": "b.png", "region": 2,
                                          "condition": "KO", "approved": False}])
    got = pp.read_annotation_overrides(path)
    assert got[("a", 1)] == {"condition": "WT", "approved": True}
    assert got[("b", 2)] == {"condition": "KO", "approved": False}


def test_confirm_each_leaves_unapproved_images_out():
    anns = pp.annotate_regions(REGIONS[:2], WORDS)
    pp.apply_overrides("fig", anns, {("fig", 1): {"condition": "edited",
                                                  "approved": True}},
                       confirm_each=True)
    assert anns[0].condition == "edited" and anns[0].approved is True
    assert anns[0].source == "manual"
    assert anns[1].approved is False


def test_a_folder_of_figures_is_measured_into_one_database(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((400, 260, 3), np.uint8)).save(tmp_path / "fig6.png")
    (tmp_path / pp.LEGENDS_FILE).write_text(
        f'file,legend\nfig6.png,"{CAPTION}"\n', encoding="utf-8")

    def detect(image, weights, **kw):
        return [SimpleNamespace(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1,
                                confidence=0.9) for r in REGIONS]

    def segment(crop):
        labels = np.zeros(crop.shape[:2], np.int32)
        labels[5:15, 5:15] = 1
        labels[20:40, 20:40] = 2
        return labels

    summary = pp.measure_figure_folder(
        tmp_path, detector="fake", segmenter="fake", imgsz=(640,),
        read_text=lambda path: WORDS, detect=detect, segment=segment)
    assert summary["regions"] == 4 and summary["plaques"] == 8
    with sqlite3.connect(summary["database"]) as db:
        conditions = sorted(r[0] for r in db.execute("SELECT condition FROM regions"))
        ratios = sorted(r[0] for r in db.execute(
            "SELECT area_vs_panel_median FROM plaques"))
    assert conditions[0] == "+MEV / +Rapamycin"
    assert ratios[0] == pytest.approx(100 / 250) and ratios[-1] == pytest.approx(400 / 250)
