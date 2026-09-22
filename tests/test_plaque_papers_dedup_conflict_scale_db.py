"""Item 424's remaining open questions, answered in the engine.

* DEDUPLICATION by paper plus image hash: a figure seen twice is measured
  once, and the second sighting is recorded in ``duplicates``, not dropped.
* THE TWO CONDITION STRATEGIES DISAGREEING: flagged, never resolved.
* SCALE: a scale bar, a legend's stated bar length or plate format, or
  "sizes in pixels" -- and a stated magnification is never a ruler.
* THE PDF TEXT LAYER comes before OCR, and survives the trip through a
  folder of figures.
* THE DATABASE: runs, papers, figures, legend passages, every annotation
  (measured or not), regions, plaques and duplicates, in one file, and an
  older file is brought up to date rather than broken.

Every model is faked; the real measurement is in
features/data/424_detector_transfer_merged_2026-09-21.json.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import plaque_papers as pp

W = pp.Word
BOXES = [pp.Region(100, 100, 180, 180), pp.Region(220, 100, 300, 180)]
WORDS = [W("A", 60, 60, 75, 75), W("WT", 120, 80, 160, 95),
         W("KO", 240, 80, 280, 95)]


def _detect(image, weights, **kw):
    return [SimpleNamespace(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1, confidence=0.9)
            for r in BOXES]


def _segment(crop):
    labels = np.zeros(crop.shape[:2], np.int32)
    labels[5:15, 5:15] = 1
    return labels


def _figure(seed=0, shape=(400, 400)):
    """A figure with structure a thumbnail keeps, different per seed."""
    rng = np.random.default_rng(seed)
    image = np.full(shape + (3,), 235, np.uint8)
    for _ in range(12):
        y0, x0 = rng.integers(0, shape[0] - 60), rng.integers(0, shape[1] - 60)
        h, w = rng.integers(20, 120, 2)
        image[y0:y0 + h, x0:x0 + w] = rng.integers(0, 200, 3)
    return image


def _save(array, path):
    from PIL import Image

    Image.fromarray(array).save(path)
    return path


def _measure_folder(folder, **kw):
    return pp.measure_figure_folder(
        folder, detector="fake", segmenter="fake", imgsz=(640,),
        read_text=kw.pop("read_text", lambda path: WORDS), detect=_detect,
        segment=_segment, **kw)


def _rows(database, sql, *args):
    with sqlite3.connect(database) as db:
        return db.execute(sql, args).fetchall()


def test_a_legend_naming_the_other_images_conditions_is_a_conflict():
    caption = "Plaques.A, plaque assay of KO parasites."
    wt, ko = pp.annotate_regions(BOXES, WORDS, caption=caption)
    assert wt.label_text == "WT" and wt.conflict is True
    assert wt.conflict_terms == ["ko"] and "names ko" in pp._conflict_reason(wt)
    assert ko.conflict is False and ko.strength == "strong"


def test_a_legend_that_names_no_condition_is_not_a_conflict():
    caption = "Plaques.A, plaque assays under the indicated conditions."
    found = pp.annotate_regions(BOXES, WORDS, caption=caption)
    assert [a.conflict for a in found] == [False, False]
    assert {a.strength for a in found} == {"medium"}
    assert all(pp._conflict_reason(a) == "" for a in found)


def test_the_annotations_file_carries_the_conflict_and_reads_back(tmp_path):
    caption = "Plaques.A, plaque assay of KO parasites."
    found = pp.annotate_regions(BOXES, WORDS, caption=caption)
    path = pp.write_annotation_overrides(
        tmp_path / pp.ANNOTATIONS_FILE,
        [pp._annotation_file_row("fig.png", i, a) for i, a in
         enumerate(found, start=1)])
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == pp.ANNOTATION_COLUMNS
    assert rows[0]["conflict"] == "true" and "ko" in rows[0]["conflict_reason"]
    assert rows[1]["conflict"] == "false"
    got = pp.read_annotation_overrides(path)
    assert got[("fig", 1)] == {"condition": "WT", "approved": False}


def test_the_same_image_twice_in_a_folder_is_measured_once(tmp_path):
    image = _figure(1)
    _save(image, tmp_path / "a.png")
    _save(image, tmp_path / "b.png")
    _save(image, tmp_path / "c.bmp")
    _save(_figure(2), tmp_path / "d.png")
    summary = _measure_folder(tmp_path)
    assert summary["duplicates"] == 2
    assert summary["regions"] == 4, "a and d measured, b and c not"
    dups = _rows(summary["database"],
                 "SELECT path, match, measured, duplicate_of_path FROM duplicates "
                 "ORDER BY path")
    assert [(Path(p).name, m, k) for p, m, k, _ in dups] == [
        ("b.png", "bytes", 0), ("c.bmp", "pixels", 0)]
    assert all(Path(of).name == "a.png" for *_, of in dups)
    again = _measure_folder(tmp_path)
    assert again["regions"] == 4 and again["duplicates"] == 2
    assert _rows(summary["database"], "SELECT COUNT(*) FROM regions")[0][0] == 4
    assert _rows(summary["database"], "SELECT COUNT(*) FROM duplicates")[0][0] == 2


def test_a_resized_copy_is_recorded_as_possible_and_still_measured(tmp_path):
    from PIL import Image

    image = _figure(3)
    _save(image, tmp_path / "a.png")
    small = np.asarray(Image.fromarray(image).resize((380, 380), Image.BILINEAR))
    _save(small, tmp_path / "b.png")
    summary = _measure_folder(tmp_path)
    assert summary["possible_duplicates"] == 1 and summary["duplicates"] == 0
    assert summary["regions"] == 4
    (match, measured), = _rows(summary["database"],
                               "SELECT match, measured FROM duplicates")
    assert (match, measured) == ("similar", 1)


class _Response:
    def __init__(self, content=b"", payload=None):
        self.status_code, self.content, self._payload = 200, content, payload

    def json(self):
        return self._payload


def _epmc(bundles):
    """A Europe PMC stand-in: ``{doi: (pmcid, png bytes)}``."""
    def get(url, params=None, **kw):
        if url.endswith("/search"):
            query = params["query"]
            for doi, (pmcid, _png) in bundles.items():
                if doi in query or pmcid in query:
                    return _Response(payload={"resultList": {"result": [{
                        "pmcid": pmcid, "doi": doi, "license": "cc by"}]}})
            return _Response(payload={"resultList": {"result": []}})
        for doi, (pmcid, png) in bundles.items():
            if pmcid in url and url.endswith("supplementaryFiles"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    z.writestr(f"{pmcid}.g001.png", png)
                return _Response(buf.getvalue())
        return _Response(b"")
    return get


def _png_bytes(array):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _measure_papers(refs, dst, get, **kw):
    return pp.measure_plaques_from_papers(
        refs, dst, detector="fake", segmenter="fake", imgsz=(640,),
        read_text=lambda path: WORDS, detect=_detect, segment=_segment,
        ask_legend=lambda *a: None, review=lambda f, a: a, get=get, **kw)


def test_a_preprint_and_its_paper_share_a_figure_once(tmp_path):
    png = _png_bytes(_figure(4))
    get = _epmc({"10.1101/pre.1": ("PMC1", png), "10.1371/pub.1": ("PMC2", png)})
    summary = _measure_papers(["10.1101/pre.1", "10.1371/pub.1"], tmp_path, get)
    assert summary["papers"] == 2 and summary["duplicates"] == 1
    assert summary["regions"] == 2
    (paper, match, of_paper), = _rows(
        summary["database"],
        "SELECT paper_key, match, duplicate_of_paper_key FROM duplicates")
    assert (paper, match, of_paper) == ("10.1371/pub.1", "bytes", "10.1101/pre.1")
    again = _measure_papers(["10.1101/pre.1"], tmp_path, get)
    assert again["skipped_figures"] == 1 and again["duplicates"] == 0


def _pdf_with_doi(path, doi):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/doi": doi})
    with open(path, "wb") as handle:
        writer.write(handle)
    return path


class _Page:
    def __init__(self, image, words, text):
        self._image, self._words, self._text = image, words, text

    def to_image(self, resolution=200):
        image = self._image
        return SimpleNamespace(save=lambda target: _save(image, target))

    def extract_text(self, **kw):
        return self._text

    def extract_words(self, **kw):
        return self._words


def _opener(image, words, text="Fig 1. Plaques.A, WT and KO plaques."):
    class _Doc:
        pages = [_Page(image, words, text)]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    return lambda path: _Doc()


def test_a_pdf_names_its_doi_and_is_one_paper_with_its_europepmc_copy(tmp_path):
    pdf = _pdf_with_doi(tmp_path / "paper.pdf", "10.1371/pub.2")
    assert pp._doi_in_pdf(pdf) == ("10.1371/pub.2", "pdf metadata")
    get = _epmc({"10.1371/pub.2": ("PMC3", _png_bytes(_figure(5)))})
    first = _measure_papers(["10.1371/pub.2"], tmp_path / "out", get)
    assert first["regions"] == 2
    second = _measure_papers([str(pdf)], tmp_path / "out", get,
                             pdf_opener=_opener(_figure(6), []))
    assert second["duplicates"] == 1 and second["regions"] == 0
    (match, of_path), = _rows(second["database"],
                              "SELECT match, duplicate_of_path FROM duplicates")
    assert match == "paper" and "europepmc" in of_path


def test_a_doi_the_text_layer_broke_across_lines_is_put_back_together():
    """The first page of the PLOS Pathogens PDF of PMC9744290, as pypdf reads
    it on 2026-09-21: taken as printed, the DOI was ``10.1371/j``."""
    text = ("Citation: PLoS Pathog 18(11):\ne1011009. https://d oi.org/10.1371/j "
            "ournal.\nppat.1011009\nEditor: Dominique Soldati-Favre, University of")
    assert pp._doi_candidates(text)[0] == "10.1371/journal.ppat.1011009"
    figures = ("https://doi.org/10.1371/journal.ppat.1011009.g004 and "
               "https://doi.org/10.1371/journal.ppat.1011009.g005 "
               "cites 10.1038/other.1")
    assert pp._doi_candidates(figures)[0] == "10.1371/journal.ppat.1011009"


def test_an_unconfirmed_pdf_doi_is_kept_but_not_used_as_the_key(tmp_path):
    pdf = _pdf_with_doi(tmp_path / "paper.pdf", "10.9999/unknown")
    paper = pp.resolve_paper(pdf, get=_epmc({}))
    assert paper.key.startswith("pdf:") and paper.doi == "10.9999/unknown"
    assert paper.doi_from == "pdf metadata, unconfirmed"


def _bar_figure():
    """Two 200 px crops side by side; a 50 px black bar under '1 mm' in the first."""
    image = np.full((400, 640, 3), 255, np.uint8)
    image[100:300, 100:300] = 128
    image[100:300, 305:505] = 128
    image[280:283, 220:270] = 0
    return image, [pp.Region(100, 100, 300, 300), pp.Region(305, 100, 505, 300)]


def test_a_labelled_scale_bar_is_the_ruler_for_its_panel():
    image, regions = _bar_figure()
    words = [W("1 mm", 222, 262, 268, 274)]
    first, second = pp._scales_for_regions(image, regions, words)
    assert first.source == "scale bar" and first.px_per_mm == pytest.approx(50.0)
    assert second.source == "scale bar, same panel"
    assert second.px_per_mm == pytest.approx(50.0) and second.unit == "mm2"


def test_a_number_and_unit_read_as_two_words_are_one_length():
    (word, mm), = pp._scale_labels([W("500", 10, 10, 30, 20), W("µm", 33, 10, 50, 20)])
    assert word.text == "500 µm" and mm == pytest.approx(0.5)


def test_a_legend_stated_bar_length_measures_an_unlabelled_bar():
    image, regions = _bar_figure()
    scales = pp._scales_for_regions(image, regions[:1], [],
                                   caption="Plaques. _Scale bar, 500 µm.")
    assert scales[0].source == "scale bar, length from legend"
    assert scales[0].px_per_mm == pytest.approx(100.0)


def test_a_whole_well_takes_the_plate_format_the_legend_names_or_the_settings():
    image = np.full((400, 400, 3), 200, np.uint8)
    well = [pp.Region(50, 50, 250, 250)]
    legend = "Plaques in 6-well plates, imaged with a 10x objective."
    (scale,) = pp._scales_for_regions(image, well, [], caption=legend)
    assert scale.source == "well: 6-well (legend)"
    assert scale.px_per_mm == pytest.approx(200 / 34.8)
    assert scale.magnification == "10x objective"
    (scale,) = pp._scales_for_regions(image, well, [], caption=legend,
                                     plate_format="12-well")
    assert scale.source == "well: 12-well (settings)"


def test_with_no_ruler_the_sizes_are_in_pixels_and_magnification_is_not_one():
    image = np.full((400, 400, 3), 200, np.uint8)
    (scale,) = pp._scales_for_regions(image, [pp.Region(50, 50, 350, 150)], [],
                                     caption="Imaged at 20x magnification.")
    assert scale.px_per_mm is None and scale.unit == "px"
    assert scale.source == "none" and scale.magnification


def test_the_run_writes_the_scale_and_unit_of_every_image(tmp_path):
    image, _regions = _bar_figure()
    _save(image, tmp_path / "fig.png")

    def detect(img, weights, **kw):
        return [SimpleNamespace(x0=100, y0=100, x1=300, y1=300, confidence=0.9),
                SimpleNamespace(x0=305, y0=100, x1=505, y1=300, confidence=0.9)]
    summary = pp.measure_figure_folder(
        tmp_path, detector="fake", segmenter="fake", imgsz=(640,),
        read_text=lambda p: [W("1 mm", 222, 262, 268, 274)], detect=detect,
        segment=_segment)
    got = _rows(summary["database"],
                "SELECT scale_source, size_unit, px_per_mm FROM regions "
                "ORDER BY region_index")
    assert got == [("scale bar", "mm2", 50.0), ("scale bar, same panel", "mm2", 50.0)]
    (mm2,) = {r[0] for r in _rows(summary["database"], "SELECT area_mm2 FROM plaques")}
    assert mm2 == pytest.approx(100 / 2500)
    assert summary["with_ruler"] == 2


def test_the_text_layer_is_read_before_ocr_and_ocr_only_when_it_is_silent(tmp_path):
    image = _figure(7)
    layer = [W("A", 60, 60, 75, 75), W("WT", 120, 80, 160, 95),
             W("KO", 240, 80, 280, 95)]

    def no_ocr(path):
        raise AssertionError("OCR must not run when the text layer answers")
    words, source = pp._figure_words(image, BOXES, layer, path="x",
                                    read_text=no_ocr)
    assert source == "pdf text layer" and words == layer
    body = [W("Introduction", 10, 380, 90, 395)]
    words, source = pp._figure_words(image, BOXES, body, path="x",
                                    read_text=lambda p: WORDS)
    assert source == "pdf text layer + ocr" and len(words) == 4
    assert pp._figure_words(image, BOXES, [], path="x",
                           read_text=lambda p: WORDS)[1] == "ocr"


def test_a_legend_paragraph_under_the_figure_is_not_read_as_its_labels():
    """PMC9744290's PDF, 2026-09-21: the legend under Fig 4 sat next to the
    bottom row of crops and every crop was labelled "4. The cytosolic"."""
    paragraph = [W(t, 100 + 45 * i, 190, 140 + 45 * i, 200) for i, t in enumerate(
        "Fig 4. The cytosolic TPI1 and apicoplast".split())]
    tail = [W(t, 100 + 45 * i, 203, 140 + 45 * i, 213) for i, t in enumerate(
        "three experiments.".split())]
    labels = [W("WT", 120, 80, 160, 95), W("KO", 240, 80, 280, 95)]
    above = [W("+Rapamycin", 100, 60, 125, 187)]
    kept = pp._label_words(paragraph + tail + labels + above)
    assert sorted(w.text for w in kept) == ["+Rapamycin", "KO", "WT"], (
        "a rotated row label just over the legend is a label")
    words, source = pp._figure_words(np.zeros((400, 400, 3), np.uint8), BOXES,
                                    paragraph + tail, path="x",
                                    read_text=lambda p: WORDS)
    assert source == "pdf text layer + ocr"


def test_a_pdfs_text_layer_goes_through_the_folder_to_the_run(tmp_path):
    pdf = _pdf_with_doi(tmp_path / "paper.pdf", "10.9999/none")
    words = [{"text": "A", "x0": 60 * 0.36, "top": 60 * 0.36, "x1": 75 * 0.36,
              "bottom": 75 * 0.36},
             {"text": "WT", "x0": 120 * 0.36, "top": 80 * 0.36,
              "x1": 160 * 0.36, "bottom": 95 * 0.36},
             {"text": "KO", "x0": 240 * 0.36, "top": 80 * 0.36,
              "x1": 280 * 0.36, "bottom": 95 * 0.36}]
    out = pp.fetch_paper_to_folder(pdf, tmp_path / "folder", get=_epmc({}),
                                   pdf_opener=_opener(_figure(8), words))
    assert out["text_layer"] == 1
    layer = pp._read_text_layer(tmp_path / "folder" / pp.TEXT_LAYER_FILE)
    assert [w.text for w in layer["page_001"]] == ["A", "WT", "KO"]

    def no_ocr(path):
        raise AssertionError("OCR must not run when the text layer answers")
    summary = _measure_folder(tmp_path / "folder", read_text=no_ocr)
    assert summary["text_layer_figures"] == 1 and summary["regions"] == 2
    (source,), = _rows(summary["database"], "SELECT words_source FROM figures")
    assert source == "pdf text layer"
    labels = {r[0] for r in _rows(summary["database"], "SELECT label_text FROM regions")}
    assert labels == {"WT", "KO"}
    (key, doi_from), = _rows(summary["database"],
                             "SELECT paper_key, doi_from FROM papers")
    assert key.startswith("pdf:") and doi_from == "pdf metadata, unconfirmed"


def test_without_pdfplumber_here_the_reader_environment_renders_the_pdf(
        tmp_path, monkeypatch):
    seen = {}

    class Worker:
        def request(self, op, **payload):
            seen[op] = payload
            target = Path(payload["dest"]) / "page_001.png"
            _save(_figure(9), target)
            return {"pages": [{"path": str(target),
                               "text": "Fig 2. Legend.A, one.",
                               "words": [["A", 1, 2, 3, 4]]}]}

    monkeypatch.setattr(pp, "_importable", lambda module: False)
    monkeypatch.setattr(pp, "reader_environment", lambda: "/env/papers")
    monkeypatch.setattr("spacr._segmentation_backends._worker_for",
                        lambda name, env: Worker())
    figures = pp.figures_from_pdf(tmp_path / "x.pdf", tmp_path / "pages")
    assert seen["read_pdf"]["dpi"] == 200
    assert figures[0].label == "Fig 2" and figures[0].words[0].text == "A"


def test_the_reader_worker_reads_a_real_pdf(tmp_path):
    pytest.importorskip("pdfplumber")
    from spacr import _segmentation_backends as backends

    pdf = _pdf_with_doi(tmp_path / "p.pdf", "10.1/x")
    reply = backends._worker_read_pdf(
        {"pdf": str(pdf), "dest": str(tmp_path / "out"), "dpi": 72}, {})
    assert len(reply["pages"]) == 1 and Path(reply["pages"][0]["path"]).is_file()
    assert "pdfplumber==0.11.10" in backends._SPECS[backends._PAPERS].requirements


def test_every_annotation_legend_passage_and_run_lands_in_the_database(tmp_path):
    _save(_figure(10), tmp_path / "fig.png")
    (tmp_path / pp.LEGENDS_FILE).write_text(
        'file,legend\nfig.png,"Plaques.A, plaque assay of KO parasites."\n',
        encoding="utf-8")
    pp.write_annotation_overrides(tmp_path / pp.ANNOTATIONS_FILE, [
        {"file": "fig.png", "region": 2, "condition": "KO", "approved": True}])
    summary = _measure_folder(tmp_path, confirm_each=True)
    db = summary["database"]
    assert summary["regions"] == 1 and summary["awaiting_approval"] == 1
    got = _rows(db, "SELECT region_index, condition, conflict, measured, "
                    "region_id IS NOT NULL FROM figure_annotations "
                    "ORDER BY region_index")
    assert got == [(1, "WT", 1, 0, 0), (2, "KO", 0, 1, 1)]
    panels = dict(_rows(db, "SELECT panel, passage FROM legend_panels"))
    assert panels["A"] == "plaque assay of KO parasites."
    (finished, summary_json, detector), = _rows(
        db, "SELECT finished, summary, detector FROM runs")
    assert finished and json.loads(summary_json)["regions"] == 1
    assert detector == "fake"
    run_ids = {r[0] for r in _rows(db, "SELECT run_id FROM regions")} | \
        {r[0] for r in _rows(db, "SELECT run_id FROM figures")}
    assert run_ids == {summary["run_id"]}


def test_the_folder_keeps_the_papers_doi_and_licence(tmp_path):
    _save(_figure(11), tmp_path / "fig.png")
    (tmp_path / pp.PAPER_FILE).write_text(json.dumps({
        "key": "10.1371/x.9", "source": "europepmc", "doi": "10.1371/x.9",
        "pmcid": "PMC9", "licence": "cc by", "unknown_field": 1}))
    summary = _measure_folder(tmp_path)
    assert _rows(summary["database"], "SELECT paper_key, licence FROM papers") \
        == [("10.1371/x.9", "cc by")]
    assert {r[0] for r in _rows(summary["database"],
                                "SELECT paper_key FROM regions")} == {"10.1371/x.9"}


def test_an_older_database_gains_the_new_columns_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE regions (region_id INTEGER PRIMARY KEY "
                   "AUTOINCREMENT, figure_sha256 TEXT, condition TEXT)")
        db.execute("CREATE TABLE figures (figure_sha256 TEXT PRIMARY KEY, "
                   "paper_key TEXT)")
        db.execute("INSERT INTO regions (figure_sha256, condition) VALUES ('s', 'WT')")
    connection = pp.open_database(path)
    columns = {r[1] for r in connection.execute("PRAGMA table_info(regions)")}
    assert {"scale_source", "size_unit", "conflict_reason", "run_id"} <= columns
    assert connection.execute("SELECT condition FROM regions").fetchall() == [("WT",)]
    tables = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(pp.TABLES) <= tables
    connection.close()


def test_of_two_copies_the_one_with_a_legend_is_measured(tmp_path):
    """Found on PMC9744290 on 2026-09-21: a copy named ahead of the original
    was measured, and the original -- the one legends.csv names -- became
    the duplicate, so the measured rows lost their legend."""
    image = _figure(12)
    _save(image, tmp_path / "copy_of_g006.png")
    _save(image, tmp_path / "g006.png")
    (tmp_path / pp.LEGENDS_FILE).write_text(
        'file,legend\ng006.png,"Plaques.A, WT and KO plaques."\n', encoding="utf-8")
    summary = _measure_folder(tmp_path)
    (path,), = _rows(summary["database"], "SELECT path FROM duplicates")
    assert Path(path).name == "copy_of_g006.png"
    assert {r[0] for r in _rows(summary["database"],
                                "SELECT strength FROM regions")} == {"strong"}
