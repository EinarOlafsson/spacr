"""The 424 transfer spike's arithmetic, and the two places it refuses to guess.

The harness (``tools/measure_plaque_detector_transfer.py``) turns a person's
verdicts into the precision and recall that instruction 424 is decided on, so
the arithmetic is the finding. These tests pin it, and they pin the two
refusals that stop a hand-labelled measurement from quietly lying: a label
file that is not finished is not scored, and a figure that claims more found
regions than it holds is an error rather than a recall above 1.

The zip test is here because of a defect this harness shipped with for one
afternoon: PLOS bundles ``g001.webp`` at 1600 px beside ``g001.gif`` at
200 px, the chooser ranked by FORMAT, and every PLOS figure reached the
detector as a thumbnail. Nothing failed -- the detector simply found nothing
in them, which is the result being measured. The rule is now size first.
"""
from __future__ import annotations

import io
import json
import sys
import types
import zipfile
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

V1 = "toxoplasma_well_detector_v1"
V2 = "toxoplasma_well_detector_v2"


def _tool():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        return import_module("measure_plaque_detector_transfer")
    finally:
        sys.path.remove(tools_dir)


spike = _tool()


def _write(tmp_path, figures, labels):
    """Write a detections/labels pair and return the output root.

    :param tmp_path: the test's directory.
    :param figures: ``[(key, {model: n_boxes})]``.
    :param labels: the label figure entries.
    :returns: the root to pass as ``--out``.
    """
    root = tmp_path / "out"
    root.mkdir()
    records = []
    for key, counts in figures:
        boxes = {model: [{"x0": 0, "y0": 0, "x1": 10, "y1": 10}] * n
                 for model, n in counts.items()}
        records.append({"key": key, "pmcid": key.split("__")[0],
                        "boxes": boxes})
    (root / "detections.json").write_text(json.dumps({
        "models": {V1: {"name": "v3"}}, "conf": 0.25, "figures": records}))
    (root / "labels.json").write_text(json.dumps({"figures": labels}))
    return root


def _args(root, **kwargs):
    """A namespace the stages accept.

    :param root: the output directory.
    :param kwargs: overrides.
    :returns: the namespace.
    """
    values = {"out": str(root), "labels": None, "allow_partial": False}
    values.update(kwargs)
    return types.SimpleNamespace(**values)


def test_precision_counts_a_duplicate_against_the_model(tmp_path, capsys):
    """Two boxes on one region is one measurement and one mistake."""
    root = _write(
        tmp_path,
        [("PMC1__f1.jpg", {V1: 3})],
        [{"key": "PMC1__f1.jpg", "regions": {"well": 2, "other": 0},
          "models": {V1: {"box_verdicts": ["tp", "dup", "tp"],
                          "found": {"well": 2, "other": 0}}}}])
    spike.stage_score(_args(root))
    summary = json.loads((root / "result.json").read_text())["summary"][V1]
    assert summary["boxes"] == 3
    assert summary["tp"] == 2 and summary["dup"] == 1 and summary["fp"] == 0
    assert summary["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["recall_micro"] == 1.0
    assert summary["recall_wells"] == 1.0


def test_the_two_recalls_disagree_and_both_are_reported(tmp_path, capsys):
    """One big figure moves micro and cannot move macro.

    Ninety-six wells in one figure and one well in another: micro is the
    pooled 96/97 the plate photograph dictates, macro is the mean of 1.0 and
    0.0. Reporting either alone would be choosing the flattering number, so
    both are asserted here.
    """
    root = _write(
        tmp_path,
        [("PMC1__plate.jpg", {V1: 96}), ("PMC2__panel.jpg", {V1: 0})],
        [{"key": "PMC1__plate.jpg", "regions": {"well": 96, "other": 0},
          "models": {V1: {"box_verdicts": ["tp"] * 96,
                          "found": {"well": 96, "other": 0}}}},
         {"key": "PMC2__panel.jpg", "regions": {"well": 1, "other": 0},
          "models": {V1: {"box_verdicts": [],
                          "found": {"well": 0, "other": 0}}}}])
    spike.stage_score(_args(root))
    summary = json.loads((root / "result.json").read_text())["summary"][V1]
    assert summary["recall_micro"] == pytest.approx(96 / 97, abs=1e-4)
    assert summary["recall_macro"] == pytest.approx(0.5)


def test_crops_and_wells_are_counted_apart(tmp_path, capsys):
    """The verdict turned on this split, so the split is pinned.

    A detector that finds every round well and no cropped panel scores 1.0 on
    wells and 0.0 on crops; a single pooled recall would have read 0.5 and
    hidden which half failed.
    """
    root = _write(
        tmp_path,
        [("PMC1__f1.jpg", {V1: 2})],
        [{"key": "PMC1__f1.jpg", "regions": {"well": 2, "other": 2},
          "models": {V1: {"box_verdicts": ["tp", "tp"],
                          "found": {"well": 2, "other": 0}}}}])
    spike.stage_score(_args(root))
    summary = json.loads((root / "result.json").read_text())["summary"][V1]
    assert summary["recall_wells"] == 1.0
    assert summary["found_other"] == 0 and summary["regions_other"] == 2
    assert summary["recall_micro"] == pytest.approx(0.5)


def test_an_unjudged_box_stops_the_score(tmp_path):
    """A half-judged sample scored as a whole one is the failure mode."""
    root = _write(
        tmp_path,
        [("PMC1__f1.jpg", {V1: 2})],
        [{"key": "PMC1__f1.jpg", "regions": {"well": 1, "other": 0},
          "models": {V1: {"box_verdicts": ["tp", "?"],
                          "found": {"well": 1, "other": 0}}}}])
    with pytest.raises(SystemExit) as caught:
        spike.stage_score(_args(root))
    assert "PMC1__f1.jpg" in str(caught.value)
    assert not (root / "result.json").exists()


def test_allow_partial_scores_what_was_judged_and_says_so(tmp_path, capsys):
    """The escape hatch must report the hole, not paper over it."""
    root = _write(
        tmp_path,
        [("PMC1__f1.jpg", {V1: 1}), ("PMC2__f1.jpg", {V1: 1})],
        [{"key": "PMC1__f1.jpg", "regions": {"well": 1, "other": 0},
          "models": {V1: {"box_verdicts": ["tp"],
                          "found": {"well": 1, "other": 0}}}},
         {"key": "PMC2__f1.jpg", "regions": {"well": None, "other": None},
          "models": {V1: {"box_verdicts": ["?"],
                          "found": {"well": None, "other": None}}}}])
    spike.stage_score(_args(root, allow_partial=True))
    result = json.loads((root / "result.json").read_text())
    assert result["summary"][V1]["figures"] == 1
    assert result["unjudged"] == ["PMC2__f1.jpg: regions"]
    assert "PARTIAL" in capsys.readouterr().out


def test_finding_more_regions_than_the_figure_holds_is_an_error(tmp_path):
    """Recall above 1.0 is a typo in the labels, not a result."""
    root = _write(
        tmp_path,
        [("PMC1__f1.jpg", {V1: 1})],
        [{"key": "PMC1__f1.jpg", "regions": {"well": 1, "other": 0},
          "models": {V1: {"box_verdicts": ["tp"],
                          "found": {"well": 2, "other": 0}}}}])
    with pytest.raises(SystemExit) as caught:
        spike.stage_score(_args(root))
    assert "more regions than the figure has" in str(caught.value)


def _bundle(monkeypatch, members):
    """Serve one zip of figure files from the supplementary endpoint.

    :param monkeypatch: the fixture.
    :param members: ``{name: bytes}``.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    payload = buffer.getvalue()

    class _Response:
        status_code = 200
        content = payload

    monkeypatch.setattr(spike, "_get", lambda *a, **k: _Response())
    monkeypatch.setattr(spike.time, "sleep", lambda *a: None)


def test_a_webp_figure_is_a_figure(tmp_path, monkeypatch):
    """The PLOS thumbnail defect, exactly as it happened.

    PLOS ships the full-size figure as webp and a 200 px preview as gif. webp
    was missing from the extension list, so the preview was the only candidate
    and the detector was handed a thumbnail of every PLOS figure.
    """
    big = b"RIFFwebp" + b"x" * 40000
    small = b"GIF89a" + b"y" * 200
    _bundle(monkeypatch, {"g001.webp": big, "g001.gif": small})
    kept = spike.fetch_figure_images("PMC1", tmp_path / "PMC1")
    assert [p.name for p in kept] == ["g001.webp"]
    assert kept[0].read_bytes() == big


def test_the_biggest_copy_of_a_figure_wins_not_the_best_extension(
        tmp_path, monkeypatch):
    """Size decides, format only breaks ties.

    A small PNG beside a large JPEG is the case that separates the two rules:
    ranking by format keeps the PNG, which is the smaller image, and ranking
    by size keeps the JPEG, which is the figure as published.
    """
    big_jpeg = b"\xff\xd8" + b"x" * 40000
    small_png = b"\x89PNG" + b"y" * 300
    _bundle(monkeypatch, {"f1.jpg": big_jpeg, "f1.png": small_png})
    kept = spike.fetch_figure_images("PMC1", tmp_path / "PMC1")
    assert [p.name for p in kept] == ["f1.jpg"]
    assert kept[0].read_bytes() == big_jpeg


def test_each_model_is_tagged_by_its_version_not_its_prefix():
    """Two zoo keys share eleven characters; the tag comes off the end."""
    assert spike._short_tag(V1) == "V1"
    assert spike._short_tag(V2) == "V2"
    assert spike._short_tag(V1) != spike._short_tag(V2)
