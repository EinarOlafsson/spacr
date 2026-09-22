"""The 424 transfer spike's arithmetic, and the two places it refuses to guess.

The harness (``tools/measure_plaque_detector_transfer.py``) turns a person's
verdicts into the precision and recall that instruction 424 is decided on, so
the arithmetic is the finding. These tests pin it, and they pin the two
refusals that stop a hand-labelled measurement from quietly lying: a label
file that is not finished is not scored, and a figure that claims more found
regions than it holds is an error rather than a recall above 1.

The zip tests are here because of a defect this harness shipped with for one
afternoon: journals bundle ``g001.webp`` at 1600 px beside ``g001.gif`` at
200 px, the chooser ranked by FORMAT, and seventeen figures across four
papers and three publishers reached the detector as thumbnails. Nothing
failed -- the detector simply found nothing in them, which is the result being
measured. The rule is now size first, and the cache is keyed on the rule, so
a rerun into a populated directory cannot serve the old choice back.

The last four tests read the committed evidence file rather than a fixture.
That file IS the deliverable of item 424's first question, so the properties
the scorer refuses to score without -- every box judged, no recall above one,
regions found equal to boxes marked tp -- are asserted of the shipped record
too, along with the two things that make it checkable by someone who was not
here: the weights named as the zoo names them, and every excluded article
listed rather than counted.
"""
from __future__ import annotations

import inspect
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


def test_figures_fetched_under_the_old_rule_are_not_reused(
        tmp_path, monkeypatch):
    """The cache is keyed on the RULE, not on "there are files here".

    This is how the thumbnail defect would have outlived its own fix. The
    fetcher used to return whatever images were already in the directory, so
    rerunning the ``figures`` stage into a populated output -- which is what
    the queued retraining job asks the next person to do -- would have
    measured the thumbnails again and reported it as a fresh run.
    """
    dest = tmp_path / "PMC1"
    dest.mkdir()
    (dest / "g001.gif").write_bytes(b"GIF89a" + b"y" * 200)
    big = b"RIFFwebp" + b"x" * 40000
    _bundle(monkeypatch, {"g001.webp": big, "g001.gif": b"GIF89a" + b"y" * 200})

    kept = spike.fetch_figure_images("PMC1", dest)

    assert [p.name for p in kept] == ["g001.webp"]
    assert kept[0].read_bytes() == big
    assert not (dest / "g001.gif").exists()


def test_a_directory_this_rule_filled_is_reused_without_refetching(
        tmp_path, monkeypatch):
    """The marker is what makes the cache safe, so it has to actually cache."""
    dest = tmp_path / "PMC1"
    dest.mkdir()
    (dest / "g001.webp").write_bytes(b"RIFFwebp" + b"x" * 40000)
    (dest / ".selection").write_text(f"{spike.SELECTION_RULE}\n")

    def _refuse(*args, **kwargs):
        raise AssertionError("refetched a directory this rule already filled")

    monkeypatch.setattr(spike, "_get", _refuse)
    kept = spike.fetch_figure_images("PMC1", dest)

    assert [p.name for p in kept] == ["g001.webp"]


def test_regions_found_must_equal_the_boxes_marked_tp(tmp_path):
    """The protocol defines them as one quantity, so they cannot disagree.

    ``found`` feeds recall and ``tp`` feeds precision, and a label file that
    says three regions were found while marking two boxes tp has a hand-typed
    error in one of them. Without this the two numbers drift apart silently
    and each looks defensible on its own.
    """
    root = _write(
        tmp_path,
        [("PMC1__f1.jpg", {V1: 3})],
        [{"key": "PMC1__f1.jpg", "regions": {"well": 4, "other": 0},
          "models": {V1: {"box_verdicts": ["tp", "tp", "fp"],
                          "found": {"well": 3, "other": 0}}}}])
    with pytest.raises(SystemExit) as caught:
        spike.stage_score(_args(root))
    assert "marks 2 boxes tp" in str(caught.value)


def test_each_model_is_tagged_by_its_version_not_its_prefix():
    """Two zoo keys share eleven characters; the tag comes off the end."""
    assert spike._short_tag(V1) == "V1"
    assert spike._short_tag(V2) == "V2"
    assert spike._short_tag(V1) != spike._short_tag(V2)


def test_this_harness_does_not_swap_the_channels_a_second_time():
    """The swap that voided the first run, and the double swap that would undo it.

    ``_load_image`` returns RGB because the overlays are drawn with Pillow,
    and the first run handed that same array straight to the detector.
    Ultralytics reads an array as BGR, so every figure was measured with red
    and blue exchanged; v4 found 15 boxes on the cropped-panel figures that
    way and 72 the right way round.

    This harness used to correct that itself. Since instruction 445 the
    correction is inside :func:`spacr.plaque.detect_wells`, so a correction
    here as well would swap twice and put the RGB bug back -- with nothing
    visible to say so, because the arrays are the right shape and dtype
    either way. A conversion in this file is therefore the regression, and
    this test is what catches it.
    """
    pytest.importorskip("numpy")
    source = inspect.getsource(spike)

    assert not hasattr(spike, "_detector_input"), (
        "the channel swap belongs to spacr.plaque.detect_wells now"
    )
    assert "[:, :, ::-1]" not in source, (
        "this harness must hand detect_wells RGB and let it convert once"
    )


EVIDENCE = ROOT / "features" / "data" / "424_detector_transfer_2026-09-19.json"


def _evidence():
    """The committed measurement, or a skip when it is not here.

    :returns: the parsed evidence file.
    """
    if not EVIDENCE.is_file():
        pytest.skip(f"{EVIDENCE} is not present")
    return json.loads(EVIDENCE.read_text())


def test_the_committed_evidence_names_the_weights_the_zoo_names():
    """The record has to be readable without knowing which field to ignore.

    The install step versions the local filename to avoid clobbering a
    download, so the two arms first recorded the same checkpoint under four
    different names, none of them the one the zoo row and the ledger use. The
    sha256 is the identity; the name beside it must agree with it.
    """
    from spacr import model_zoo

    by_sha = {row["sha256"]: row["name"]
              for row in model_zoo.BUNDLED_REMOTE_MODELS
              if row.get("sha256")}
    seen = set()
    for arm in _evidence()["arms"].values():
        for key, model in arm["models"].items():
            assert model["sha256"] in by_sha, key
            assert model["name"] == by_sha[model["sha256"]], key
            assert model["zoo_key"] == key
            seen.add((key, model["name"]))
    assert seen == {(V1, "yolo_welldetect_v3.pt"),
                    (V2, "yolo_welldetect_v4.pt")}


def test_the_committed_evidence_lists_every_excluded_article():
    """Refusing the training articles is the control; a count is not a record.

    Whether the current detector's score means anything rests entirely on it
    never having seen the paper, and the file first recorded that as a number
    beside a mutable upstream path. The ids themselves are what a rescoring
    run needs.
    """
    excluded = _evidence()["excluded"]
    ids = excluded["pmcid_list"]
    assert len(ids) == excluded["pmcids"]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("PMC") and i[3:].isdigit() for i in ids)
    assert excluded["revision"] and excluded["source_sha256"]


def test_no_sampled_paper_is_one_the_detector_trained_on():
    """The exclusion is checkable now, so check it rather than trust it."""
    evidence = _evidence()
    excluded = set(evidence["excluded"]["pmcid_list"])
    for arm in evidence["arms"].values():
        for paper in arm["papers"]:
            assert paper["pmcid"] not in excluded, paper["pmcid"]


def test_the_committed_labels_obey_the_rule_the_scorer_enforces():
    """The guards are worth nothing if the shipped record predates them."""
    for arm in _evidence()["arms"].values():
        for figure in arm["labels"]:
            regions = figure["regions"]
            for model, judged in figure["models"].items():
                verdicts = judged["box_verdicts"]
                found = judged["found"]
                assert found["well"] <= regions["well"], figure["key"]
                assert found["other"] <= regions["other"], figure["key"]
                assert (found["well"] + found["other"]
                        == verdicts.count("tp")), figure["key"]
                assert "?" not in verdicts, figure["key"]


def test_several_sizes_are_merged_by_the_modules_own_merge():
    """2026-09-21: the harness measures the sizes the module ships (640 and
    1280) the way the module merges them, not by a merge of its own."""
    assert spike._sizes("640,1280") == [640, 1280]
    assert spike._sizes(640) == [640]
    same = {"x0": 10, "y0": 10, "x1": 50, "y1": 50, "confidence": 0.5}
    per_size = {"640": [same, {"x0": 100, "y0": 100, "x1": 140, "y1": 140,
                               "confidence": 0.6}],
                "1280": [dict(same, x1=52, confidence=0.7),
                         {"x0": 200, "y0": 10, "x1": 230, "y1": 40,
                          "confidence": 0.4}]}
    merged = spike._merged_like_the_module(per_size)
    assert [(b["x0"], b["sizes"]) for b in merged] == [
        (10, [640, 1280]), (200, [1280]), (100, [640])]
    assert merged[0]["confidence"] == 0.7, "the higher score is kept"


def test_the_merged_run_record_adds_up():
    """The 2026-09-21 record: 640 and 1280 reproduce the earlier totals, and
    the merged boxes split exactly into the kept 640 boxes and the new ones."""
    record = json.loads((ROOT / "features" / "data" /
                         "424_detector_transfer_merged_2026-09-21.json").read_text())
    totals = record["totals"]
    assert totals["v2_boxes"] == {"640": 148, "1280": 194, "merged": 206}
    rows = record["per_figure"]
    assert len(rows) == 182
    assert sum(r["merged"] for r in rows) == 206
    assert totals["merged_boxes_on_the_24_region_bearing_figures"] == (
        totals["of_those_also_found_at_640"]
        + totals["of_those_found_only_at_1280"])
    assert all(r["boxes_640_identical_to_2026_09_20"] for r in rows)
