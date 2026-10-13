"""The methods digest is written at the end of a long run, so it never raises.

Every source it reads is optional and every one is read defensively: a
subsystem that cannot answer contributes a note, not an exception. These
tests drive the readers that can fail (a run folder with no settings file, a
seed that cannot be resolved) and the prose renderers on a digest complete
enough to exercise the sentences that a bare digest never reaches.
"""
from __future__ import annotations

import json
import os

import pytest

from spacr import methods_export as me
from spacr import run_journal, runctx, seg_qc


# ---------------------------------------------------------------------------
# Reading a run folder
# ---------------------------------------------------------------------------

def test_a_run_folder_with_no_settings_file_still_yields_its_manifest(
        tmp_path):
    """The manifest is the versions, timings and status; losing all of it
    because the settings CSV is absent would throw away the run's whole
    provenance for a file that is optional."""
    folder = tmp_path / "run_20260101_000000"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({
        "app_key": "measure", "status": "completed",
        "run_id": "abc123",
        "environment": {"python": "3.11.0"},
        "performance": {"elapsed_s": 12.0},
        "seeds": {"seed": 7},
    }), encoding="utf-8")

    fragment = me._read_manifest(folder)

    assert fragment["settings"] == {}
    assert fragment["run"]["app_key"] == "measure"
    assert fragment["run"]["status"] == "completed"


def test_a_seed_that_cannot_be_resolved_becomes_a_note_not_a_crash(
        monkeypatch):
    """`None` here means "no seed recorded", and the note is what stops that
    reading as "the run was unseeded on purpose"."""
    def refuse(_settings):
        raise ValueError("random_seed is 'auto'")

    monkeypatch.setattr(runctx, "resolve_seed", refuse)
    notes = []

    assert me._resolve_seed({"random_seed": "auto"}, notes) is None
    assert len(notes) == 1
    assert "the seed could not be resolved" in notes[0]
    assert "random_seed is 'auto'" in notes[0]


def test_an_active_run_contributes_its_id_seed_and_error_policy():
    """A digest built while the run is still open reads the context rather
    than the journal, and the two must agree about the policy in force."""
    with runctx.run_context(module="measure",
                            settings={"random_seed": 7, "on_error": "retry",
                                      "on_error_attempts": 3},
                            log=False) as context:
        live = me._live_run()

    assert live["run_id"] == context.run_id
    assert live["seed"] == 7
    assert live["on_error"] == "retry"
    assert live["on_error_attempts"] == 3
    assert live["n_skipped"] == 0
    assert live["skipped"] == []
    assert live["deterministic"] is False
    assert live["started_utc"].startswith("20"), live["started_utc"]


def test_no_active_run_contributes_nothing():
    """The contrast: outside a run the digest must record no run at all
    rather than a context of empty strings."""
    assert me._live_run() == {}


def test_the_macro_supplies_the_settings_when_nothing_else_does(tmp_path):
    """A digest built from an emitted macro alone -- no journal, no settings
    argument -- still knows what the run was configured with, which is where
    the illumination and seed caveats come from."""
    macro = tmp_path / "macro.py"
    macro.write_text(
        "SETTINGS_1 = {'src': '/data/plate1', 'nucleus_channel': 0,\n"
        "              'random_seed': 11, 'illumination_correction': True}\n"
        "MACRO = {\n"
        "    'schema': 1,\n"
        "    'steps': [\n"
        "        {'index': 1, 'module': 'measure',\n"
        "         'entry': 'spacr.measure:measure_crop', 'run_id': 'run_9',\n"
        "         'settings_hash': 'abc', 'status': 'completed',\n"
        "         'elapsed_s': 12.5, 'spacr_version': '1.0',\n"
        "         'settings': SETTINGS_1,\n"
        "         'user_set': ['nucleus_channel'], 'defaulted': ['src'],\n"
        "         'outputs': ['measurements.db'], 'link': ''},\n"
        "    ],\n"
        "}\n", encoding="utf-8")

    digest = me.build_digest(macro_path=str(macro))

    assert digest["run"]["run_id"] == "run_9"
    assert digest["run"]["n_settings"] == 4
    assert digest["run"]["seed"] == 11
    assert digest["qc"]["illumination_correction"] is True
    assert [step["module"] for step in digest["modules"]] == ["measure"]
    assert digest["parameters"]["measure"] == {"nucleus_channel": 0}


def test_a_digest_built_inside_a_run_records_that_run(tmp_path):
    """The live context fills in run fields the journal has not written yet,
    because the journal entry is only complete once the run ends."""
    with runctx.run_context(module="measure", settings={"random_seed": 3},
                            log=False) as context:
        digest = me.build_digest(project=str(tmp_path))

    assert digest["run"]["run_id"] == context.run_id
    assert digest["run"]["seed"] == 3


# ---------------------------------------------------------------------------
# Segmentation QC roll-up
# ---------------------------------------------------------------------------

def test_the_qc_rollup_adds_up_every_scorecard(monkeypatch, tmp_path):
    """One card per object type. Reporting only the last one would silently
    drop the nucleus verdict from a run that scored cells and nuclei."""
    def digest_of(_project, **_kw):
        return seg_qc.QCDigest(
            root=str(tmp_path), verdict="warn", headline="two fields blurred",
            stale=True,
            scorecards=[
                seg_qc.Scorecard(path="a.csv", object_type="cell",
                                 field_qcs=[object(), object()],
                                 summary={"n_ok": 1, "n_warn": 1, "n_fail": 0,
                                          "flag_counts": {"blurred": 1}}),
                seg_qc.Scorecard(path="b.csv", object_type="nucleus",
                                 field_qcs=[object()],
                                 summary={"n_ok": 0, "n_warn": 1, "n_fail": 2,
                                          "flag_counts": {"blurred": 3,
                                                          "empty": 1}}),
            ])

    monkeypatch.setattr(seg_qc, "read_digest", digest_of)

    out = me._segmentation_qc(str(tmp_path))

    assert out["n_fields"] == 3
    assert (out["n_ok"], out["n_warn"], out["n_fail"]) == (1, 2, 2)
    assert out["flags_fired"] == {"blurred": 4, "empty": 1}
    assert out["object_types"] == ["cell", "nucleus"]
    assert out["stale"] is True


# ---------------------------------------------------------------------------
# The prose
# ---------------------------------------------------------------------------

def _full_digest():
    return {
        "spacr_version": "1.2.3",
        "environment": {"python": "3.11.0", "torch": "2.4.0",
                        "cellpose": "4.0.1"},
        "run": {"run_id": "run_9", "on_error": "retry",
                "on_error_attempts": 3},
        "modules": [{"module": "mask"}, {"module": "measure"}],
        "parameters": {"mask": {"diameter": 30}, "measure": {}},
        "qc": {"segmentation": {"verdict": "warn", "stale": True,
                                "n_fields": 3, "n_ok": 1, "n_warn": 1,
                                "n_fail": 1, "flags_fired": {}}},
        "classifier": {},
        "statistics": {},
        "hits": [],
        "provenance": {},
    }


def test_the_methods_paragraph_names_the_versions_modules_and_parameters():
    """Every claim in a methods section has to be traceable to the run. The
    module order, the parameters the user set and the ones left at defaults
    are three different statements and all three have to appear."""
    text = me.render_methods(_full_digest())

    assert "spaCR 1.2.3" in text
    assert "on Python 3.11.0" in text
    assert "using torch 2.4.0, cellpose 4.0.1" in text
    assert "2 module(s) in the order mask → measure" in text
    assert "For mask, the parameters set explicitly were: diameter = 30." in text
    assert "measure ran entirely at its spaCR defaults." in text
    assert "recorded under id run_9" in text


def test_a_run_with_no_macro_still_names_the_module_it_ran():
    """Without a macro there are no steps, but the journal knows the app
    key; a methods section that names no module at all is unusable."""
    text = me.render_methods({"run": {"app_key": "measure"}})

    assert "The measure module was run." in text
    assert "- None recorded." in text, "the caveat list must say it is empty"


def test_the_retry_policy_and_a_stale_scorecard_are_both_caveats():
    """Retries change what a failure means, and a scorecard older than the
    masks describes a segmentation nobody has looked at."""
    caveats = me.caveats_for(_full_digest())

    assert any("on_error=retry with 3 attempts" in c for c in caveats)
    assert any("older than the masks it describes" in c for c in caveats)


def test_the_results_section_reports_the_segmentation_counts():
    """'QC ran' and 'QC scored 3 fields, 1 failed' are different claims, and
    only the second one lets a reader judge the masks."""
    text = me.render_results(_full_digest())

    assert "Segmentation QC scored 3 field(s)" in text
    assert "1 passed" in text
    assert "1 raised a warning" in text
    assert "1 failed" in text
