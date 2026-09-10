"""The edges of the run digest: what it does when a source cannot answer.

A digest is assembled at the end of a long run from subsystems that are each
allowed to fail, and the number check in front of the model has to hold even
when the prose hands it something no sane run would produce. These tests
drive those edges: a run folder that is not there, a registry file that is not
a database, settings supplied by hand rather than read from the journal, a
measurement that is ``None``, and numbers written so extremely that parsing
them is itself a failure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr.methods_export import (build_digest, digest_numbers,  # noqa: E402
                                  extract_numbers, render_methods,
                                  render_results, verify_numbers)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Keep a shared registry override from answering for the tmp project."""
    from spacr import artifacts

    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


# ---------------------------------------------------------------------------
# Harvesting numbers out of a digest
# ---------------------------------------------------------------------------

def test_a_measurement_that_was_never_taken_asserts_no_number():
    """A hit whose effect was not finite reaches the digest as ``None``, and
    a null is the absence of a claim: it must contribute no allowed number,
    and it must not be treated as a string either."""
    digest = {"run": {"elapsed_s": None, "n_settings": 4},
              "hits": [{"effect": None, "p_value": "0.031", "flags": []}]}

    assert digest_numbers(digest) == {4.0, 0.031}


def test_an_empty_quotable_string_does_not_shatter_the_numbers():
    """``str.replace("")`` inserts its replacement between every character,
    so an empty entry in the quotable strings would turn ``17`` into ``1``
    and ``7`` and let a real invention past as two structural digits."""
    found = extract_numbers("17 fields and 4 plates",
                            strings=("", "fields"))

    assert found == ["17", "4"]


# ---------------------------------------------------------------------------
# Numbers written past the edge of what can be parsed
# ---------------------------------------------------------------------------

def test_a_number_too_large_to_be_finite_is_never_supported():
    """``1e999`` parses to infinity, and infinity is within any tolerance of
    anything, so accepting it would license every figure in the digest at
    once. It is an invention and has to be reported as one."""
    digest = {"statistics": {"n_genes_tested": 517, "alpha": 0.05}}

    verdict = verify_numbers("The largest effect was 1e999.", digest)

    assert verdict.ok is False
    assert verdict.checked == 1
    assert verdict.unsupported == ("1e999",)
    assert "1e999" in verdict.problem()


def test_an_exponent_too_long_to_parse_does_not_widen_the_tolerance():
    """The exponent decides how much rounding a written number is allowed:
    ``1E5`` may stand for anything within 50000. An exponent CPython refuses
    to parse -- past its integer-string digit limit -- must fall back to the
    strictest reading rather than crashing the check, so the same value
    written that way is held to half a unit and the near miss is caught."""
    limit = sys.get_int_max_str_digits()
    written = "1E" + "0" * (limit + 1) + "5"
    digest = {"statistics": {"max_abs_effect": 120000.0}}

    unparseable = verify_numbers(f"The effect was {written} overall.", digest)
    plain = verify_numbers("The effect was 1E5 overall.", digest)

    assert float(written) == 100000.0, "both spellings are the same number"
    assert plain.ok is True, "1E5 rounds to 120000 within its own tolerance"
    assert unparseable.ok is False
    assert unparseable.unsupported == (written,)


# ---------------------------------------------------------------------------
# Sources that cannot answer
# ---------------------------------------------------------------------------

def test_an_unreadable_run_folder_becomes_a_note_the_methods_section_carries(
        tmp_path):
    """Losing the journal must not lose the digest, and the loss must not be
    silent: the paragraph says the run journal could not be read rather than
    quietly omitting the versions and the status."""
    digest = build_digest(run_dir=str(tmp_path / "no_such_run"))

    assert digest["run"] == {}
    assert digest["environment"] == {}
    assert any("the run journal could not be read" in note
               for note in digest["notes"]), digest["notes"]
    assert "the run journal could not be read" in render_methods(digest)


def test_settings_given_by_hand_are_not_overwritten_by_the_journals(
        tmp_path, monkeypatch):
    """``settings=`` is the caller saying what the run was configured with.
    A journal is read for its manifest either way, but it must not replace
    the settings the caller supplied, or the seed and the illumination caveat
    would describe a different run from the one being written up."""
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root",
                        lambda: Path(tmp_path / "runs"))
    journal_settings = {"diameter": 30, "random_seed": 99,
                        "illumination_correction": True}
    with run_journal.open_run("mask", journal_settings) as run:
        run_dir = run.dir

    digest = build_digest(run_dir=run_dir,
                          settings={"random_seed": 4242,
                                    "illumination_correction": False})

    assert digest["run"]["app_key"] == "mask", "the manifest is still read"
    assert digest["run"]["journal_run_id"] == os.path.basename(str(run_dir))
    assert digest["run"]["seed"] == 4242
    assert digest["qc"]["illumination_correction"] is False
    assert any("The random seed was 4242" in caveat
               for caveat in digest["caveats"]), digest["caveats"]
    assert any("Illumination correction was not applied" in caveat
               for caveat in digest["caveats"]), digest["caveats"]


def test_segmentation_qc_that_cannot_be_read_leaves_no_qc_verdict(
        tmp_path, monkeypatch):
    """A QC reader that raises must leave the section EMPTY rather than
    filling it with a null verdict: ``None`` under ``segmentation`` would
    render as a scored project whose verdict happened to be blank, which is
    the one reading a missing QC card must never get."""
    from spacr import seg_qc

    def refuse(_project, **_kwargs):
        raise OSError("the scorecard is unreadable")

    monkeypatch.setattr(seg_qc, "read_digest", refuse)
    project = tmp_path / "plate1"
    project.mkdir()

    digest = build_digest(project=str(project))

    assert "segmentation" not in digest["qc"]
    assert any("the segmentation QC could not be read" in note
               for note in digest["notes"]), digest["notes"]
    assert digest["provenance"]["n_artifacts"] == 0, (
        "the registry is read even though QC failed")
    assert render_results(digest).startswith("## Results")


def test_a_registry_that_is_not_a_database_leaves_no_provenance(tmp_path):
    """The artifact registry is a sqlite file beside the project. A corrupt
    one must cost the digest its provenance block and nothing else -- and it
    must not be recorded as a project with zero artifacts, which is what an
    unguarded null would render as."""
    from spacr import artifacts

    project = tmp_path / "plate2"
    project.mkdir()
    (project / artifacts.ARTIFACTS_DB_NAME).write_bytes(b"not a database")

    digest = build_digest(project=str(project))

    assert digest["provenance"] == {}
    assert any("the artifact registry could not be read" in note
               for note in digest["notes"]), digest["notes"]
    assert digest["qc"]["segmentation"]["verdict"] == "missing", (
        "the QC card is still read even though the registry failed")
    assert "registered artifact(s)" not in render_results(digest)


# ---------------------------------------------------------------------------
# Rendering statistics that stop short
# ---------------------------------------------------------------------------

def test_results_omit_the_effect_size_sentence_when_there_is_no_effect_size():
    """A statistics block from a run that recorded counts but no effect
    magnitudes still has a results paragraph. The sentence about the largest
    effect is dropped whole rather than printed with a hole in it, and the
    hit list after it is unaffected."""
    digest = {
        "statistics": {"n_genes_tested": 517, "n_significant": 12,
                       "alpha": 0.05, "n_up": 7, "n_down": 5,
                       "n_corroborated": 4},
        "hits": [{"name": "TSG101", "gene": "7251", "effect": 2.4,
                  "p_value": 0.000004, "q_value": 0.002,
                  "n_agree": 3, "n_guides": 4}],
    }

    text = render_results(digest)

    assert ("Of 517 gene(s) tested, 12 cleared the threshold of 0.05 — 7 with "
            "a positive effect and 5 with a negative one.") in text
    assert "4 of them were corroborated" in text
    assert "largest absolute effect" not in text
    assert "median" not in text
    assert ("- TSG101 (7251): effect 2.4, p = 4e-06, q = 0.002, 3 of 4 gRNAs "
            "agreeing.") in text


def test_a_screen_with_no_hits_does_not_report_an_effect_of_nan():
    """An empty screen has no largest effect, and must not claim one.

    ``hits.summary()`` reports ``float("nan")`` for ``max_abs_effect`` when no
    guide cleared the threshold. ``nan is not None``, so a plain None-check
    lets it through and the methods section renders the sentence "The largest
    absolute effect among them was nan, with a median of nan." into text that
    goes to a journal.
    """
    digest = {
        "statistics": {
            "n_tested": 12, "n_significant": 0, "alpha": 0.05,
            "n_up": 0, "n_down": 0, "n_corroborated": 0,
            "max_abs_effect": float("nan"),
            "median_abs_effect": float("nan"),
        },
    }

    text = render_results(digest)

    assert "nan" not in text
    assert "largest absolute effect" not in text


def test_a_screen_with_hits_still_reports_its_largest_effect():
    """Screening out nan must not screen out a real effect size."""
    digest = {
        "statistics": {
            "n_tested": 12, "n_significant": 3, "alpha": 0.05,
            "n_up": 2, "n_down": 1, "n_corroborated": 2,
            "max_abs_effect": 1.75, "median_abs_effect": 0.5,
        },
    }

    text = render_results(digest)

    assert "The largest absolute effect among them was 1.75" in text
    assert "median of 0.5" in text
