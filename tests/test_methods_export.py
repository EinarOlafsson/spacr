"""Real tests for :mod:`spacr.methods_export` — the run digest and its numbers.

The module's claim is narrow and testable: **every number in the output comes
from the digest**. Two families of test hold it to that.

* *Provenance forward* — a distinctive number planted in the digest has to
  come out in the rendered sections and in the prompt, verbatim. If it does
  not, the sections are not made of the run.
* *Provenance backward* — a number that is NOT in the digest has to be
  reported as unsupported, wherever it appears and however it is dressed up.
  If it is not, the check is decoration.

The rest is the digest itself: assembled from six independent subsystems,
each optional, none of them able to take the whole thing down.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr.methods_export import (ALWAYS_ALLOWED, DIGEST_VERSION,  # noqa: E402
                                  build_digest, caveats_for, check_draft,
                                  digest_numbers, digest_strings,
                                  extract_numbers, render_methods,
                                  render_prompt, render_results, system_prompt,
                                  verify_numbers)

#: A number nothing else in spaCR would produce, so finding it anywhere is
#: proof it came from the digest and nowhere else.
PLANTED = 48.3179


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    from spacr import artifacts

    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


@pytest.fixture
def digest():
    """A digest with the planted number in the statistics."""
    return build_digest(
        title="the pilot screen",
        settings={"illumination_correction": True, "on_error": "skip",
                  "random_seed": 7714},
        hits=_FakeHits(),
    )


class _FakeHit:
    """One hit row, shaped like :class:`spacr.hits.Hit`."""

    def __init__(self, rank, gene, effect, p_value, q_value):
        self.rank = rank
        self.gene = gene
        self.name = f"name-{gene}"
        self.effect = effect
        self.p_value = p_value
        self.q_value = q_value
        self.n_guides = 4
        self.n_agree = 3
        self.agreement = 0.75
        self.direction = "up" if effect > 0 else "down"
        self.flags = ()


class _FakeHits(list):
    """A stand-in hit list carrying the planted maximum effect."""

    def __init__(self):
        super().__init__([
            _FakeHit(1, "233460", PLANTED, 1.2e-6, 3.4e-5),
            _FakeHit(2, "239740", -2.9013, 4.4e-4, 0.0031),
        ])

    def summary(self):
        return {
            "source": "(test)", "regression_type": "ols", "ranking": "q-value",
            "alpha": 0.05, "n_terms": 611, "n_genes_tested": 517,
            "n_listed": 517, "n_significant": 23, "n_up": 14, "n_down": 9,
            "n_corroborated": 11, "max_abs_effect": PLANTED,
            "median_abs_effect": 1.8842, "top_genes": ["233460", "239740"],
            "filters": {}, "flag_counts": {}, "notes": [],
        }


# ---------------------------------------------------------------------------
# Provenance forward: a planted number reaches the output verbatim
# ---------------------------------------------------------------------------

def test_a_number_planted_in_the_digest_appears_in_the_results_verbatim(digest):
    """The figures in the text are the figures in the run."""
    assert PLANTED in digest_numbers(digest)

    text = render_results(digest)

    assert str(PLANTED) in text, (
        f"{PLANTED} is in the digest but not in the results section; the "
        f"prose is not made of the run")


def test_the_planted_number_reaches_the_prompt_verbatim(digest):
    _system, user = render_prompt(digest)

    assert str(PLANTED) in user, (
        "the model cannot quote a number it was never shown")


def test_every_headline_statistic_reaches_the_results_section(digest):
    text = render_results(digest)

    for value in ("517", "23", "14", "9", "11", "0.05"):
        assert value in text, f"{value} is in the digest and not in the prose"


def test_the_rendered_sections_pass_their_own_number_check(digest):
    methods_check, results_check = check_draft(
        render_methods(digest), render_results(digest), digest)

    assert methods_check.ok, methods_check.problem()
    assert results_check.ok, results_check.problem()
    assert results_check.checked > 5, (
        "a results section with almost no numbers in it is not being checked")


def test_the_planted_number_survives_a_json_round_trip(digest):
    restored = json.loads(json.dumps(digest))

    assert PLANTED in digest_numbers(restored)
    assert str(PLANTED) in render_results(restored)


# ---------------------------------------------------------------------------
# Provenance backward: a number absent from the digest cannot get through
# ---------------------------------------------------------------------------

def test_a_number_absent_from_the_digest_is_reported(digest):
    verdict = verify_numbers("Sequencing recovered 9999 barcodes.", digest)

    assert verdict.ok is False
    assert verdict.unsupported == ("9999",)
    assert "9999" in verdict.problem()
    assert bool(verdict) is False


def test_an_invented_number_is_caught_however_it_is_written(digest):
    for invention in ("41.7", "1.28e+04", "-773.5", "0.00042"):
        verdict = verify_numbers(f"The value was {invention}.", digest)
        assert not verdict.ok, f"{invention} passed but is not in the digest"


def test_a_plausible_near_miss_is_still_an_invention(digest):
    """48.3179 is in the digest; 48.4 is not, and the difference matters."""
    assert verify_numbers(f"An effect of {PLANTED}.", digest).ok
    assert not verify_numbers("An effect of 48.4.", digest).ok
    assert not verify_numbers("An effect of 483.179.", digest).ok


def test_a_correct_rounding_of_a_digest_number_is_allowed(digest):
    assert verify_numbers("An effect of 48.32.", digest).ok
    assert verify_numbers("An effect of 48.3.", digest).ok


def test_an_integer_is_a_count_and_a_count_is_exact(digest):
    """48.3179 rounds to 48, but ``48`` in a results section is a headcount.

    Letting an integer stand for a rounding is how a hit count and an effect
    size end up as the same sentence.
    """
    assert not verify_numbers("An effect of 48.", digest).ok
    assert verify_numbers("There were 517 genes.", digest).ok


def test_only_the_structural_numbers_are_free(digest):
    assert ALWAYS_ALLOWED == frozenset({0.0, 1.0, 2.0, 100.0})
    for value in ALWAYS_ALLOWED:
        assert verify_numbers(f"There were {value:g}.", digest).ok
    assert 37.0 not in digest_numbers(digest)
    assert not verify_numbers("There were 37.", digest).ok


def test_quoting_a_digest_string_verbatim_is_not_an_invention(digest):
    version = digest["spacr_version"]

    verdict = verify_numbers(f"Analysis used spaCR {version}.", digest)

    assert verdict.ok
    assert version in digest_strings(digest)


def test_generated_caveats_are_not_quotable_cover_for_their_numbers(digest):
    """A caveat spaCR wrote must not license the figures inside it.

    If quoting ``The random seed was 7714.`` were enough, the check would be
    testing spaCR's own prose against itself.
    """
    caveat = next(c for c in digest["caveats"] if "seed" in c)

    assert caveat not in digest_strings(digest)
    assert verify_numbers(caveat, digest).checked >= 1
    assert verify_numbers(caveat, digest).ok, "7714 IS in the digest"
    assert not verify_numbers("The random seed was 7715.", digest).ok


def test_structure_is_not_mistaken_for_a_claim(digest):
    text = ("1. First we did a thing.\n"
            "2. Then another.\n"
            "Run 8f21c0a3b4d5 finished on 2026-07-14T09:12:03Z with "
            "spaCR 1.2.3.")

    verdict = verify_numbers(text, digest)

    assert verdict.ok, f"structure was read as a claim: {verdict.unsupported}"


def test_extract_numbers_removes_quoted_strings_before_looking(digest):
    numbers = extract_numbers("The project was /data/plate7 and n was 517.",
                              ["/data/plate7"])

    assert numbers == ["517"], (
        "a path quoted verbatim must not be read as asserting its digits")


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------

def test_the_methods_section_states_every_caveat(digest):
    text = render_methods(digest)

    assert digest["caveats"], "a run always has caveats worth stating"
    for caveat in digest["caveats"]:
        assert caveat in text


def test_a_methods_section_that_drops_a_caveat_fails(digest):
    verdict = verify_numbers(
        "Image analysis was performed with spaCR.", digest,
        require_caveats=True)

    assert verdict.ok is False
    assert verdict.missing_caveats
    assert "caveat" in verdict.problem()


def test_the_caveats_name_the_seed_the_policy_and_the_correction(digest):
    joined = " ".join(digest["caveats"])

    assert "7714" in joined
    assert "on_error=skip" in joined
    assert "Illumination correction was applied" in joined


def test_no_seed_is_itself_a_caveat():
    digest = build_digest(settings={"random_seed": None})

    assert any("not bit-for-bit reproducible" in c for c in digest["caveats"])


def test_illumination_not_running_is_a_caveat():
    digest = build_digest(settings={"illumination_correction": False})

    assert any("Illumination correction was not applied" in c
               for c in digest["caveats"])


def test_a_failing_qc_verdict_and_its_flags_become_a_caveat():
    caveats = caveats_for({
        "qc": {"segmentation": {"verdict": "fail",
                                "flags_fired": {"empty_field": 12,
                                                "touching_border": 4}}}})

    assert any("Segmentation QC returned fail" in c for c in caveats)
    assert any("empty_field (12)" in c for c in caveats)


def test_the_classifiers_held_out_metrics_become_a_caveat():
    caveats = caveats_for({
        "classifier": {"held_out": {"n": 240, "accuracy": 0.9125,
                                    "f1_macro": 0.8871},
                       "split_rule": "by plate",
                       "warnings": ["no held-out split was recorded"]}})

    assert any("held-out accuracy was 0.9125" in c for c in caveats)
    assert any("240 objects" in c for c in caveats)
    assert any("by plate" in c for c in caveats)
    assert any("no held-out split" in c for c in caveats)


def test_stale_and_missing_artifacts_become_caveats():
    caveats = caveats_for({"provenance": {"n_stale": 3, "n_missing": 1}})

    assert any("3 registered artifact(s) are stale" in c for c in caveats)
    assert any("1 registered artifact(s) are no longer on disk" in c
               for c in caveats)


def test_a_skipped_unit_count_reaches_the_caveat():
    caveats = caveats_for({"run": {"on_error": "skip", "n_skipped": 11}})

    assert any("11 were skipped" in c for c in caveats)


# ---------------------------------------------------------------------------
# The digest itself
# ---------------------------------------------------------------------------

def test_the_digest_carries_its_schema_and_is_json_serializable(digest):
    assert digest["digest_version"] == DIGEST_VERSION
    assert digest["generated_utc"].endswith("Z")
    payload = json.dumps(digest)
    assert "statistics" in payload


def test_the_digest_has_every_top_level_section(digest):
    for key in ("digest_version", "generated_utc", "title", "project",
                "spacr_version", "run", "environment", "modules",
                "parameters", "qc", "classifier", "statistics", "hits",
                "provenance", "constants", "caveats", "notes"):
        assert key in digest, f"the digest is missing {key}"


def test_the_hit_rows_reach_the_digest_and_the_prose(digest):
    assert len(digest["hits"]) == 2
    assert digest["hits"][0]["gene"] == "233460"
    assert digest["hits"][0]["effect"] == PLANTED

    text = render_results(digest)
    assert "233460" in text
    assert "3 of 4 gRNAs agreeing" in text


def test_an_empty_digest_still_renders_both_sections():
    digest = build_digest()

    methods = render_methods(digest)
    results = render_results(digest)

    assert methods.startswith("## Methods")
    assert results.startswith("## Results")
    assert check_draft(methods, results, digest)[0].ok


def test_the_digest_reads_a_real_run_journal(tmp_path, monkeypatch):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root",
                        lambda: Path(tmp_path / "runs"))
    with run_journal.open_run("mask", {"diameter": 30,
                                       "illumination_correction": True,
                                       "random_seed": 99}) as run:
        run.record_warning("two fields had no objects")
        run_dir = run.dir

    digest = build_digest(run_dir=run_dir)

    assert digest["run"]["app_key"] == "mask"
    assert digest["run"]["status"] == "success"
    assert digest["run"]["journal_run_id"] == os.path.basename(str(run_dir))
    assert digest["environment"]["spacr"], "the versions must reach the digest"
    assert digest["environment"]["python"]
    assert digest["run"]["elapsed_s"] is not None
    assert any("two fields had no objects" in c for c in digest["caveats"])
    assert digest["qc"]["illumination_correction"] is True
    assert digest["run"]["seed"] == 99


def test_the_digest_reads_the_emitted_macro(tmp_path, monkeypatch):
    from spacr import macro, run_journal

    monkeypatch.setattr(run_journal, "runs_root",
                        lambda: Path(tmp_path / "runs"))
    monkeypatch.setenv("SPACR_MACRO_DIR", str(tmp_path / "macros"))
    macro.reset()
    with run_journal.open_run("mask", {"diameter": 45, "src": str(tmp_path)}):
        pass
    run_dir = sorted((tmp_path / "runs").iterdir())[0]
    assert (run_dir / "macro.py").is_file(), "the run must emit its script"

    digest = build_digest(run_dir=run_dir)

    assert digest["modules"], "the macro's steps must reach the digest"
    assert digest["modules"][0]["module"] == "mask"
    assert digest["modules"][0]["n_settings"] > 0
    assert "diameter" in digest["parameters"]["mask"]
    assert digest["parameters"]["mask"]["diameter"] == 45


def test_the_digest_reads_a_regression_results_folder(tmp_path):
    folder = tmp_path / "results" / "pred" / "ols"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "feature": ["gene_fraction:gene[100]", "gene_fraction:gene[200]"],
        "coefficient": [2.4, -1.8],
        "p_value": [1e-6, 0.3]}).to_csv(folder / "results_gene.csv",
                                        index=False)

    digest = build_digest(results_folder=str(folder), regression_type="ols")

    assert digest["statistics"]["regression_type"] == "ols"
    assert digest["statistics"]["n_genes_tested"] == 2
    assert len(digest["hits"]) == 2
    assert "ols regression" in render_methods(digest)


def test_the_digest_reads_a_model_card(tmp_path):
    from spacr.deep_spacr import build_model_card, write_model_card

    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    card = build_model_card(
        str(model_path), classes=["neg", "pos"], epochs=12,
        split_rule="held out by plate",
        held_out={"n": 240, "accuracy": 0.9125, "f1_macro": 0.8871,
                  "per_class_accuracy": [0.9, 0.93], "class_support": [120, 120]})
    write_model_card(str(model_path), card)

    digest = build_digest(model_path=str(model_path))

    assert digest["classifier"]["classes"] == ["neg", "pos"]
    assert digest["classifier"]["held_out"]["accuracy"] == 0.9125
    assert digest["classifier"]["epochs"] == 12
    assert "0.9125" in render_results(digest)
    assert check_draft(render_methods(digest),
                       render_results(digest), digest)[1].ok


def test_the_digest_reads_the_artifact_registry(tmp_path):
    from spacr import artifacts

    root = tmp_path / "plate1"
    root.mkdir()
    registry = artifacts.Registry(project=str(root))
    masks = root / "masks.npy"
    masks.write_text("x", encoding="utf-8")
    registry.register(module="mask", kind="masks", role="cell_mask",
                      path=str(masks), settings={"diameter": 30})

    digest = build_digest(project=str(root))

    assert digest["provenance"]["n_artifacts"] == 1
    assert digest["provenance"]["modules"] == ["mask"]
    assert "1 registered artifact" in render_results(digest)


def test_a_subsystem_that_cannot_answer_is_a_note_not_an_exception(tmp_path):
    digest = build_digest(model_path=str(tmp_path / "no_such_model.pth"))

    assert digest["classifier"] == {}
    assert digest["notes"], "the failure must be recorded"
    assert any("model card" in note for note in digest["notes"])
    assert render_methods(digest).startswith("## Methods")


def test_a_broken_macro_does_not_take_the_digest_down(tmp_path):
    (tmp_path / "macro.py").write_text("this is not python (", encoding="utf-8")

    digest = build_digest(run_dir=None, macro_path=str(tmp_path / "macro.py"))

    assert digest["modules"] == []
    assert any("macro" in note for note in digest["notes"])


def test_extra_material_reaches_the_digest():
    digest = build_digest(extra={"plate_count": 6})

    assert digest["extra"]["plate_count"] == 6
    assert 6.0 in digest_numbers(digest)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

def test_the_prompt_is_a_pure_function_of_the_digest(digest):
    first = render_prompt(digest)
    second = render_prompt(json.loads(json.dumps(digest)))

    assert first == second, (
        "if the prompt depended on anything but the digest, the digest would "
        "not be the model's only input")


def test_the_prompt_states_the_rule_it_will_be_held_to():
    prompt = system_prompt()

    assert "EVERY NUMBER" in prompt
    assert "rejected" in prompt
    assert "## Methods" in prompt and "## Results" in prompt


def test_the_prompt_lists_the_caveats_the_model_must_state(digest):
    _system, user = render_prompt(digest)

    for caveat in digest["caveats"]:
        assert caveat in user


def test_the_prompt_carries_the_digest_and_nothing_else(digest):
    _system, user = render_prompt(digest)

    body = user.split("```json", 1)[1].rsplit("```", 1)[0]
    assert json.loads(body) == json.loads(json.dumps(digest))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_digest_numbers_ignores_booleans_and_non_finite_values():
    numbers = digest_numbers({"flag": True, "nan": float("nan"),
                              "inf": float("inf"), "real": 3.5})

    assert numbers == {3.5}


def test_digest_numbers_reads_a_numeric_string_as_a_number():
    assert digest_numbers({"gene": "233460"}) == {233460.0}
    assert digest_numbers({"path": "/data/plate7"}) == set()


def test_digest_strings_are_longest_first():
    strings = digest_strings({"a": "1.2.3", "b": "1.2.3.4.5", "c": "x"})

    assert strings == ["1.2.3.4.5", "1.2.3"]


def test_verification_is_json_serializable(digest):
    payload = json.dumps(verify_numbers("There were 517 genes.",
                                        digest).to_dict())

    assert "unsupported" in payload
