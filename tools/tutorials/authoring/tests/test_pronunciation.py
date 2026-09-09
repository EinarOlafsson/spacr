"""Focused regression coverage for tutorial-only TTS substitutions."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from pronunciation import (
    PRONUNCIATION_VERSION,
    PYPI_SPEECH,
    assert_pronunciation_safe,
    spoken_form,
)


def test_pronunciation_profile_has_a_stable_version() -> None:
    assert PRONUNCIATION_VERSION == "2026-08-28-pype-v11"


def test_english_hardware_terms_stay_cohesive() -> None:
    speech = spoken_form(
        "The CPU-compatible build uses NVIDIA hardware and CUDA libraries.",
        "en",
    )

    assert speech == (
        "The CPU compatible build uses NVIDIA hardware and "
        "[CUDA](/kˈuːdᵊ/) libraries."
    )
    assert "C P U" not in speech
    assert "en vid ee uh" not in speech
    assert "koo duh" not in speech


def test_spacr_pronunciation_is_unchanged() -> None:
    speech = spoken_form("spaCR", "en")

    assert speech == "spacer"


def test_pypi_is_one_continuous_word_in_every_language() -> None:
    for language in (
            "en", "es", "fr", "hi", "it", "pt-BR", "ja", "zh-CN",
            "de", "sv", "is", "nb", "ko", "da"):
        speech = spoken_form("Install spaCR from PyPI.", language)
        assert PYPI_SPEECH[language] in speech
        assert_pronunciation_safe("Install spaCR from PyPI.", speech)

    assert PYPI_SPEECH["en"] == "[pype](/pˈIp/)"


def test_pypi_guard_rejects_hyphenated_and_letter_by_letter_aliases() -> None:
    for alias in (
            "pie-P-I", "P-Y-P-I", "P Y P I", "P, Y, P, I",
            "pai pi", "PyPie", "P Y P E", "P-Y-P-E",
            "pypie", "pypie and PyPie", "pypie and P-Y-P-I"):
        try:
            assert_pronunciation_safe("Install from PyPI.", alias)
        except ValueError as error:
            assert "PyPI" in str(error) or "pype" in str(error)
        else:  # pragma: no cover - guard failure
            raise AssertionError(f"unsafe PyPI alias was accepted: {alias}")


def test_pypi_guard_requires_one_exact_token_per_display_occurrence() -> None:
    assert_pronunciation_safe(
        "Install from PyPI, then verify PyPI metadata.",
        "Install from [pype](/pˈIp/), then verify [pype](/pˈIp/) metadata.",
    )
    for speech in (
            "Install from [pype](/pˈIp/) [pype](/pˈIp/) [pype](/pˈIp/).",
            "Install from [pype](/pˈIp/)."):
        try:
            assert_pronunciation_safe(
                "Install from PyPI, then verify PyPI metadata.", speech)
        except ValueError as error:
            assert "pype" in str(error)
        else:  # pragma: no cover - guard failure
            raise AssertionError(
                "PyPI display/speech occurrence mismatch was accepted: "
                f"{speech}"
            )


def test_pype_token_does_not_join_the_following_word() -> None:
    for speech in (
            "[pype](/pˈIp/) provides a package",
            "[pype](/pˈIp/) package"):
        assert_pronunciation_safe("PyPI provides a package", speech)


def test_pypi_visual_spec_uses_the_exact_spoken_token() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "production" / "01_pypi_github" / "scenes.json").read_text(
            encoding="utf-8"))
    display = re.compile(r"(?<!\w)PyPI(?!\w)")
    failures = []
    for index, scene in enumerate(spec["scenes"], start=1):
        expected_count = len(display.findall(scene["narration"]))
        if not expected_count:
            continue
        speech = scene.get("speech_text", "")
        try:
            assert_pronunciation_safe(scene["narration"], speech)
        except ValueError:
            failures.append(index)
            continue
    assert not failures, f"visual scenes without exact pype: {failures}"


def test_every_narrated_catalog_uses_one_exact_pype_per_display_token() -> None:
    root = Path(__file__).resolve().parents[1]
    display = re.compile(r"(?<!\w)PyPI(?!\w)")
    failures = []
    for path in sorted((root / "catalog").glob("lessons_*.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        lesson = next(
            item for item in catalog["lessons"]
            if item["id"] == "01_pypi_github"
        )
        for index, scene in enumerate(lesson["scenes"], start=1):
            expected = len(display.findall(scene["narration"]))
            speech = scene.get("speech_text", "")
            try:
                assert_pronunciation_safe(scene["narration"], speech)
            except ValueError:
                failures.append(f"{path.name}: scene {index}")
    assert not failures, failures


def test_conda_forge_is_two_naturally_separated_words_in_every_language() -> None:
    for language in (
            "en", "es", "fr", "hi", "it", "pt-BR", "ja", "zh-CN",
            "de", "sv", "is", "nb", "ko", "da"):
        speech = spoken_form(
            "Use conda-forge, then run conda install conda-forge::spacr.",
            language,
        )
        assert speech.count("Conda Forge") == 2, (language, speech)
        assert "conda-forge" not in speech.casefold(), (language, speech)
        assert "::" not in speech, (language, speech)
        assert "Conda  Forge" not in speech, (language, speech)


def test_qt_is_spoken_as_two_letters_in_every_language() -> None:
    from pronunciation import QT_SPEECH

    for language, letter_names in QT_SPEECH.items():
        speech = spoken_form("Qt and QT", language)
        assert speech == f"{letter_names} and {letter_names}", (language, speech)


def test_every_narrated_catalog_uses_current_qt_and_conda_forge_speech() -> None:
    root = Path(__file__).resolve().parents[1]
    failures = []
    for path in sorted((root / "catalog").glob("lessons_*.json")):
        language = path.stem.removeprefix("lessons_")
        catalog = json.loads(path.read_text(encoding="utf-8"))
        for lesson in catalog["lessons"]:
            for index, scene in enumerate(lesson["scenes"], start=1):
                narration = scene["narration"]
                if not (
                    re.search(r"(?i:\bqt\b)", narration)
                    or re.search(r"(?i:\bconda-forge\b)", narration)
                ):
                    continue
                expected = spoken_form(narration, language)
                if scene.get("speech_text") != expected:
                    failures.append(
                        f"{path.name}: {lesson['id']} scene {index}"
                    )
    assert not failures, failures


def test_us_scientific_terms_keep_complete_unstressed_endings() -> None:
    speech = spoken_form(
        "An ASSAY and assays classify each classifier and CLASSIFIERS.",
        "en",
    )

    assert speech == (
        "An [ASSAY](/ˈæsA/) and [assays](/ˈæsAz/) "
        "[classify](/klˈæsəfI/) each [classifier](/klˈæsəfIəɹ/) "
        "and [CLASSIFIERS](/klˈæsəfIəɹz/)."
    )


def test_uk_scientific_terms_keep_complete_unstressed_endings() -> None:
    speech = spoken_form(
        "An assay and ASSAYS classify each CLASSIFIER and classifiers.",
        "en",
        dialect="uk",
    )

    assert speech == (
        "An [assay](/əsˈA/) and [ASSAYS](/əsˈAz/) "
        "[classify](/klˈasɪfI/) each [CLASSIFIER](/klˈasɪfIə/) "
        "and [classifiers](/klˈasɪfIəz/)."
    )


def test_word_boundaries_help_misaki_without_changing_captions() -> None:
    speech = spoken_form(
        "Qt QT qt preprocessing pretrained hyperparameter hyperparameters "
        "and colocalization",
        "en",
    )

    assert speech == (
        "Q T Q T Q T pre-processing pre-trained hyper-parameter "
        "hyper-parameters and co-localization"
    )


def test_cohesive_initialisms_are_not_expanded_with_spaces() -> None:
    text = (
        "API CLI CV ML PCA PC1 PC2 PC3 2D 3D xD QC XG GPU HPC SSH HTML "
        "PDF CSV FDR 4PL"
    )

    speech = spoken_form(text, "en")

    assert speech == text


def test_catalog_terms_with_missing_misaki_entries_have_spoken_forms() -> None:
    speech = spoken_form(
        "Conda conda-forge t-SNE Napari Slurm Timelapse backends denoising "
        "heatmaps queried copied AnnData measurements dot db overfitting",
        "en",
    )

    assert speech == (
        "Conda Conda Forge "
        "[t-SNE](/tˌi snˈi/) [Napari](/nəpˈɑɹi/) "
        "[Slurm](/slˈɜɹm/) time lapse back ends de-noising heat maps "
        "queryd copyd Anne data measurements dot D B over-fitting"
    )


def test_repeated_product_names_use_cohesive_stress() -> None:
    speech = spoken_form(
        "GitHub macOS PyTorch XGBoost Cellpose", "en"
    )

    assert speech == (
        "[GitHub](/ɡˈɪthˌʌb/) [macOS](/mˈækOˌɛs/) "
        "[PyTorch](/pˈItˌɔɹʧ/) "
        "[XGBoost](/ˌɛksʤˌibˈust/) [Cellpose](/sˈɛlpˌOz/)"
    )


def test_dialect_specific_fallbacks_preserve_full_word_endings() -> None:
    speech = spoken_form(
        "unnotarized untruncated rerunnable multi-rater auditable rotatable "
        "endoplasmic",
        "en",
        dialect="uk",
    )

    assert speech == (
        "[unnotarized](/ʌnnˈQtəɹIzd/) "
        "[untruncated](/ʌntɹʌŋkˈAtɪd/) "
        "[rerunnable](/ɹˌiːɹˈʌnəbᵊl/) "
        "[multi-rater](/mˌʌltiɹˈAtə/) "
        "[auditable](/ˈɔːdɪtəbᵊl/) "
        "[rotatable](/ɹQtˈAtəbᵊl/) "
        "[endoplasmic](/ˌɛndQplˈazmɪk/)"
    )


def test_power_design_slash_is_spoken_as_a_conjunction() -> None:
    speech = spoken_form("Open Power / Design.", "en")

    assert speech == "Open Power and Design."


def test_uk_axis_z_uses_the_british_letter_name() -> None:
    speech = spoken_form("Choose X, Y, and Z.", "en", dialect="uk")

    assert speech == "Choose X, Y, and [Z](/zˈɛd/)."


def test_torch_backend_flag_is_verbalized_after_compound_resegmentation() -> None:
    speech = spoken_form("pass --torch-backend auto", "en")

    assert speech == "pass dash dash torch back end auto"


def test_spacr_cli_entry_point_does_not_retain_a_spoken_hyphen() -> None:
    speech = spoken_form("Run spacr-run with --upgrade.", "en")

    assert speech == "Run spacer run with dash dash upgrade."
