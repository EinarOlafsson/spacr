"""Translation and routing contracts for the authored tutorial catalogs."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog"
FULL_LOCALES = ("en", "es", "fr", "hi", "it", "ja", "pt-BR", "zh-CN")
CAPTION_LOCALES = ("da", "de", "is", "ko", "nb", "sv")

FEATURE_DICTIONARY_TITLES = {
    "es": "Diccionario de características",
    "fr": "Dictionnaire des caractéristiques",
    "hi": "विशेषता शब्दकोश",
    "it": "Dizionario delle caratteristiche",
    "ja": "特徴量辞書",
    "pt-BR": "Dicionário de características",
    "zh-CN": "特征词典",
}


def _catalog(prefix: str, locale: str) -> dict:
    path = CATALOG_DIR / f"{prefix}_{locale}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_authored_catalogs_match_the_73_lesson_inventory_and_routes():
    """Every locale must carry the complete lesson and folded-host topology."""
    english = _catalog("lessons", "en")["lessons"]
    ids = [lesson["id"] for lesson in english]
    scene_counts = [len(lesson["scenes"]) for lesson in english]
    expected_routes = {
        lesson["id"]: lesson["host_app_key"]
        for lesson in english
        if lesson.get("host_app_key") is not None
    }
    assert len(ids) == len(set(ids)) == 73
    assert len(expected_routes) == 23

    for locale in FULL_LOCALES:
        lessons = _catalog("lessons", locale)["lessons"]
        assert [lesson["id"] for lesson in lessons] == ids, locale
        assert [len(lesson["scenes"]) for lesson in lessons] == scene_counts, locale
        routes = {
            lesson["id"]: lesson.get("host_app_key")
            for lesson in lessons
            if lesson.get("host_app_key") is not None
        }
        assert routes == expected_routes, locale

    for locale in CAPTION_LOCALES:
        lessons = _catalog("captions", locale)["lessons"]
        assert [lesson["id"] for lesson in lessons] == ids, locale
        assert [len(lesson["scenes"]) for lesson in lessons] == scene_counts, locale


def test_spoken_pypi_is_exactly_one_continuous_pypie_token():
    """Keep the release-site pronunciation stable in every spoken locale."""
    token = re.compile(r"(?<!\w)pypie(?!\w)")
    pypi_family = re.compile(r"(?i)(?<!\w)pypi\w*(?!\w)")
    split_spelling = re.compile(r"(?i)\bp\W+y\W+p\W+i\b")

    for locale in FULL_LOCALES:
        lessons = _catalog("lessons", locale)["lessons"]
        speech = "\n".join(
            scene.get("speech_text", "")
            for lesson in lessons
            for scene in lesson["scenes"]
        )
        assert token.findall(speech) == ["pypie"] * 4, locale
        assert pypi_family.findall(speech) == ["pypie"] * 4, locale
        assert not split_spelling.search(speech), locale


def test_caption_only_installation_lessons_keep_reviewed_display_copy():
    """Caption locales retain visible brands and current installation facts."""
    known_machine_literals = (
        "skal du kende de tre officielle",
        "kennen Sie die drei offiziellen Vertriebswege",
        "endurskoðað pakkann",
        "공식적인 배포 경로를 세 가지로 알고 있습니다",
        "Receptet til conda-forge",
        "vet du de tre officiella distributionsvägarna",
    )
    installation_ids = {
        "01_pypi_github",
        "02_conda_install",
        "03_pip_install",
        "04_platform_installers",
        "05_home",
    }
    for locale in CAPTION_LOCALES:
        catalog = _catalog("captions", locale)
        lessons = {
            lesson["id"]: lesson
            for lesson in catalog["lessons"]
            if lesson["id"] in installation_ids
        }
        assert set(lessons) == installation_ids, locale
        assert all(
            scene["narration"].strip()
            for lesson in lessons.values()
            for scene in lesson["scenes"]
        ), locale

        release = lessons["01_pypi_github"]
        display = "\n".join(scene["narration"] for scene in release["scenes"])
        assert display.count("PyPI") == 4, locale
        assert "pypie" not in display, locale
        assert "GitHub" not in release["scenes"][1]["narration"], locale
        assert "conda-forge" not in release["scenes"][1]["narration"].casefold(), locale
        assert "nightly" in release["scenes"][3]["narration"], locale
        assert "nightly" in release["scenes"][6]["narration"], locale

        serialized = json.dumps(lessons, ensure_ascii=False).casefold()
        assert not [
            phrase for phrase in known_machine_literals
            if phrase.casefold() in serialized
        ], locale


def test_localized_navigation_chrome_and_reviewed_copy_do_not_regress():
    """Reject the specific untranslated and literal mistranslations repaired."""
    english_sections = {
        "Core", "Segmentation models", "Results and quality control",
        "Toxoplasma assays", "Data and batch runs", "Data", "Explore",
        "Design",
    }
    first_lesson_bans = (
        " and conda-forge", "installator", "solo di notte",
        "somente à noite", "sólo por la noche", "chaque nuit",
    )
    global_bans = ("Télégraphie d'un modèle", "Meter as paradas")

    for locale in FULL_LOCALES[1:]:
        catalog = _catalog("lessons", locale)
        lessons = catalog["lessons"]
        assert not ({lesson["section"] for lesson in lessons} & english_sections), locale
        by_id = {lesson["id"]: lesson for lesson in lessons}
        assert (
            by_id["62_feature_dictionary"]["title"]
            == FEATURE_DICTIONARY_TITLES[locale]
        ), locale
        first = json.dumps(lessons[0], ensure_ascii=False).casefold()
        assert not [phrase for phrase in first_lesson_bans if phrase in first], locale
        complete = json.dumps(catalog, ensure_ascii=False)
        assert not [phrase for phrase in global_bans if phrase in complete], locale

    english = json.dumps(_catalog("lessons", "en"), ensure_ascii=False)
    assert "appropriate validation controls" in english
    assert "Import third-party images" in english
    assert "Turn third-party images" not in english
