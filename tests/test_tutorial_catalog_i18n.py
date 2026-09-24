"""Translation and routing contracts for the authored tutorial catalogs."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog"
FULL_LOCALES = ("en", "es", "fr", "hi", "it", "ja", "pt-BR", "zh-CN")
CAPTION_LOCALES = ("da", "de", "is", "ko", "nb", "sv")

# The refreshed lesson's reviewed titles (tools/tutorials/lessons/reviews/
# 62_feature_dictionary.<locale>.json, 2026-09-09). The screen name stays the
# literal "Feature Dictionary" the recording shows, followed by a translated
# subtitle, and the older "Diccionario de características"-style titles are
# retired. They reached this tree when candidate 8738b_pd was published on
# 2026-09-15.
FEATURE_DICTIONARY_TITLES = {
    "es": "Feature Dictionary: consultar significados, unidades y funciones de cálculo",
    "fr": "Feature Dictionary : consulter les définitions, les unités et les fonctions de calcul",
    "hi": "Feature Dictionary: अर्थ, इकाइयाँ और गणना करने वाले फ़ंक्शन समझें",
    "it": "Feature Dictionary: consultare significati, unità e funzioni di calcolo",
    "ja": "Feature Dictionary：意味・単位・計算関数を調べる",
    "pt-BR": "Feature Dictionary: consultar significados, unidades e funções de cálculo",
    "zh-CN": "Feature Dictionary：查询含义、单位和计算函数",
}


def _catalog(prefix: str, locale: str) -> dict:
    path = CATALOG_DIR / f"{prefix}_{locale}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_authored_catalogs_match_the_85_lesson_inventory_and_routes():
    """Every locale must carry the complete lesson and folded-host topology."""
    english = _catalog("lessons", "en")["lessons"]
    ids = [lesson["id"] for lesson in english]
    scene_counts = [len(lesson["scenes"]) for lesson in english]
    expected_routes = {
        lesson["id"]: lesson["host_app_key"]
        for lesson in english
        if lesson.get("host_app_key") is not None
    }
    # The authored inventory grew from 75 to 77 when the Coming soon screens
    # were folded into the public player. Diffing the id lists of every one of
    # the fourteen catalogs against the previous release shows two arrivals and
    # no departures: 76_ops and 77_embeddings, both scene-less placeholders
    # carrying the unavailable-route copy rather than a recorded walkthrough.
    assert len(ids) == len(set(ids)) == 85
    assert set(ids[-4:]) == {
        '82_toxoplasma', '83_plasmodium', '84_candida', '85_host_pathogen'}
    # Folded-host routes grew from 25 to 39 in the same change, because the
    # catalogs are now stamped from the navigation tree instead of carrying a
    # hand-maintained subset of it. Diffing the route maps names fourteen
    # arrivals and no departures or rehostings. Thirteen of them were already
    # submodules of an existing host in the navigation tree and were simply
    # missing the catalog field -- 28_training_runs, 31_external_masks,
    # 33_plate_viewer, 35_converter, 51_control_charts, 53_prediction_profiler,
    # 56_lineage, 57_layer_viewer, 61_tabulate, 63_small_multiples,
    # 65_feature_explorer, 66_outliers and 71_investigate_hit -- and the
    # fourteenth is the new 76_ops, which the navigation tree folds under Mask.
    assert len(expected_routes) == 42
    assert expected_routes['85_host_pathogen'] == 'toxoplasma'
    assert expected_routes['61_tabulate'] == 'db_browser'
    assert expected_routes['63_small_multiples'] == 'graph_builder'

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


def test_spoken_pypi_is_the_reviewed_pype_form_in_every_spoken_locale():
    """Every scene that shows PyPI speaks the reviewed "pype" form, never "pypie".

    The rule is ``PRONUNCIATION_VERSION = "2026-08-28-pype-v11"`` in the
    renderer's pronunciation module: PyPI is the single syllable "pype" in
    English and ``PYPI_SPEECH[locale]`` elsewhere, "never PyPy, pypie, a
    sequence of letters, or two paused syllables". This test was
    ``test_spoken_pypi_is_exactly_one_continuous_pypie_token`` and pinned the
    pre-rule "pypie" from 2026-08-26. The maintainer's decision of 2026-09-15
    (question tool) was '"pype" (Recommended)': keep the published narration
    and move the tests to the 08-28 rule, re-rendering nothing.

    The English catalog carries no ``speech_text`` for these scenes (English
    speech is in the hosted timing sidecars, gated by the same rule), so it is
    checked only if it ever carries some. Translated locales must carry it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tutorial_pronunciation",
        ROOT / "tools" / "tutorials" / "authoring" / "tools" / "pronunciation.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    assert rule.PRONUNCIATION_VERSION == "2026-08-28-pype-v11"
    display_token = re.compile(r"(?<!\w)PyPI(?!\w)")
    english = {lesson['id']: lesson for lesson in _catalog('lessons', 'en')['lessons']}
    compatibility = json.loads((CATALOG_DIR.parent / 'translation-compatibility.json').read_text())
    registered = {(row['lesson'], row['language']) for row in compatibility['entries']
                  if row['status'] == 'english_fallback' and row['reason']}

    for locale in FULL_LOCALES:
        lessons = _catalog("lessons", locale)["lessons"]
        named, fallback = [], []
        for lesson in lessons:
            scenes = [scene for scene in lesson['scenes']
                      if display_token.search(scene.get('narration', ''))]
            voices = lesson.get('narration_voices')
            if scenes and locale != 'en' and voices is not None and locale not in voices:
                assert lesson == english[lesson['id']]
                assert voices.get('en') and (lesson['id'], locale) in registered
                fallback.extend(scenes)
            else:
                named.extend(scenes)
        assert named or fallback, locale
        if locale == "en" and all("speech_text" not in scene for scene in named):
            continue
        form = rule.PYPI_SPEECH[locale]
        for scene in named:
            speech = scene.get("speech_text")
            assert speech is not None, (locale, scene["narration"][:60])
            assert speech.count(form) == len(display_token.findall(scene["narration"])), locale
            assert not rule._REJECTED_PYPI_ALIAS.search(speech), locale
        speech_all = "\n".join(
            scene.get("speech_text", "") for lesson in lessons for scene in lesson["scenes"])
        assert "pypie" not in speech_all.casefold(), locale


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
    # Scenes whose ENGLISH names the nightly branch; derived, not pinned. The
    # 2026-09-11 re-recording made lesson 01 eight scenes and dropped the old
    # closing "nightly only when..." sentence, so only scene 4 names it now.
    english_release = next(
        lesson for lesson in _catalog("lessons", "en")["lessons"]
        if lesson["id"] == "01_pypi_github"
    )
    nightly_scenes = [
        index for index, scene in enumerate(english_release["scenes"])
        if "nightly" in scene["narration"]
    ]
    assert nightly_scenes
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
        assert display.count("PyPI") == 3, locale
        assert "pypie" not in display, locale
        assert "GitHub" not in release["scenes"][1]["narration"], locale
        assert "conda-forge" not in release["scenes"][1]["narration"].casefold(), locale
        for index in nightly_scenes:
            assert "nightly" in release["scenes"][index]["narration"], locale

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

def test_current_english_module_objectives_remain_explicit():
    """English content checks remain required even with translation fallback."""
    lessons = {row['id']: row for row in _catalog("lessons", "en")['lessons']}
    # The repaired phrases this pinned ("appropriate validation controls" in
    # 16_activation's objectives, "Import third-party images" in
    # 31_external_masks' description) were replaced when those lessons were
    # re-recorded on 2026-09-09 (tools/tutorials/lessons/16_activation.json,
    # 31_external_masks.json). The published candidate carries the refreshed
    # wording, so the pins follow it; the retired mistranslation stays banned.
    activation = ' '.join(lessons['16_activation']['objectives']).lower()
    external = ' '.join(lessons['31_external_masks']['objectives']).lower()
    for concept in ('crop archive', 'model', 'preprocessing', 'channel saliency', 'image overlay', 'saved grids'):
        assert concept in activation
    for concept in ('intensity images', 'label masks', 'channels', 'output location', 'preview', 'measure', 'output project'):
        assert concept in external
    assert 'Turn third-party images' not in json.dumps(lessons['31_external_masks'])
