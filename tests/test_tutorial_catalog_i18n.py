"""Translation and routing contracts for the authored tutorial catalogs."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog"
FULL_LOCALES = ("en", "es", "fr", "hi", "it", "ja", "pt-BR", "zh-CN")
CAPTION_LOCALES = ("da", "de", "is", "ko", "nb", "sv")

# The refreshed lesson's reviewed titles (tools/tutorials/lessons/reviews/
# 62_feature_dictionary.<locale>.json, 2026-09-25). The screen name stays the
# literal "Feature Dictionary" the recording shows, followed by a translated
# subtitle, and the older "Diccionario de características"-style titles are
# retired. Commit bd33b41ad bound the current practical walkthrough titles
# and all thirteen translations to the unchanged English lesson.
FEATURE_DICTIONARY_TITLES = {
    "es": "Feature Dictionary: buscar definiciones y unidades",
    "fr": "Feature Dictionary : trouver des définitions et des unités",
    "hi": "Feature Dictionary: परिभाषाएँ और इकाइयाँ खोजें",
    "it": "Feature Dictionary: trovare definizioni e unità",
    "ja": "Feature Dictionary：定義と単位を調べる",
    "pt-BR": "Feature Dictionary: encontrar definições e unidades",
    "zh-CN": "Feature Dictionary：查找定义和单位",
}


sys.path.insert(0, str(ROOT / "tools" / "tutorials"))
from archived import ARCHIVED_IDS  # noqa: E402
from archive_lessons import full_file  # noqa: E402


def _catalog(prefix: str, locale: str) -> dict:
    """The published catalog, archived lessons included.

    The library shows the visible lessons only; the archived ones keep their
    entries under ``tools/tutorials/archive/`` and return to their places on
    restore. These contracts are about the whole published inventory, so
    the archived entries are put back first; the deployed files themselves
    are checked by :func:`test_deployed_catalogs_hide_exactly_the_archived_lessons`.
    """
    return full_file(f"catalog/{prefix}_{locale}.json")


def _before_navigator_arrival(lessons):
    """Validate lesson 88 separately and retain the historical 84-lesson contract."""
    arrivals = [lesson for lesson in lessons if lesson["id"] == "88_navigator"]
    assert len(arrivals) == 1
    arrival = arrivals[0]
    assert lessons[-1] is arrival
    assert arrival["id"] not in ARCHIVED_IDS
    assert arrival["number"] == 88
    assert arrival["slug"] == "navigator"
    assert arrival["series"] == 1
    assert arrival["section"] == "spaCR"
    assert arrival["app_key"] is None
    assert arrival.get("host_app_key") is None
    assert len(arrival["scenes"]) == 18
    return lessons[:-1]


def test_deployed_catalogs_hide_exactly_the_archived_lessons():
    for prefix, locales in (("lessons", FULL_LOCALES), ("captions", CAPTION_LOCALES)):
        for locale in locales:
            path = CATALOG_DIR / f"{prefix}_{locale}.json"
            deployed = [lesson["id"] for lesson in
                        json.loads(path.read_text(encoding="utf-8"))["lessons"]]
            lessons = _catalog(prefix, locale)["lessons"]
            published = [lesson["id"] for lesson in lessons]
            assert deployed == [i for i in published if i not in ARCHIVED_IDS], path.name
            assert len(_before_navigator_arrival(lessons)) == 84


def test_all_authored_catalogs_match_the_84_lesson_inventory_and_routes():
    """Every locale must carry the complete lesson and folded-host topology."""
    english = _before_navigator_arrival(_catalog("lessons", "en")["lessons"])
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
    # Tutorial wave 3 (2026-10-04) withdrew 83_plasmodium and 84_candida,
    # whose pages moved behind Show alpha species.
    # Wave 4 appended the two allow-listed alpha lessons.
    assert len(ids) == len(set(ids)) == 84
    assert ids[:3] == ['01_pypi_github', '02_install_spacr', '04_platform_installers']
    assert not {'02_conda_install', '03_pip_install'} & set(ids)
    assert ids[-4:] == ['82_toxoplasma', '85_host_pathogen',
                        '86_alpha_organism_modules', '87_alpha_features']
    assert not {'83_plasmodium', '84_candida'} & set(ids)
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
        lessons = _before_navigator_arrival(_catalog("lessons", locale)["lessons"])
        assert [lesson["id"] for lesson in lessons] == ids, locale
        assert [len(lesson["scenes"]) for lesson in lessons] == scene_counts, locale
        routes = {
            lesson["id"]: lesson.get("host_app_key")
            for lesson in lessons
            if lesson.get("host_app_key") is not None
        }
        assert routes == expected_routes, locale

    for locale in CAPTION_LOCALES:
        lessons = _before_navigator_arrival(_catalog("captions", locale)["lessons"])
        assert [lesson["id"] for lesson in lessons] == ids, locale
        assert [len(lesson["scenes"]) for lesson in lessons] == scene_counts, locale


def test_spoken_pypi_is_the_reviewed_pype_form_in_every_spoken_locale():
    """Keep the explicit v13 speech policy and every localized scene check.

    English PyPI includes the final I from the 2026-10-21 review; the other
    reviewed spoken forms remain unchanged. Published captions retain PyPI,
    and this test neither edits media nor treats registered English fallback
    as independently translated speech. English timing-sidecar speech remains
    outside the catalog when no speech_text is supplied, as before.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tutorial_pronunciation",
        ROOT / "tools" / "tutorials" / "authoring" / "tools" / "pronunciation.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    assert rule.PRONUNCIATION_VERSION == "2026-10-21-pypi-yokogawa-v13"
    expected_forms = {
        "en": "[pie pee eye](/pˌI pˌi ˈI/)", "es": "paip", "fr": "païpe",
        "hi": "पाइप", "it": "paip", "pt-BR": "paip", "ja": "パイプ",
        "zh-CN": "派普", "de": "Peip", "sv": "pajp", "is": "pæp",
        "nb": "paip", "ko": "파이프", "da": "pajp",
    }
    assert rule.PYPI_SPEECH == expected_forms
    # Japanese particles and Chinese prose can touch a Latin brand directly.
    # Keep ASCII identifier boundaries while checking those genuine mentions.
    display_token = re.compile(r"(?<![A-Za-z0-9_])PyPI(?![A-Za-z0-9_])")
    english = {lesson['id']: lesson for lesson in _catalog('lessons', 'en')['lessons']}
    compatibility = json.loads((CATALOG_DIR.parent / 'translation-compatibility.json').read_text(encoding="utf-8"))
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
        form = expected_forms[locale]
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
        "02_install_spacr",
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
    source_pypi_counts = [scene['narration'].count('PyPI')
                          for scene in english_release['scenes']]
    assert sum(source_pypi_counts) > 0
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
        assert [scene['narration'].count('PyPI') for scene in release['scenes']] == source_pypi_counts, locale
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
    navigation_path = CATALOG_DIR.parent / 'module_navigation.js'
    navigation = json.loads(re.search(
        r'window\.SPACR_TUTORIAL_NAVIGATION\s*=\s*Object\.freeze\((\{.*\})\);',
        navigation_path.read_text(encoding="utf-8"), re.DOTALL).group(1))
    review = json.loads((ROOT / 'tools/tutorials/lessons/navigation_categories.review.json').read_text(encoding="utf-8"))
    main = next(section for section in navigation['sections'] if section['id'] == 'main')
    assert [group['title'] for group in main['groups']] == review['source']
    assert set(navigation['labels']) == set(FULL_LOCALES + CAPTION_LOCALES)
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
        # Lesson section identifiers remain stable through publication. The
        # player displays the generated navigation labels, whose category
        # values must match their complete source-bound review.
        assert not (set(navigation['labels'][locale]) & english_sections), locale
        assert [navigation['labels'][locale][group['label_index']]
                for group in main['groups']] == review['labels'][locale], locale
        dictionary_review = json.loads((ROOT / 'tools/tutorials/lessons/reviews'
                                        / f'62_feature_dictionary.{locale}.json').read_text(encoding="utf-8"))
        assert dictionary_review['english_sha256'] == hashlib.sha256(
            json.dumps(json.loads((ROOT / 'tools/tutorials/lessons/62_feature_dictionary.json').read_text(encoding="utf-8")),
                       sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert dictionary_review['title'] == FEATURE_DICTIONARY_TITLES[locale], locale
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
