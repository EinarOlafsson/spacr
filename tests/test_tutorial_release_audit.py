from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# Catalog identities retained for capabilities whose internal fold key has
# changed. Cellpose Workbench exposes ``cellpose_all``; the published lesson
# has kept its existing ``cellpose_masks`` URL and media directory.
FOLD_CATALOG_KEY_ALIASES = {"cellpose_all": "cellpose_masks"}

# These three are host modes rather than entries in a ``FOLDED_APPS`` tuple.
# They therefore cannot be derived from ``folded_modules()``. Keeping this
# small exception table separate makes a newly folded module fail the guard
# below instead of letting a duplicated hand-maintained inventory drift.
HOSTED_MODE_LESSONS = {
    "classify": "classify_merged",
    "ml_analyze": "classify_merged",
    "parameter_sweep": "regression",
}

# Most host module names equal their public app keys. These are the two
# deliberate exceptions in the current registry.
HOST_APP_KEY_ALIASES = {
    "classify": "classify_merged",
    "image_umap": "umap",
}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


live = _load("verify_tutorial_live", "tools/verify_tutorial_live.py")
frames = _load("sample_tutorial_frames", "tools/sample_tutorial_frames.py")


def test_release_audit_parsers_pin_the_current_inventory():
    tutorial_root = ROOT / "docs" / "source" / "_extra" / "tutorials"
    catalog = frames.load_catalog(tutorial_root / "lesson_catalog.js")
    languages, voices = live._voice_inventory(
        (tutorial_root / "voice_catalog.js").read_text(encoding="utf-8")
    )
    assert len(catalog["lessons"]) == 73
    assert sum(len(lesson["scenes"]) for lesson in catalog["lessons"]) == 508
    assert len(languages) == 8
    assert len(voices) == 50
    assert not (live.RETIRED_VOICES & set(voices))
    assert live.EXPECTED_CACHE_KEY in (
        tutorial_root / "index.html"
    ).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _live_gui_inventory():
    """Return live Home keys and physical fold ownership in a clean process."""
    code = """
import json
import spacr.qt
spacr.qt.register_self_registering_modules()
from spacr.qt.app import APPS
from spacr.qt.widgets.fold_strip import folded_modules
print(json.dumps({
    "registry": sorted({row[0] for row in APPS}),
    "folded": {
        key: entry[3].rsplit(".", 1)[-1]
        for key, entry in folded_modules().items()
    },
}))
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _registered_gui_modules():
    """Return the live Home-module keys."""
    return set(_live_gui_inventory()["registry"])


def _folded_lesson_hosts():
    """Translate physical fold ownership into stable tutorial routes."""
    routes = {}
    for physical_key, module_name in _live_gui_inventory()["folded"].items():
        lesson_key = FOLD_CATALOG_KEY_ALIASES.get(physical_key, physical_key)
        host_key = HOST_APP_KEY_ALIASES.get(module_name, module_name)
        assert lesson_key not in routes, (
            f"multiple folds resolve to tutorial key {lesson_key!r}")
        routes[lesson_key] = host_key
    overlap = set(routes) & set(HOSTED_MODE_LESSONS)
    assert not overlap, f"host modes became physical folds: {sorted(overlap)}"
    routes.update(HOSTED_MODE_LESSONS)
    return routes


def _tutorial_catalog():
    return frames.load_catalog(
        ROOT / "docs" / "source" / "_extra" / "tutorials" /
        "lesson_catalog.js"
    )


def test_every_module_and_fold_has_exactly_one_current_tutorial_route():
    """Keep the course synchronized with tiles *and* consolidated modules.

    The old subset assertion allowed any number of retired tiles to remain
    presented as standalone modules.  A lesson now either names a live Home
    key or carries the exact ``host_app_key`` through which its folded
    workflow is reached.  Unknown keys, missing modules, duplicate routes,
    and an obsolete route with no explicit host all fail together.
    """
    registry = _registered_gui_modules()
    folded_hosts = _folded_lesson_hosts()
    lessons = _tutorial_catalog()["lessons"]
    routed = [lesson for lesson in lessons if lesson.get("app_key")]
    lesson_keys = [lesson["app_key"] for lesson in routed]

    assert len(lesson_keys) == len(set(lesson_keys)), (
        "each module or folded workflow must have exactly one lesson")
    assert set(lesson_keys) == registry | set(folded_hosts)

    failures = []
    for lesson in routed:
        key = lesson["app_key"]
        host = lesson.get("host_app_key")
        if key in registry:
            if host is not None:
                failures.append(
                    f"{lesson['id']}: live module {key!r} must not name "
                    f"a folded host ({host!r})")
            continue
        expected = folded_hosts.get(key)
        if host != expected:
            failures.append(
                f"{lesson['id']}: folded workflow {key!r} must name "
                f"host_app_key={expected!r}, found {host!r}")
        if host not in registry:
            failures.append(
                f"{lesson['id']}: host {host!r} is not a live module")
    assert not failures, failures


def test_every_translated_tutorial_catalog_has_the_english_structure():
    """Translations may change prose, never lesson identity or routing."""
    catalog_dir = (
        ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog"
    )
    english = json.loads((catalog_dir / "lessons_en.json").read_text(
        encoding="utf-8"))
    expected = [
        (lesson["id"], lesson["number"], lesson.get("app_key"),
         lesson.get("host_app_key"), len(lesson["scenes"]))
        for lesson in english["lessons"]
    ]
    failures = []
    for path in sorted(catalog_dir.glob("lessons_*.json")):
        translated = json.loads(path.read_text(encoding="utf-8"))
        actual = [
            (lesson["id"], lesson["number"], lesson.get("app_key"),
             lesson.get("host_app_key"), len(lesson["scenes"]))
            for lesson in translated["lessons"]
        ]
        if actual != expected:
            failures.append(path.name)

    english_scenes = {
        lesson["id"]: len(lesson["scenes"])
        for lesson in english["lessons"]
    }
    for path in sorted(catalog_dir.glob("captions_*.json")):
        translated = json.loads(path.read_text(encoding="utf-8"))
        actual = {
            lesson["id"]: len(lesson["scenes"])
            for lesson in translated["lessons"]
        }
        if actual != english_scenes:
            failures.append(path.name)
    assert not failures, f"tutorial catalogs with structural drift: {failures}"


def test_player_routes_folded_lessons_through_their_current_host():
    """The public player must not advertise a retired key as a module."""
    player = (ROOT / "docs" / "source" / "_extra" / "tutorials" /
              "app_v2.js").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "source" / "_extra" / "tutorials" /
            "index.html").read_text(encoding="utf-8")

    assert 'id="lesson-route"' in page
    assert "lesson?.host_app_key || lesson?.app_key" in player
    assert "item.app_key === lesson.host_app_key && !item.host_app_key" in player
    assert "localizedLesson(route.host.id)" in player
    assert "elements.content.dataset.appKey = route.appKey" in player


def test_every_spoken_pypi_is_the_single_word_pypie():
    """Keep every narration language on the user's exact brand reading."""
    catalog_dir = (
        ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog"
    )
    english = json.loads((catalog_dir / "lessons_en.json").read_text(
        encoding="utf-8"))
    source_lesson = next(
        lesson for lesson in english["lessons"]
        if lesson["id"] == "01_pypi_github"
    )
    display_pypi = re.compile(r"(?<!\w)PyPI(?!\w)")
    pypi_scenes = {
        index: len(display_pypi.findall(scene["narration"]))
        for index, scene in enumerate(source_lesson["scenes"])
        if display_pypi.search(scene["narration"])
    }
    assert pypi_scenes, "the installation lesson no longer names PyPI"

    exact_pypie = re.compile(r"(?<!\w)pypie(?!\w)")
    any_case_pypie = re.compile(r"(?<!\w)pypie(?!\w)", re.IGNORECASE)
    separator_char = r"[\s,./_—–-]"
    separator = rf"{separator_char}+"
    banned_alias = re.compile(
        rf"(?<!\w)(?:(?i:pypi|pype)|"
        rf"(?i:(?:pie|pai|paj|pæ|paï){separator}p"
        rf"(?:{separator_char}*[ieí])?|"
        rf"p{separator}y{separator}p{separator}[ie]))(?!\w)",
        flags=re.IGNORECASE,
    )
    for rejected in (
            "PyPie", "pype", "P Y P E", "P-Y-P-I", "P, Y, P, I",
            "pypie and pype", "pypie and PyPie"):
        assert (len(exact_pypie.findall(rejected)) != 1
                or len(any_case_pypie.findall(rejected)) != 1
                or banned_alias.search(rejected))
    failures = []
    for path in sorted(catalog_dir.glob("lessons_*.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        lesson = next(
            item for item in catalog["lessons"]
            if item["id"] == "01_pypi_github"
        )
        for index, expected_count in pypi_scenes.items():
            speech = lesson["scenes"][index].get("speech_text", "")
            if (len(exact_pypie.findall(speech)) != expected_count
                    or len(any_case_pypie.findall(speech)) != expected_count
                    or banned_alias.search(speech)):
                failures.append(f"{path.name}:scene-{index + 1}")
    assert not failures, f"PyPI speech is not continuous 'pypie': {failures}"

    published = _tutorial_catalog()
    for lesson in published["lessons"]:
        lesson.pop("poster", None)
        lesson.pop("silent", None)
    assert published == english, (
        "the public lesson_catalog.js does not match lessons_en.json")


def test_localized_installation_lessons_keep_release_terms_semantically_exact():
    """Reject known literal translations of branch and packaging terms.

    ``nightly`` is a branch label, not a time of day.  Scene 2 describes the
    PyPI release header only; stale translations that still tell users to
    open GitHub and conda-forge no longer match the current visual or lesson.
    """
    catalog_dir = (
        ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog"
    )
    banned = (
        "PyPI, GitHub, and conda-forge",
        "only use packaging after release",
        "只使用包装发布后",
        "installator",
        "solo di notte",
        "somente à noite",
        "sólo por la noche",
        "chaque nuit seulement",
        "Télégraphie d'un modèle",
        "Meter as paradas",
    )
    failures = []
    for path in sorted(catalog_dir.glob("lessons_*.json")):
        if path.name == "lessons_en.json":
            continue
        catalog = json.loads(path.read_text(encoding="utf-8"))
        lesson = next(
            item for item in catalog["lessons"]
            if item["id"] == "01_pypi_github"
        )
        scene_2 = lesson["scenes"][1]["narration"]
        if "GitHub" in scene_2 or "conda-forge" in scene_2.casefold():
            failures.append(f"{path.name}: lesson 01 scene 2 is stale")
        for scene_index in (3, 6):
            if "nightly" not in lesson["scenes"][scene_index]["narration"]:
                failures.append(
                    f"{path.name}: lesson 01 scene {scene_index + 1} must "
                    "retain the branch label 'nightly'")
        serialized = json.dumps(catalog, ensure_ascii=False)
        for phrase in banned:
            if phrase == "installator":
                present = re.search(
                    r"(?<!\w)installator(?!\w)", serialized,
                    flags=re.IGNORECASE,
                ) is not None
            else:
                present = phrase.casefold() in serialized.casefold()
            if present:
                failures.append(f"{path.name}: known-bad phrase {phrase!r}")
    assert not failures, failures


def test_scene_midpoint_stays_inside_speech():
    scene = {"speech_start": 14.105, "speech_end": 20.26}
    midpoint = frames.scene_midpoint(scene)
    assert scene["speech_start"] < midpoint < scene["speech_end"]
    assert midpoint == pytest.approx(17.1825)


def test_frame_sampler_recreates_a_reviewable_still(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for visual audit sampling")
    video = tmp_path / "source.mp4"
    image = tmp_path / "scene.jpg"
    subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=1", "-pix_fmt",
         "yuv420p", str(video)],
        check=True,
    )
    frames.extract_frame(video, 0.5, image, 160)
    assert image.is_file()
    assert image.stat().st_size > 0
