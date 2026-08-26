from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# These lessons teach a capability that still exists, but no longer has a
# Home tile.  The map is deliberately explicit: a retired key may remain in
# the course only when the lesson names the current module through which the
# user reaches it.  Adding a lesson key to this table is a product decision,
# not a way to silence the inventory test below.
FOLDED_LESSON_HOSTS = {
    "activation": "classify_merged",
    "agreement": "annotate",
    "anndata_export": "measure",
    "barcode_qc": "map_barcodes",
    "cellpose_masks": "make_masks",
    "classifier_evaluation": "classify_merged",
    "classify": "classify_merged",
    "curate": "make_masks",
    "explain_cv": "classify_merged",
    "illumination": "measure",
    "image_scatter": "umap",
    "ml_analyze": "classify_merged",
    "model_compare": "make_masks",
    "model_zoo": "make_masks",
    "motility": "measure",
    "parameter_sweep": "regression",
    "pca": "umap",
    "timelapse": "mask",
    "train_cellpose": "make_masks",
    "volcano_explorer": "regression",
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
    assert sum(len(lesson["scenes"]) for lesson in catalog["lessons"]) == 507
    assert len(languages) == 8
    assert len(voices) == 50
    assert not (live.RETIRED_VOICES & set(voices))
    assert live.EXPECTED_CACHE_KEY in (
        tutorial_root / "index.html"
    ).read_text(encoding="utf-8")


def _registered_gui_modules():
    """Return the live Home-module keys in a clean Python process."""
    code = """
import json
import spacr.qt
spacr.qt.register_self_registering_modules()
from spacr.qt.app import APPS
print(json.dumps(sorted({row[0] for row in APPS})))
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        check=True, capture_output=True, text=True,
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


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
    lessons = _tutorial_catalog()["lessons"]
    routed = [lesson for lesson in lessons if lesson.get("app_key")]
    lesson_keys = [lesson["app_key"] for lesson in routed]

    assert len(lesson_keys) == len(set(lesson_keys)), (
        "each module or folded workflow must have exactly one lesson")
    assert set(lesson_keys) == registry | set(FOLDED_LESSON_HOSTS)

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
        expected = FOLDED_LESSON_HOSTS.get(key)
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


def test_tutorial_release_skill_has_no_template_placeholders():
    skill_path = (
        ROOT / ".claude" / "skills" / "tutorial-release-audit" / "SKILL.md"
    )
    skill = skill_path.read_text(encoding="utf-8")
    contract = (skill_path.parent / "references" / "release-contract.md").read_text(
        encoding="utf-8"
    )
    assert "TODO" not in skill
    assert "tools/verify_tutorial_live.py" in skill
    assert "tools/sample_tutorial_frames.py" in skill
    assert "verify_audio_release.py" in contract
