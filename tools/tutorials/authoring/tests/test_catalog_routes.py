"""Structural contracts for consolidated tutorial workflows."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACR_REPO = Path(os.environ.get(
    "SPACR_REPO", "/mnt/firecuda2/codex/repo/spacr"
))
sys.path.insert(0, str(ROOT / "tools"))

from translate_pre_app import (  # noqa: E402
    CATALOG_TITLES,
    MANUAL_BRANCH,
    MANUAL_DOCTOR,
    SECTION_LABELS,
    SERIES_TITLES,
    reviewed_lesson_ids,
    reviewed_records,
)


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
    "hit_list": "regression",
    "illumination": "measure",
    "image_scatter": "umap",
    "methods_export": "regression",
    "ml_analyze": "classify_merged",
    "model_compare": "make_masks",
    "model_zoo": "make_masks",
    "motility": "measure",
    "napari_bridge": "make_masks",
    "parameter_sweep": "regression",
    "pca": "umap",
    "timelapse": "mask",
    "train_cellpose": "make_masks",
    "volcano_explorer": "regression",
}

FOLD_CATALOG_KEY_ALIASES = {"cellpose_all": "cellpose_masks"}
HOST_APP_KEY_ALIASES = {
    "classify": "classify_merged",
    "image_umap": "umap",
}
HOSTED_MODE_LESSONS = {
    "classify": "classify_merged",
    "ml_analyze": "classify_merged",
    "parameter_sweep": "regression",
}


def _live_folded_lesson_hosts() -> dict[str, str]:
    """Derive current tutorial routes from spaCR's physical fold owners."""
    code = """
import json
import spacr.qt
spacr.qt.register_self_registering_modules()
from spacr.qt.widgets.fold_strip import folded_modules
print(json.dumps({
    key: entry[3].rsplit('.', 1)[-1]
    for key, entry in folded_modules().items()
}))
"""
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SPACR_REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    physical = json.loads(result.stdout.strip().splitlines()[-1])
    routes = {
        FOLD_CATALOG_KEY_ALIASES.get(key, key):
        HOST_APP_KEY_ALIASES.get(module_name, module_name)
        for key, module_name in physical.items()
    }
    overlap = set(routes) & set(HOSTED_MODE_LESSONS)
    assert not overlap, f"host modes became physical folds: {sorted(overlap)}"
    routes.update(HOSTED_MODE_LESSONS)
    return routes


def _live_home_inventory() -> dict:
    """Return the registry and Home category bands from the current app."""
    code = """
import json
import spacr.qt
spacr.qt.register_self_registering_modules()
from spacr.qt.app import APPS, home_categories
print(json.dumps({
    "count": len(APPS),
    "keys": [row[0] for row in APPS],
    "categories": home_categories(APPS),
    "core_names": [row[1] for row in APPS if row[3] == "Core"],
}))
"""
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SPACR_REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_authoring_route_map_matches_the_live_fold_owners() -> None:
    assert FOLDED_LESSON_HOSTS == _live_folded_lesson_hosts()


def test_lesson_inventory_matches_every_live_primary_and_folded_workflow() -> None:
    """Every current workflow has one lesson; retired workflows have none."""
    catalog = json.loads(
        (ROOT / "catalog" / "lessons_en.json").read_text(encoding="utf-8")
    )
    routed = [
        lesson["app_key"] for lesson in catalog["lessons"]
        if lesson.get("app_key")
    ]
    expected = set(_live_home_inventory()["keys"]) | set(
        FOLDED_LESSON_HOSTS
    )
    assert len(routed) == len(set(routed)) == len(expected) == 67
    assert set(routed) == expected
    assert [
        lesson["id"] for lesson in catalog["lessons"]
        if not lesson.get("app_key")
    ] == [
        "01_pypi_github", "02_conda_install", "03_pip_install",
        "04_platform_installers", "05_home", "06_api",
    ]


def test_every_full_catalog_preserves_the_exact_host_routes() -> None:
    for path in sorted((ROOT / "catalog").glob("lessons_*.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        actual = {
            lesson["app_key"]: lesson.get("host_app_key")
            for lesson in catalog["lessons"]
            if lesson.get("host_app_key")
        }
        assert actual == FOLDED_LESSON_HOSTS, path.name


def test_caption_catalogs_have_the_complete_lesson_inventory() -> None:
    english = json.loads(
        (ROOT / "catalog" / "lessons_en.json").read_text(encoding="utf-8"))
    expected = {
        lesson["id"]: len(lesson["scenes"])
        for lesson in english["lessons"]
    }
    for path in sorted((ROOT / "catalog").glob("captions_*.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        actual = {
            lesson["id"]: len(lesson["scenes"])
            for lesson in catalog["lessons"]
        }
        assert actual == expected, path.name


def test_all_eight_narrated_catalogs_are_complete_and_fully_reviewed() -> None:
    """Ratcheted authority: every narrated locale covers every lesson."""
    english = json.loads(
        (ROOT / "catalog" / "lessons_en.json").read_text(encoding="utf-8")
    )
    expected = {
        lesson["id"]: len(lesson["scenes"])
        for lesson in english["lessons"]
    }
    paths = sorted((ROOT / "catalog").glob("lessons_*.json"))
    assert {path.name for path in paths} == {
        "lessons_en.json", "lessons_es.json", "lessons_fr.json",
        "lessons_hi.json", "lessons_it.json", "lessons_ja.json",
        "lessons_pt-BR.json", "lessons_zh-CN.json",
    }
    for path in paths:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        actual = {
            lesson["id"]: len(lesson["scenes"])
            for lesson in catalog["lessons"]
        }
        assert actual == expected, path.name
        if path.name == "lessons_en.json":
            continue
        language = path.stem.removeprefix("lessons_")
        assert reviewed_lesson_ids(language) == set(expected), language


def test_generated_catalog_text_matches_effective_reviewed_authority() -> None:
    """Every maintained review must survive catalog regeneration exactly."""
    effective: dict[tuple[str, str], dict[str, dict]] = {}
    for _path, lesson_id, languages in reviewed_records():
        for language, review in languages.items():
            target = effective.setdefault(
                (language, lesson_id),
                {"fields": {}, "objectives": {}, "narrations": {}},
            )
            if isinstance(review, dict):
                for field in ("description", "prerequisite"):
                    if field in review:
                        target["fields"][field] = review[field]
                objectives = review.get("objectives", {})
                if isinstance(objectives, dict):
                    for index, value in objectives.items():
                        target["objectives"][int(index)] = value
                else:
                    target["objectives"] = {
                        index: value
                        for index, value in enumerate(objectives, start=1)
                    }
                narrations = review.get("narrations", {})
            else:
                narrations = review
            if isinstance(narrations, dict):
                for index, value in narrations.items():
                    target["narrations"][int(index)] = value
            else:
                target["narrations"] = {
                    index: value
                    for index, value in enumerate(narrations, start=1)
                }

    for language, narration in MANUAL_BRANCH.items():
        effective[(language, "01_pypi_github")]["narrations"][4] = narration
    for language, narration in MANUAL_DOCTOR.items():
        effective[(language, "02_conda_install")]["narrations"][6] = narration

    failures = []
    for language in sorted({key[0] for key in effective}):
        full = language in CATALOG_TITLES
        name = (
            f"lessons_{language}.json" if full
            else f"captions_{language}.json"
        )
        catalog = json.loads(
            (ROOT / "catalog" / name).read_text(encoding="utf-8")
        )
        by_id = {lesson["id"]: lesson for lesson in catalog["lessons"]}
        for (entry_language, lesson_id), expected in effective.items():
            if entry_language != language:
                continue
            lesson = by_id[lesson_id]
            if full:
                for field, value in expected["fields"].items():
                    if lesson[field] != value:
                        failures.append(f"{name}: {lesson_id} {field}")
                for index, value in expected["objectives"].items():
                    if lesson["objectives"][index - 1] != value:
                        failures.append(
                            f"{name}: {lesson_id} objective {index}"
                        )
            for index, value in expected["narrations"].items():
                if lesson["scenes"][index - 1]["narration"] != value:
                    failures.append(f"{name}: {lesson_id} scene {index}")
    assert not failures, failures


def test_narrated_catalog_navigation_metadata_is_fully_localized() -> None:
    english = json.loads(
        (ROOT / "catalog" / "lessons_en.json").read_text(encoding="utf-8")
    )
    source_sections = {
        lesson["id"]: lesson["section"] for lesson in english["lessons"]
    }
    first_titles = {
        "es": "PyPI, GitHub y conda-forge",
        "fr": "PyPI, GitHub et conda-forge",
        "hi": "PyPI, GitHub और conda-forge",
        "it": "PyPI, GitHub e conda-forge",
        "ja": "PyPI、GitHub、conda-forge",
        "pt-BR": "PyPI, GitHub e conda-forge",
        "zh-CN": "PyPI、GitHub 和 conda-forge",
    }
    feature_titles = {
        "es": "Diccionario de características",
        "fr": "Dictionnaire des caractéristiques",
        "hi": "विशेषता शब्दकोश",
        "it": "Dizionario delle caratteristiche",
        "ja": "特徴量辞書",
        "pt-BR": "Dicionário de características",
        "zh-CN": "特征词典",
    }

    for language in sorted(CATALOG_TITLES):
        catalog = json.loads((
            ROOT / "catalog" / f"lessons_{language}.json"
        ).read_text(encoding="utf-8"))
        assert catalog["title"] == CATALOG_TITLES[language]
        assert [item["title"] for item in catalog["series"]] == list(
            SERIES_TITLES[language]
        )
        by_id = {lesson["id"]: lesson for lesson in catalog["lessons"]}
        assert by_id["01_pypi_github"]["title"] == first_titles[language]
        assert by_id["62_feature_dictionary"]["title"] == feature_titles[language]
        for lesson in catalog["lessons"]:
            assert lesson["section"] == SECTION_LABELS[language][
                source_sections[lesson["id"]]
            ]


def test_localized_installation_terms_do_not_regress_to_literal_translations() -> None:
    """Keep the nightly branch label and current scene semantics exact."""
    banned = (
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
    for path in sorted((ROOT / "catalog").glob("lessons_*.json")):
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


def test_conda_forge_publication_truth_is_ratcheted_across_every_catalog() -> None:
    """Keep the two installation lessons on the live, direct Conda route."""
    command = "conda install conda-forge::spacr"
    expected_catalogs = {
        "lessons_en.json", "lessons_es.json", "lessons_fr.json",
        "lessons_hi.json", "lessons_it.json", "lessons_ja.json",
        "lessons_pt-BR.json", "lessons_zh-CN.json",
        "captions_da.json", "captions_de.json", "captions_is.json",
        "captions_ko.json", "captions_nb.json", "captions_sv.json",
    }
    paths = sorted(
        list((ROOT / "catalog").glob("lessons_*.json"))
        + list((ROOT / "catalog").glob("captions_*.json"))
    )
    assert {path.name for path in paths} == expected_catalogs

    stale_phrases = (
        "does not publish spacr yet",
        "will become an option once",
        "todavía no publica spacr",
        "ne publie pas encore spacr",
        "non pubblica ancora spacr",
        "ainda não publica o spacr",
        "अभी spacr प्रकाशित नहीं करता",
        "まだ spacr が公開されていません",
        "尚未发布 spacr",
        "noch nicht veröffentlicht",
        "publicerar ännu inte spacr",
        "hefur ekki verið gefinn út enn",
        "ennå ikke publisert",
        "아직 spacr이 게시되지",
        "endnu ikke udgivet",
    )
    failures = []
    for path in paths:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        by_id = {lesson["id"]: lesson for lesson in catalog["lessons"]}
        orientation = by_id["01_pypi_github"]
        conda = by_id["02_conda_install"]
        if len(orientation["scenes"]) != 7 or len(conda["scenes"]) != 7:
            failures.append(f"{path.name}: installation scene count changed")
            continue
        if command not in orientation["scenes"][5]["narration"]:
            failures.append(f"{path.name}: lesson 01 lacks the official command")
        if command not in conda["scenes"][3]["narration"]:
            failures.append(f"{path.name}: lesson 02 lacks the official command")
        conda_prose = " ".join(
            scene["narration"] for scene in conda["scenes"]
        )
        if re.search(r"(?<!\w)pip(?!\w)", conda_prose, re.IGNORECASE):
            failures.append(f"{path.name}: lesson 02 still invokes pip")
        # Only the source overview and conda publication scene state package
        # availability. Branch guidance legitimately says that nightly holds
        # work that is not yet released.
        combined = " ".join(
            orientation["scenes"][index]["narration"]
            for index in (0, 5)
        ).casefold()
        for phrase in stale_phrases:
            if phrase.casefold() in combined:
                failures.append(f"{path.name}: stale claim {phrase!r}")
    assert not failures, failures


def test_conda_authoring_and_keyframes_use_only_the_official_command() -> None:
    command = "conda install conda-forge::spacr"
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            ROOT / "tools" / "build_full_catalog.py",
            ROOT / "tools" / "render_install_keyframes.py",
            ROOT / "tools" / "translate_pre_app.py",
        )
    }
    stale = (
        "conda install -c conda-forge spacr",
        "channel package not published yet",
        "does not publish spaCR yet",
        "pip installs spaCR today",
        "Install from PyPI",
        "Future channel",
    )
    for name, source in sources.items():
        assert command in source, name
        assert not any(phrase in source for phrase in stale), name


def test_installer_tutorial_keeps_acceleration_default_and_cpu_fallback() -> None:
    """Every catalog must preserve the hardware-selection contract."""
    paths = sorted(
        list((ROOT / "catalog").glob("lessons_*.json"))
        + list((ROOT / "catalog").glob("captions_*.json"))
    )
    failures = []
    for path in paths:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        lesson = next(
            item for item in catalog["lessons"]
            if item["id"] == "04_platform_installers"
        )
        windows = lesson["scenes"][2]["narration"]
        linux_default = lesson["scenes"][4]["narration"]
        linux_fallback = lesson["scenes"][5]["narration"]
        if "CUDA" not in windows or "CPU" not in windows:
            failures.append(
                f"{path.name}: Windows CUDA default or CPU fallback missing"
            )
        if "CUDA" not in linux_default:
            failures.append(f"{path.name}: Linux CUDA default missing")
        if "--torch-backend cpu" not in linux_fallback:
            failures.append(f"{path.name}: explicit CPU fallback missing")
        if "CUDA" not in linux_fallback:
            failures.append(f"{path.name}: fallback does not contrast CUDA")
    renderer = (
        ROOT / "tools" / "render_install_keyframes.py"
    ).read_text(encoding="utf-8")
    assert "CPU default" not in renderer
    assert renderer.count("Automatic CUDA") == 2
    assert not failures, failures


def test_home_tutorial_is_ratcheted_to_the_live_registry_and_categories() -> None:
    live = _live_home_inventory()
    assert live["count"] == 44
    assert [item[0] for item in live["categories"]] == [
        "Core", "Data", "Results & QC", "Explore", "Assays", "Design",
    ]
    assert live["core_names"] == [
        "Mask", "Measure", "Annotate", "Classify", "Map Barcodes",
        "Regression",
    ]

    english = json.loads(
        (ROOT / "catalog" / "lessons_en.json").read_text(encoding="utf-8")
    )
    home = next(
        lesson for lesson in english["lessons"] if lesson["id"] == "05_home"
    )
    overview = home["scenes"][0]["narration"]
    modules = home["scenes"][2]["narration"]
    performance = home["scenes"][3]["narration"]
    assert "all forty four registered spaCR modules" in overview
    assert "Core, Data, Results & QC, Explore, Assays, and Design" in overview
    assert all(name in modules for name in live["core_names"])
    assert "Hover over a tile" in modules
    assert "module description" in modules
    assert all(stage in modules for stage in ("Alpha", "Beta", "Stable"))
    assert "Preferences can hide Alpha or Beta modules" in modules
    assert "Stable modules remain visible" in modules
    assert "one Performance level selector" in performance
    assert all(label in performance for label in (
        "Laptop", "Extra Performance", "Performance", "Balanced",
        "Workstation",
    ))
    assert "not scientific settings or results" in performance


def test_every_home_localization_preserves_current_exact_ui_labels() -> None:
    paths = sorted(
        list((ROOT / "catalog").glob("lessons_*.json"))
        + list((ROOT / "catalog").glob("captions_*.json"))
    )
    labels = (
        "Core", "Data", "Results & QC", "Explore", "Assays", "Design",
    )
    failures = []
    for path in paths:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        lesson = next(
            item for item in catalog["lessons"] if item["id"] == "05_home"
        )
        if len(lesson["scenes"]) != 5:
            failures.append(f"{path.name}: Home scene count changed")
            continue
        overview = lesson["scenes"][0]["narration"]
        modules = lesson["scenes"][2]["narration"]
        performance = lesson["scenes"][3]["narration"]
        for label in labels:
            if label not in overview:
                failures.append(f"{path.name}: Home label {label!r} missing")
        for name in (
                "Mask", "Measure", "Annotate", "Classify", "Map Barcodes",
                "Regression", "Alpha", "Beta", "Stable"):
            if name not in modules:
                failures.append(f"{path.name}: module label {name!r} missing")
        for label in (
                "Laptop", "Extra Performance", "Performance", "Balanced",
                "Workstation"):
            if label not in performance:
                failures.append(
                    f"{path.name}: performance label {label!r} missing")
    assert not failures, failures


def test_authoritative_tutorial_text_uses_dependencies_without_modifier() -> None:
    paths = [ROOT / "tools" / "build_full_catalog.py"]
    paths.extend(sorted((ROOT / "catalog").glob("*.json")))
    paths.extend(sorted((ROOT / "localization" / "reviewed").glob("*.json")))
    failures = [
        str(path.relative_to(ROOT))
        for path in paths
        if "scientific dependencies" in path.read_text(
            encoding="utf-8"
        ).casefold()
    ]
    assert not failures, failures


def test_caption_only_regression_matches_the_complete_reviewed_source() -> None:
    review = json.loads((
        ROOT / "localization" / "reviewed"
        / "13_regression_auxiliary_captions.json"
    ).read_text(encoding="utf-8"))
    reviewed = review["lessons"]["13_regression"]["languages"]
    assert set(reviewed) == {"da", "de", "is", "ko", "nb", "sv"}
    for language, payload in reviewed.items():
        expected = payload["narrations"]
        assert len(expected) == 11
        catalog = json.loads((
            ROOT / "catalog" / f"captions_{language}.json"
        ).read_text(encoding="utf-8"))
        lesson = next(
            item for item in catalog["lessons"]
            if item["id"] == "13_regression"
        )
        actual = [scene["narration"] for scene in lesson["scenes"]]
        assert actual == expected, language
