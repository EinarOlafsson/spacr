"""Contracts for packaged visual-setting animations and their registry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageSequence

from spacr.setting_animations import (
    animation_for_setting,
    animations_by_setting,
    setting_animations,
    validate_setting_animation_assets,
)


ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "docs" / "source" / "setting_animations.rst"
TEMPLATES = ROOT / "tools" / "setting_animation_templates"
MANIFEST = ROOT / "spacr" / "resources" / "setting_animations" / "manifest.json"


def test_registry_has_complete_unique_exact_key_mapping():
    animations = setting_animations()
    by_setting = animations_by_setting()

    assert len(animations) == 94
    assert len(by_setting) == 143
    assert len({animation.slug for animation in animations}) == 94
    assert animation_for_setting("merge_edge_pathogen_cells").slug == (
        "merge_edge_pathogen_cells"
    )
    assert animation_for_setting("n_neighbors").slug == "n_neighbors"
    assert animation_for_setting("N_NEIGHBORS") is None
    assert animation_for_setting("not_a_setting") is None


def test_every_asset_is_square_animated_and_matches_manifest_hash():
    summary = validate_setting_animation_assets(check_hashes=True)
    assert summary["animations"] == 94
    assert summary["setting_keys"] == 143
    assert summary["bytes"] > 0

    for animation in setting_animations():
        assert animation.path.is_file()
        assert animation.path.suffix == ".gif"
        assert animation.relative_file.startswith("gifs/")
        assert ".." not in Path(animation.relative_file).parts
        with Image.open(animation.path) as image:
            assert image.size == (360, 360)
            frames = sum(1 for _frame in ImageSequence.Iterator(image))
            assert frames == animation.frames
            assert frames >= 4
            assert animation.unique_frames >= 4


def test_mapped_keys_are_real_settings_or_explicit_align_controls():
    from spacr.settings import descriptions, expected_types, tooltips

    known = set(expected_types) | set(descriptions) | set(tooltips)
    custom_align_controls = {"overlap", "blend"}
    assert set(animations_by_setting()) - known == custom_align_controls


def test_docs_gallery_has_one_stable_anchor_and_image_per_animation():
    gallery = GALLERY.read_text(encoding="utf-8")
    for animation in setting_animations():
        assert gallery.count(f".. _{animation.docs_anchor}:") == 1
        assert gallery.count(
            f"gifs/{animation.slug}.gif"
        ) == 1
        assert animation.docs_url.endswith("#" + animation.docs_anchor)


def test_packaging_rules_include_manifest_and_all_gifs():
    manifest_rules = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "recursive-include spacr/resources/setting_animations *.gif *.json" in (
        manifest_rules
    )
    assert "resources/setting_animations/*.json" in setup
    assert "resources/setting_animations/gifs/*.gif" in setup


def test_reviewed_svg_templates_are_safe_and_recorded_in_manifest():
    expected = {
        "all.svg",
        "cell.svg",
        "cell_all.svg",
        "golgi.svg",
        "nucleus.svg",
        "pathogen.svg",
    }
    paths = sorted(TEMPLATES.glob("*.svg"))
    assert {path.name for path in paths} == expected

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    hashes = manifest["template_sha256"]
    assert set(hashes) == expected
    for path in paths:
        content = path.read_text(encoding="utf-8")
        lowered = content.lower()
        assert "<path" in lowered
        assert all(
            token not in lowered
            for token in ("<image", "<script", "<use", "xlink:href", "<!entity")
        )
        assert hashes[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
