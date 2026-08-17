"""``docs/source/setting_animations.rst`` is GENERATED, and it went stale.

The page is written by ``tools/generate_setting_animations.py``. Nothing
enforced that, so it drifted twice, in opposite directions, and both drifts
survived for over a week:

  * the page listed ``all_to_mip`` and ``pick_slice`` under Z projection and
    the four ``remove_border_*`` aliases under mask filtering, months after
    those settings were retired -- documentation naming settings spaCR does
    not have;
  * the page's INTRO PROSE was the correct one and the generator's was not.
    The maintainer hand-fixed the page in 87f3d2b7 and 250ec3d3 to describe
    the tooltip footer that ``spacr/qt/widgets/hover_tooltip.py`` actually
    implements, while the generator still described a purple dot belonging to
    ``spacr/qt/widgets/animation_link.py``, a widget that has been deleted.

The second one is why this file exists rather than a tidier page. Every
regeneration pass used ``--only``, so the page was never rewritten and the
hand fix was never overwritten -- but the next full run would have replaced
correct documentation with a description of a deleted widget, silently. A
hand-edit to a generated file is work that disappears the next time somebody
runs the tool, and until now nothing said so out loud.

WHAT THESE TESTS DO NOT DO. They do not make a full regeneration safe. 60 of
94 GIFs still come out byte-different under a different Pillow build, and a
full run still rewrites every hash in the manifest. These make a full run
UNNECESSARY for the page: ``--docs-only`` writes the page and nothing else,
and the first test below is what proves the committed page is exactly what
that command produces.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

gen = pytest.importorskip("generate_setting_animations")

PAGE = ROOT / "docs" / "source" / "setting_animations.rst"
MANIFEST = ROOT / "spacr" / "resources" / "setting_animations" / "manifest.json"
GIFS = ROOT / "spacr" / "resources" / "setting_animations" / "gifs"

# The two keys in the gallery that are deliberately not spaCR settings: the
# alignment controls of the image-registration screen, which own their own
# widgets. tests/test_setting_animations.py pins the same pair for the
# runtime registry.
CUSTOM_ALIGN_CONTROLS = {"overlap", "blend"}

FIX_IT = (
    "The page is generated. Edit tools/generate_setting_animations.py, then "
    "run `python tools/generate_setting_animations.py --docs-only` and commit "
    "both. Editing the .rst by hand is reverted by the next run of the tool."
)


def _settings_lines(page_text: str) -> list[list[str]]:
    """Every ``**Settings:**`` line of the gallery, as lists of setting keys."""
    return [
        re.findall(r"``([^`]+)``", line)
        for line in page_text.splitlines()
        if line.startswith("**Settings:**")
    ]


def _manifest_settings() -> dict[str, list[str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        entry["slug"]: list(entry["settings"])
        for entry in manifest["animations"]
    }


def test_the_committed_gallery_is_what_the_generator_writes_today(
    tmp_path, monkeypatch
):
    """The shipped page must equal a freshly generated one, character for character.

    This is the whole point of the file. Without it, the page and the tool
    that writes it can disagree indefinitely -- and the disagreement is
    invisible until somebody runs a full regeneration, at which point the
    tool wins and any hand fix is gone. Fails today on any drift in the
    prose, the anchors, the image paths or the setting keys.
    """
    fresh = tmp_path / "setting_animations.rst"
    monkeypatch.setattr(gen, "DOCS_PAGE", fresh)
    gen._write_docs_gallery(gen._specs())

    assert fresh.read_text(encoding="utf-8") == PAGE.read_text(
        encoding="utf-8"
    ), FIX_IT


def test_no_animation_documents_a_setting_the_shipped_manifest_does_not_carry():
    """The generator's specs and the shipped manifest must name the same keys.

    This is the drift the page-diff test alone cannot see, because the page
    and the generator are stale in the SAME direction here. Commit 60f7798e
    removed five retired keys from manifest.json line-wise on 2026-08-11 and
    left the generator declaring them, so a full regeneration would have
    reverted that commit -- putting the manifest back to 142 setting keys and
    failing test_setting_animations.py, which asserts 137.
    """
    shipped = _manifest_settings()
    specs = {spec.slug: list(spec.settings) for spec in gen._specs()}

    assert set(specs) == set(shipped)
    disagreeing = {
        slug: (specs[slug], shipped[slug])
        for slug in sorted(specs)
        if specs[slug] != shipped[slug]
    }
    assert not disagreeing, (
        "a full regeneration would rewrite these manifest entries: "
        f"{disagreeing}"
    )
    assert sum(len(keys) for keys in specs.values()) == 137


def test_the_gallery_names_only_settings_spacr_actually_has():
    """Every key printed in the page must be a live setting, or a named exception.

    Guards the page DIRECTLY, without going through the generator, so it
    holds even if someone edits the .rst by hand. This is the test that would
    have caught ``all_to_mip`` on the day it was retired instead of nine days
    later: a reader who looks up a documented setting and finds spaCR does
    not accept it learns to distrust the settings next to it.
    """
    from spacr.settings import descriptions, expected_types, tooltips

    live = set(expected_types) | set(descriptions) | set(tooltips)
    named = {key for line in _settings_lines(PAGE.read_text("utf-8")) for key in line}

    assert named, "the gallery printed no setting keys at all"
    assert named - live == CUSTOM_ALIGN_CONTROLS, FIX_IT


def test_every_animation_in_the_page_is_one_the_registry_ships():
    """94 anchors, 94 images, 94 ``**Settings:**`` lines, one per animation.

    The anchors are link targets: ``SettingAnimation.docs_url`` builds a URL
    from each slug and the Qt tooltip's **API** word opens it. A page that
    lost an anchor would send a reader to the top of a 1000-line gallery.
    """
    from spacr.setting_animations import setting_animations

    page = PAGE.read_text(encoding="utf-8")
    animations = setting_animations()

    assert len(_settings_lines(page)) == len(animations) == 94
    for animation in animations:
        assert page.count(f".. _{animation.docs_anchor}:") == 1
        assert page.count(f"gifs/{animation.slug}.gif") == 1


def test_the_page_describes_the_reveal_the_tooltip_widget_implements(
    tmp_path, monkeypatch
):
    """The intro must describe the tooltip footer, not the deleted purple dot.

    ``spacr/qt/widgets/animation_link.py`` -- the purple dot -- is gone;
    ``spacr/qt/widgets/hover_tooltip.py`` reveals the animation from a teal
    **Animation** word in the tooltip footer, per setting. Until 2026-08-17
    the PAGE said the true thing and the GENERATOR said the dead one, so a
    full run would have replaced correct documentation with a description of
    a widget that does not exist. Both sides are asserted here, separately
    from the character-for-character diff above, so that particular revert
    fails with its own name on it instead of a 1000-line diff.
    """
    fresh = tmp_path / "setting_animations.rst"
    monkeypatch.setattr(gen, "DOCS_PAGE", fresh)
    gen._write_docs_gallery(gen._specs())

    assert not (ROOT / "spacr" / "qt" / "widgets" / "animation_link.py").exists()
    for source, text in (
        ("the committed page", PAGE.read_text(encoding="utf-8")),
        ("the generator's output", fresh.read_text(encoding="utf-8")),
    ):
        intro = text.split("The diagrams use a shared biological grammar")[0]
        assert "purple dot" not in text, f"{source} describes a deleted widget"
        assert "**Animation**" in intro, source
        assert "tooltip's footer" in intro, source


def test_regenerating_the_docs_page_touches_no_gif_and_no_manifest(
    tmp_path, monkeypatch
):
    """``--docs-only`` writes the page and nothing else. That is what makes it safe.

    A full run re-encodes all 94 GIFs and rewrites every hash in the
    manifest, which is why nobody ran one and why the page rotted instead.
    This pins the property that lets the page be regenerated in the same
    commit as a generator fix: manifest bytes unchanged, gif directory
    listing and sizes unchanged.
    """
    fresh = tmp_path / "setting_animations.rst"
    monkeypatch.setattr(gen, "DOCS_PAGE", fresh)
    before_manifest = MANIFEST.read_bytes()
    before_gifs = sorted((p.name, p.stat().st_size) for p in GIFS.iterdir())

    assert gen.main(["--docs-only"]) == 0

    assert fresh.read_text(encoding="utf-8") == PAGE.read_text(encoding="utf-8")
    assert MANIFEST.read_bytes() == before_manifest
    assert sorted((p.name, p.stat().st_size) for p in GIFS.iterdir()) == before_gifs
    assert PAGE.read_text(encoding="utf-8").startswith("Setting animation gallery")


def test_docs_only_refuses_to_be_combined_with_only(capsys):
    """``--only SLUG --docs-only`` is a mistake, not a narrower page.

    ``--only`` names GIFs to re-encode and ``--docs-only`` encodes none, so
    the combination silently ignores the slugs the caller typed. Rejecting it
    is cheaper than a maintainer believing one animation was rebuilt. The
    MESSAGE is asserted, not just the exit code: argparse also exits 2 for an
    unrecognised flag, so a code-only check passes on a build where
    ``--docs-only`` does not exist at all.
    """
    with pytest.raises(SystemExit) as failure:
        gen.main(["--docs-only", "--only", "z_projection"])

    assert failure.value.code == 2
    complaint = capsys.readouterr().err
    assert "--docs-only writes no GIFs" in complaint
    assert "unrecognized arguments" not in complaint
