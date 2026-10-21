"""A bump redraws the README images after it restamps the deck.

The README cover is drawn from the deck's title slide. A bump that
restamped the slide and left the README alone shipped a stale cover, and
the release CI's README check failed on it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.test_release_metadata_sync import CITATION, _tree

ROOT = Path(__file__).resolve().parents[1]


def _release():
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper_images", ROOT / "packaging" / "release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installer_readmes(root, release):
    paths = {"en": root / "README.rst"}
    paths.update({code: root / "docs/i18n/readme" / f"README.{code}.rst"
                  for code in release.LOCALIZED_README_CODES})
    for code, path in paths.items():
        lines = [f"Preserved {code} prose Å: historical spaCR 1.5.1.0.",
                 "docker run ghcr.io/einarolafsson/spacr:1.5.1.0",
                 release.README_BEGIN, ""]
        for index, (_label, suffix) in enumerate(release.PLATFORMS):
            lines.extend([
                f".. |installer-{index}| image:: icons/{index}.png",
                f"   :alt: translated-{code} Å spaCR 1.5.1.0",
                f"   :target: {release.RELEASE_DOWNLOAD_ROOT}/v1.5.1.0/spaCR-1.5.1.0-{suffix}",
                "",
            ])
        lines.extend([release.README_END, f"Preserved {code} final prose Å."])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def test_the_steps_restamp_the_deck_then_redraw_the_readme():
    release = _release()
    assert release.BUMP_ARTEFACT_STEPS == (release.restamp_info_deck,
                                           release.rebuild_readme_images)


def test_a_bump_runs_every_step_with_the_new_version(tmp_path, monkeypatch):
    release = _release()
    root = _tree(tmp_path, recipe=False)
    citation = root / "CITATION.cff"
    citation.write_text(CITATION.replace("spaCR 1.5.0.9.", "spaCR 1.5.1.0."),
                        encoding="utf-8")
    ran = []
    monkeypatch.setattr(release, "BUMP_ARTEFACT_STEPS", (
        lambda path, version: ran.append(("deck", path, version)),
        lambda path, version: ran.append(("images", path, version))))
    release.bump_release(root / "setup.py", citation, "1.5.1.1",
                         release_date="2026-09-30")
    assert ran == [("deck", root, "1.5.1.1"), ("images", root, "1.5.1.1")]


def test_the_images_step_runs_the_tool_in_the_tree(tmp_path, monkeypatch):
    release = _release()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "build_readme_images.py").write_text(
        "from pathlib import Path\nPath('built').write_text('yes')\n", encoding="utf-8")
    (tmp_path / "README.rst").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(release, "README_IMAGE_MODULES", ())
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is True
    assert (tmp_path / "built").read_text(encoding="utf-8") == "yes"


def test_the_images_step_skips_without_its_modules_or_tool(tmp_path,
                                                          monkeypatch, capsys):
    release = _release()
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is False
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "build_readme_images.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    (tmp_path / "README.rst").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(release, "README_IMAGE_MODULES", ("no_such_module_x",))
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is False
    assert "no_such_module_x" in capsys.readouterr().err
    monkeypatch.setattr(release, "README_IMAGE_MODULES", ())
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is False
    assert "failed (3)" in capsys.readouterr().err


def test_versioned_sources_are_rewritten_before_the_single_image_build(tmp_path, monkeypatch):
    release = _release()
    root = _tree(tmp_path, recipe=False)
    citation = root / "CITATION.cff"
    citation.write_text(CITATION.replace("spaCR 1.5.0.9.", "spaCR 1.5.1.0."),
                        encoding="utf-8")
    readmes = _installer_readmes(root, release)
    originals = {code: path.read_text(encoding="utf-8") for code, path in readmes.items()}
    readme = readmes["en"]
    source = root / "packaging" / "readme_source.yaml"
    source.parent.mkdir(exist_ok=True)
    source.write_text("docker run ghcr.io/einarolafsson/spacr:1.5.1.0\n")
    tools = root / "tools"
    tools.mkdir()
    (tools / "build_readme_images.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "with Path('rendered').open('a') as stream:\n"
        "    stream.write(json.dumps({'args': sys.argv[1:], "
        "'source': Path('packaging/readme_source.yaml').read_text(), "
        "'readmes': {str(p): p.read_text(encoding='utf-8') for p in "
        "[Path('README.rst'), *sorted(Path('docs/i18n/readme').glob('README.*.rst'))]}}) + '\\n')\n")
    monkeypatch.setattr(release, "docker_tag_pages", lambda path: (*readmes.values(), source))
    monkeypatch.setattr(release, "README_IMAGE_MODULES", ())
    monkeypatch.setattr(release, "BUMP_ARTEFACT_STEPS", (
        lambda path, version: None, release.rebuild_readme_images))
    release.bump_release(root / "setup.py", citation, "1.5.1.1",
                         release_date="2026-09-30")
    rendered = [json.loads(line) for line in (root / "rendered").read_text().splitlines()]
    assert len(rendered) == 1
    assert rendered[0]["args"] == ["--refresh-generated"]
    assert rendered[0]["source"] == "docker run ghcr.io/einarolafsson/spacr:1.5.1.1\n"
    assert len(rendered[0]["readmes"]) == 10
    for code, path in readmes.items():
        text = rendered[0]["readmes"][str(path.relative_to(root))]
        assert text == path.read_text(encoding="utf-8")
        before, block = text.split(release.README_BEGIN)
        block, after = block.split(release.README_END)
        old_before, old_block = originals[code].split(release.README_BEGIN)
        _old_block, old_after = old_block.split(release.README_END)
        assert before == old_before.replace("ghcr.io/einarolafsson/spacr:1.5.1.0",
                                            "ghcr.io/einarolafsson/spacr:1.5.1.1")
        assert after == old_after
        assert f"translated-{code} Å spaCR 1.5.1.1" in block
        for _label, suffix in release.PLATFORMS:
            assert f"{release.RELEASE_DOWNLOAD_ROOT}/v1.5.1.1/spaCR-1.5.1.1-{suffix}" in block
        assert "1.5.1.0" not in block
    assert "1.5.1.1" in readme.read_text()


@pytest.mark.parametrize("defect", ["missing", "malformed"])
def test_invalid_installer_readmes_refuse_the_bump_before_any_write_or_render(
        tmp_path, monkeypatch, defect):
    release = _release()
    root = _tree(tmp_path, recipe=False)
    readmes = _installer_readmes(root, release)
    last = readmes[release.LOCALIZED_README_CODES[-1]]
    if defect == "missing":
        last.unlink()
        error, message = FileNotFoundError, "missing localized README"
    else:
        last.write_text(last.read_text(encoding="utf-8").replace(
            "/v1.5.1.0/", "/v1.5.0.9/", 1), encoding="utf-8")
        error, message = ValueError, "malformed installer link"
    facade = root / "_version.py"
    facade.write_text('__version__ = "1.5.1.0"\n', encoding="utf-8")
    paths = [root / "setup.py", root / "CITATION.cff", facade,
             *(path for path in readmes.values() if path.exists())]
    before = {path: path.read_bytes() for path in paths}
    rendered = []
    monkeypatch.setattr(release, "BUMP_ARTEFACT_STEPS", (
        lambda path, version: rendered.append("deck"),
        lambda path, version: rendered.append("images")))
    with pytest.raises(error, match=message):
        release.bump_release(root / "setup.py", root / "CITATION.cff", "1.5.1.1",
                             package_version_path=facade, release_date="2026-09-30")
    assert {path: path.read_bytes() for path in paths} == before
    assert rendered == []
    assert last.exists() == (defect != "missing")
