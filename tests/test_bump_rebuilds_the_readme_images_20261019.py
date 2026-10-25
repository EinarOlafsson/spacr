"""A bump redraws the README images after it restamps the deck.

The README cover is drawn from the deck's title slide. A bump that
restamped the slide and left the README alone shipped a stale cover, and
the release CI's README check failed on it.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
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


def _prepared_release_tree(tmp_path, release):
    root = _tree(tmp_path, recipe=False)
    readmes = _installer_readmes(root, release)
    facade = root / "spacr" / "_version.py"
    facade.parent.mkdir()
    facade.write_text('__version__ = "1.5.1.0"\n', encoding="utf-8")
    sources = [root / "packaging/readme_source.yaml", *(
        root / "docs/i18n/readme/source" / f"{code}.yaml"
        for code in release.LOCALIZED_README_CODES)]
    for path in [*sources, *(root / path for path in release.DOCKER_TAG_PAGES)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("docker run ghcr.io/einarolafsson/spacr:1.5.1.0\n",
                        encoding="utf-8")
    deck = root / "docs/source/_static/deck/slides.json"
    deck.parent.mkdir(parents=True)
    deck.write_text(json.dumps({"version": "1.5.1.0"}), encoding="utf-8")
    tool = root / "tools/build_readme_images.py"
    tool.parent.mkdir()
    tool.write_text("import sys\nfrom pathlib import Path\n"
                    "assert sys.argv[1:] == ['--check']\n"
                    "raise SystemExit(int(Path('stale-images').exists()))\n",
                    encoding="utf-8")
    return root, readmes, facade, sources, deck


@pytest.mark.parametrize("stale", ["installer", "docker", "deck", "images",
                                   "facade", "citation"])
def test_same_version_resume_keeps_normal_preparation_for_stale_inputs(
        tmp_path, monkeypatch, stale):
    release = _release()
    root, readmes, facade, sources, deck = _prepared_release_tree(tmp_path, release)
    if stale == "installer":
        path = readmes[release.LOCALIZED_README_CODES[-1]]
        path.write_text(path.read_text(encoding="utf-8").replace(
            "1.5.1.0", "1.5.0.9"), encoding="utf-8")
    elif stale == "docker":
        sources[-1].write_text("docker run ghcr.io/einarolafsson/spacr:1.5.0.9\n",
                               encoding="utf-8")
    elif stale == "deck":
        deck.write_text(json.dumps({"version": "1.5.0.9"}), encoding="utf-8")
    elif stale == "images":
        (root / "stale-images").touch()
    elif stale == "facade":
        facade.write_text('__version__ = "1.5.0.9"\n', encoding="utf-8")
    else:
        path = root / "CITATION.cff"
        path.write_text(path.read_text(encoding="utf-8").replace(
            'version: "1.5.1.0"', 'version: "1.5.0.9"'), encoding="utf-8")
    called = []
    monkeypatch.setattr(release, "BUMP_ARTEFACT_STEPS", (
        lambda path, version: called.append(("deck", path, version)),
        lambda path, version: called.append(("images", path, version))))
    assert release.bump_release(
        root / "setup.py", root / "CITATION.cff", "1.5.1.0",
        package_version_path=facade, allow_current=True) == "1.5.1.0"
    assert called == [("deck", root, "1.5.1.0"),
                      ("images", root, "1.5.1.0")]
    assert release.verify_release_metadata(
        root / "setup.py", root / "CITATION.cff",
        package_version_path=facade) == "1.5.1.0"
    assert all(release._updated_readme_text(path, "1.5.1.0") ==
               path.read_text(encoding="utf-8") for path in readmes.values())
    assert all(release.docker_tag_versions(path.read_text(encoding="utf-8")) ==
               {"1.5.1.0"} for path in release.docker_tag_pages(root))


def test_prepared_current_release_resumes_in_a_real_lean_process_without_writes(tmp_path):
    import os
    import shutil
    import subprocess
    import venv

    lean = tmp_path / "lean"
    venv.EnvBuilder(with_pip=False).create(lean)
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    for name in ("packaging", "yaml"):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.submodule_search_locations
        shutil.copytree(next(iter(spec.submodule_search_locations)), dependencies / name)
    interpreter = lean / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    script = """
import importlib.util
import sys
from pathlib import Path
root = Path(sys.argv[1])
assert importlib.util.find_spec('PySide6') is None
spec = importlib.util.spec_from_file_location('lean_release', root / 'packaging/release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
def forbidden(*args, **kwargs):
    raise AssertionError('A prepared current release must not write or rerender')
Path.write_text = Path.write_bytes = forbidden
release.BUMP_ARTEFACT_STEPS = (forbidden,)
sys.argv = ['release.py', 'bump', '--allow-current', release.read_version(root / 'setup.py')]
raise SystemExit(release.main())
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(dependencies), str(ROOT))),
                       PYTHONDONTWRITEBYTECODE="1")
    before = subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff", "--no-renames"], cwd=ROOT)
    result = subprocess.run([str(interpreter), "-c", script, str(ROOT)], cwd=ROOT,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == _release().read_version(ROOT / "setup.py")
    assert result.stderr == ""
    assert subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff", "--no-renames"], cwd=ROOT) == before


def test_preparation_refreshes_existing_news_before_freezing_the_release_source():
    import shlex
    import yaml

    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    steps = jobs["bump"]["steps"]
    refresh = [index for index, step in enumerate(steps)
               if step.get("name") == "Refresh notes for releases already published"]
    assert len(refresh) == 1
    version = next(index for index, step in enumerate(steps) if step.get("id") == "version")
    commit = next(index for index, step in enumerate(steps) if step.get("id") == "commit")
    finalize = next(index for index, step in enumerate(steps) if step.get("id") == "finalize")
    assert refresh[0] < version < commit < finalize
    assert steps[refresh[0]]["env"]["GH_TOKEN"] == "${{ github.token }}"
    staging = next(line for line in steps[commit]["run"].splitlines()
                   if line.strip().startswith("git add -u --"))
    assert "spacr/resources/release_notes.json" in shlex.split(staging)
    after_publication = jobs["release-notes"]
    assert "github-release" in after_publication["needs"]
    assert "needs.github-release.result == 'success'" in after_publication["if"]
    assert sum("python tools/build_release_notes.py" in step.get("run", "")
               for step in after_publication["steps"]) == 1


def test_preparation_news_producer_keeps_published_history_without_announcing_drafts(
        tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "release_preparation_news", ROOT / "tools/build_release_notes.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    target = tmp_path / "release_notes.json"
    monkeypatch.setattr(tool, "TARGET", target)
    published = {"tag": "v1.5.1.5", "name": "spaCR 1.5.1.5",
                 "published": "2026-10-01T00:00:00Z", "url": "https://example.invalid/1.5.1.5",
                 "body": "Published note https://example.invalid/details", "prerelease": False}
    monkeypatch.setattr(tool, "fetch", lambda: [dict(published, tag="v1.5.1.6", draft=True),
                                               published])
    assert tool.main([]) == 0
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["releases"] == [{
        "tag": "v1.5.1.5", "name": "spaCR 1.5.1.5", "published": "2026-10-01",
        "url": published["url"], "body": published["body"],
        "links": ["https://example.invalid/details"], "prerelease": False}]
    before = target.read_bytes()
    monkeypatch.setattr(tool, "fetch", lambda: [dict(published, body="Updated published note")])
    assert tool.main(["--check"]) == 1
    assert target.read_bytes() == before
    def unavailable():
        raise RuntimeError("release endpoint unavailable")
    monkeypatch.setattr(tool, "fetch", unavailable)
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        tool.main([])
    assert target.read_bytes() == before


@pytest.mark.parametrize("event,expected", [
    ("workflow_dispatch", ["tools/build_release_notes.py"]),
    ("push", ["tools/build_release_notes.py", "--check"]),
])
@pytest.mark.parametrize("result", [0, 3])
@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                    reason="the release workflow step runs under bash on its Linux runner")
def test_news_preparation_refreshes_only_when_the_result_will_be_committed(
        tmp_path, event, expected, result):
    import os
    import subprocess
    import yaml

    steps = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]["bump"]["steps"]
    step = next(step for step in steps
                if step.get("name") == "Refresh notes for releases already published")
    executable = tmp_path / "python"
    executable.write_text("""#!/bin/sh
printf '%s\\n' "$@" > "$NEWS_CALL"
exit "$NEWS_RESULT"
""")
    executable.chmod(0o755)
    call = tmp_path / "call"
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                       NEWS_CALL=str(call), NEWS_RESULT=str(result))
    completed = subprocess.run(["bash", "-e", "-c", step["run"].replace(
        "${{ github.event_name }}", event)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=10)
    assert completed.returncode == result
    assert call.read_text().splitlines() == expected
