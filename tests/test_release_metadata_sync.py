"""`packaging/release.py sync-release-metadata`, with Zenodo and PyPI faked.

Asked for on 2026-09-26: "CITATION.cff and conda recipe still at 1.5.0.9 ...
should be automatically updated upon release". Nothing moved either file's
release-specific values after `bump`: CITATION.cff kept the previous
release's version DOI and the reference conda recipe kept the previous
sdist's version and sha256. The subcommand looks both up once the release is
published and `release.yml`'s `release-metadata` job commits the result.

No test here touches the network: every lookup goes through a fake `fetch`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"

OLD_DOI = "10.5281/zenodo.22890536"
NEW_DOI = "10.5281/zenodo.22940761"
CONCEPT_DOI = "10.5281/zenodo.21343316"
OLD_SHA = "af79aee13a3fd7a0d2e55b0a0e1068a6f77c12b16cb9605863c9ee0dbd5ef152"
NEW_SHA = "013c6617d59e26ed54f9132bb906c2677e1c2fbfe878ab8d8f651d00c6dd69ff"

CITATION = f"""cff-version: 1.2.0
title: "spaCR"
version: "1.5.1.0"
date-released: "2026-09-23"
doi: "{OLD_DOI}"
identifiers:
  - type: doi
    value: "{OLD_DOI}"
    description: "Version DOI: this release, spaCR 1.5.0.9. Cite this to point at the exact code a result came from."
  - type: doi
    value: "{CONCEPT_DOI}"
    description: "Concept DOI: all versions of spaCR. Resolves to the most recent release."
license: BSD-3-Clause
"""

RECIPE = f"""context:
  version: "1.5.0.9"

package:
  name: spacr
  version: ${{{{ version }}}}

source:
  url: https://pypi.org/packages/source/s/spacr/spacr-${{{{ version }}}}.tar.gz
  sha256: {OLD_SHA}
"""


def _release_module():
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper_metadata", ROOT / "packaging" / "release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zenodo(*versions):
    """A Zenodo search answer holding ``(tag, doi, date)`` versions."""
    return {"hits": {"hits": [
        {"id": int(doi.rsplit(".", 1)[1]), "doi": doi,
         "conceptrecid": "21343316", "conceptdoi": CONCEPT_DOI,
         "metadata": {"version": tag, "publication_date": published}}
        for tag, doi, published in versions
    ]}}


def _pypi(sha=NEW_SHA):
    return {"urls": [
        {"packagetype": "bdist_wheel", "filename": "spacr-1.5.1.0-py3-none-any.whl",
         "digests": {"sha256": "0" * 64}},
        {"packagetype": "sdist", "filename": "spacr-1.5.1.0.tar.gz",
         "digests": {"sha256": sha}},
    ]}


class _FakeServices:
    """Answers like Zenodo and PyPI, and records every URL it was asked."""

    def __init__(self, zenodo_answers, pypi_answers):
        self.zenodo_answers = list(zenodo_answers)
        self.pypi_answers = list(pypi_answers)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        queue = (self.zenodo_answers if url.startswith("https://zenodo.org/")
                 else self.pypi_answers)
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _tree(tmp_path, *, recipe=True):
    (tmp_path / "CITATION.cff").write_text(CITATION, encoding="utf-8")
    if recipe:
        path = tmp_path / "conda-forge" / "recipe" / "recipe.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(RECIPE, encoding="utf-8")
    (tmp_path / "setup.py").write_text('VERSION = "1.5.1.0"\n', encoding="utf-8")
    return tmp_path


def test_the_lookup_waits_for_zenodo_to_mint_the_version_doi():
    helper = _release_module()
    earlier = ("v1.5.0.9", OLD_DOI, "2026-09-22")
    fake = _FakeServices(
        [_zenodo(earlier),
         urllib.error.URLError("connection reset"),
         _zenodo(("v1.5.1.0", NEW_DOI, "2026-09-24"), earlier)],
        [_pypi()])
    slept = []

    metadata = helper.lookup_release_metadata(
        "1.5.1.0", fetch=fake, attempts=5, delay=7, sleep=slept.append)

    assert metadata == {"version": "1.5.1.0", "doi": NEW_DOI,
                        "released": "2026-09-24", "sha256": NEW_SHA}
    assert slept == [7, 7], "two not-yet answers, two waits, then the DOI"
    zenodo_urls = [url for url in fake.urls if "zenodo.org" in url]
    assert all("conceptrecid%3A21343316" in url for url in zenodo_urls)
    assert all("all_versions=true" in url for url in zenodo_urls)
    assert fake.urls[-1] == "https://pypi.org/pypi/spacr/1.5.1.0/json"


def test_the_lookup_gives_up_after_its_attempts_rather_than_guessing():
    helper = _release_module()
    fake = _FakeServices([_zenodo(("v1.5.0.9", OLD_DOI, "2026-09-22"))],
                         [_pypi()])
    slept = []
    with pytest.raises(helper.MetadataNotReady, match="1.5.1.0"):
        helper.lookup_release_metadata(
            "1.5.1.0", fetch=fake, attempts=3, delay=1, sleep=slept.append)
    assert slept == [1, 1]


def test_a_request_zenodo_refuses_is_not_retried():
    helper = _release_module()
    refused = urllib.error.HTTPError(
        "https://zenodo.org/api/records", 400, "BAD REQUEST", {}, None)
    fake = _FakeServices([refused], [_pypi()])
    slept = []
    with pytest.raises(urllib.error.HTTPError):
        helper.lookup_release_metadata(
            "1.5.1.0", fetch=fake, attempts=5, delay=1, sleep=slept.append)
    assert slept == []


def test_the_pypi_lookup_waits_for_an_sdist_and_ignores_the_wheel():
    helper = _release_module()
    wheel_only = {"urls": _pypi()["urls"][:1]}
    fake = _FakeServices(
        [_zenodo(("v1.5.1.0", NEW_DOI, "2026-09-24"))],
        [wheel_only, _pypi()])
    slept = []
    metadata = helper.lookup_release_metadata(
        "1.5.1.0", fetch=fake, attempts=3, delay=2, sleep=slept.append)
    assert metadata["sha256"] == NEW_SHA
    assert slept == [2]


def test_the_concept_doi_is_never_accepted_as_a_version_doi():
    helper = _release_module()
    fake = _FakeServices([_zenodo(("v1.5.1.0", CONCEPT_DOI, "2026-09-24"))],
                         [_pypi()])
    with pytest.raises(ValueError, match="concept DOI"):
        helper.zenodo_version_record("1.5.1.0", fetch=fake)
    with pytest.raises(ValueError, match="version DOI"):
        helper.updated_citation_text(CITATION, {
            "version": "1.5.1.0", "doi": CONCEPT_DOI,
            "released": "2026-09-24", "sha256": NEW_SHA})


def test_sync_rewrites_exactly_the_release_specific_fields(tmp_path):
    helper = _release_module()
    root = _tree(tmp_path)
    metadata = {"version": "1.5.1.0", "doi": NEW_DOI,
                "released": "2026-09-24", "sha256": NEW_SHA}

    changed = helper.sync_release_metadata(root, metadata)

    assert [path.name for path in changed] == ["CITATION.cff", "recipe.yaml"]
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    assert citation == (CITATION
                        .replace('"2026-09-23"', '"2026-09-24"')
                        .replace(OLD_DOI, NEW_DOI)
                        .replace("this release, spaCR 1.5.0.9.",
                                 "this release, spaCR 1.5.1.0."))
    assert f'value: "{CONCEPT_DOI}"' in citation
    recipe = (root / "conda-forge/recipe/recipe.yaml").read_text(encoding="utf-8")
    assert recipe == (RECIPE.replace('"1.5.0.9"', '"1.5.1.0"')
                      .replace(OLD_SHA, NEW_SHA))
    assert "${{ version }}" in recipe

    assert helper.sync_release_metadata(root, metadata) == [], (
        "a rerun for the same release must change nothing")


def test_the_result_satisfies_the_repository_citation_contract(tmp_path):
    """The rewritten file passes the same rule test_packaging_metadata uses."""
    yaml = pytest.importorskip("yaml")
    import re

    helper = _release_module()
    root = _tree(tmp_path)
    helper.sync_release_metadata(root, {
        "version": "1.5.1.0", "doi": NEW_DOI,
        "released": "2026-09-24", "sha256": NEW_SHA})
    citation = yaml.safe_load(
        (root / "CITATION.cff").read_text(encoding="utf-8"))
    version_dois = [
        entry for entry in citation["identifiers"]
        if re.search(r"\bspaCR\s+\d[\d.]*", entry.get("description", ""))]
    assert len(version_dois) == 1
    named = re.search(r"\bspaCR\s+(\d[\d.]*)",
                      version_dois[0]["description"]).group(1).rstrip(".")
    assert named == citation["version"] == "1.5.1.0"
    assert citation["doi"] == version_dois[0]["value"] == NEW_DOI


def test_sync_refuses_to_move_either_file_back_to_an_older_release(tmp_path):
    helper = _release_module()
    root = _tree(tmp_path)
    before = {path: path.read_text(encoding="utf-8")
              for path in (root / "CITATION.cff",
                           root / "conda-forge/recipe/recipe.yaml")}
    with pytest.raises(ValueError, match="refusing to move it back"):
        helper.sync_release_metadata(root, {
            "version": "1.5.0.8", "doi": "10.5281/zenodo.22797833",
            "released": "2026-09-16", "sha256": NEW_SHA})
    after = {path: path.read_text(encoding="utf-8") for path in before}
    assert after == before, "a refused sync must write nothing"


def test_a_malformed_recipe_leaves_the_citation_untouched(tmp_path):
    helper = _release_module()
    root = _tree(tmp_path)
    recipe = root / "conda-forge/recipe/recipe.yaml"
    recipe.write_text(RECIPE.replace(f"  sha256: {OLD_SHA}\n", ""),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        helper.sync_release_metadata(root, {
            "version": "1.5.1.0", "doi": NEW_DOI,
            "released": "2026-09-24", "sha256": NEW_SHA})
    assert (root / "CITATION.cff").read_text(encoding="utf-8") == CITATION


def test_an_absent_recipe_is_skipped(tmp_path):
    helper = _release_module()
    root = _tree(tmp_path, recipe=False)
    changed = helper.sync_release_metadata(root, {
        "version": "1.5.1.0", "doi": NEW_DOI,
        "released": "2026-09-24", "sha256": NEW_SHA})
    assert [path.name for path in changed] == ["CITATION.cff"]


def test_the_command_line_applies_recorded_metadata_offline(tmp_path):
    root = _tree(tmp_path)
    recorded = tmp_path / "release-metadata.json"
    recorded.write_text(json.dumps({
        "version": "1.5.1.0", "doi": NEW_DOI,
        "released": "2026-09-24", "sha256": NEW_SHA}), encoding="utf-8")
    command = [sys.executable, str(ROOT / "packaging" / "release.py"),
               "sync-release-metadata", "--root", str(root),
               "--metadata", str(recorded)]

    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert NEW_DOI in (root / "CITATION.cff").read_text(encoding="utf-8")

    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"version": "1.5.0.9"}), encoding="utf-8")
    refused = subprocess.run(
        [*command[:-1], str(stale), "--version", "1.5.1.0"],
        capture_output=True, text=True, timeout=60)
    assert refused.returncode != 0
    assert "records spaCR 1.5.0.9" in refused.stderr


def test_the_release_workflow_records_the_doi_after_the_github_release():
    """Zenodo mints from the GitHub release, so the job must follow it."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["release-metadata"]

    assert "github-release" in job["needs"]
    assert "needs.github-release.result == 'success'" in job["if"]
    assert job["permissions"] == {"contents": "write"}
    script = "\n".join(step.get("run", "") for step in job["steps"])
    assert "sync-release-metadata" in script
    assert "--lookup-only" in script and "--metadata" in script
    for branch in ('"$RELEASE_BRANCH"', "nightly"):
        assert branch in script
    assert ("chore(release): record spaCR $VERSION's DOI and sdist checksum"
            in script)
    assert 'git push origin "HEAD:$branch"' in script
    assert "git rebase" in script, "a raced push must rebase and retry"
    assert "CITATION.cff conda-forge/recipe/recipe.yaml" in script
    assert "Co-Authored-By" not in script
