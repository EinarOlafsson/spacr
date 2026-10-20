"""The Apptainer images are built only after the GitHub release has created their tag.

apptainer-images.yml checks out ``v<version>``, which the github-release job creates. Ordered only after
container-images, it could start before the tag existed and fail its checkout.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _jobs():
    return yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))["jobs"]


def test_apptainer_images_need_the_github_release():
    job = _jobs()["apptainer-images"]
    needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    assert {"container-images", "github-release"} <= set(needs)
    assert "needs.github-release.result == 'success'" in job["if"]
    assert "needs.container-images.result == 'success'" in job["if"]


def test_the_apptainer_workflow_checks_out_the_release_tag():
    text = (ROOT / ".github" / "workflows" / "apptainer-images.yml").read_text(encoding="utf-8")
    assert "ref: v${{ inputs.release_version }}" in text


def test_the_github_release_does_not_wait_for_apptainer():
    needs = _jobs()["github-release"]["needs"]
    assert "apptainer-images" not in needs
