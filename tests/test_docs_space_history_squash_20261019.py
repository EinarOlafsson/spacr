"""The nightly docs Space squashes its history before it fills.

Each nightly upload keeps the previous site in the Space's history; the first
Space reached its 1 GB limit in about five days and refused every deploy.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import publish_docs_channels as publisher  # noqa: E402

GB = 1000 * 1000 * 1000


@pytest.mark.parametrize("used, incoming, commits, expected", [
    (0, 0, 0, None),
    (200_000_000, 300_000_000, 5, None),
    (400_000_000, 250_000_000, 5, "storage"),
    (990_000_000, 0, 1, "storage"),
    (100_000_000, 100_000_000, 19, None),
    (100_000_000, 100_000_000, 20, "commits"),
    (700_000_000, 0, 40, "storage"),
])
def test_the_decision(used, incoming, commits, expected):
    assert publisher.squash_reason(used, incoming, commits) == expected


def test_the_threshold_and_cadence_can_be_set():
    assert publisher.squash_reason(10, 10, 0, limit=100, fraction=0.1) == "storage"
    assert publisher.squash_reason(10, 10, 3, limit=100, every=3) == "commits"


class _Api:
    def __init__(self, used, commits, served, stored):
        self.used, self.commits = used, commits
        self.served, self.stored = served, stored
        self.calls = []

    def space_info(self, space, **kwargs):
        self.calls.append(("info", kwargs))
        return SimpleNamespace(sha="head", usedStorage=self.used)

    def list_repo_commits(self, space, **kwargs):
        return [object()] * self.commits

    def super_squash_history(self, space, **kwargs):
        self.calls.append(("squash", kwargs["repo_type"]))

    def list_repo_tree(self, space, **kwargs):
        assert kwargs["revision"] == "head" and kwargs["recursive"]
        return [SimpleNamespace(lfs=SimpleNamespace(sha256=s)) for s in self.served] + [
            SimpleNamespace(path="index.html")]

    def list_lfs_files(self, space, **kwargs):
        return [SimpleNamespace(file_oid=oid, size=size) for oid, size in self.stored]

    def permanently_delete_lfs_files(self, space, files, **kwargs):
        self.calls.append(("delete", sorted(f.file_oid for f in files)))


@pytest.fixture()
def site(tmp_path):
    (tmp_path / "index.html").write_bytes(b"x" * 1000)
    return tmp_path


def test_a_small_space_is_left_alone(site):
    api = _Api(100_000_000, 3, ["a"], [("a", 5), ("b", 7)])
    result = publisher.squash_nightly_if_needed(site, "o/s", api=api)
    assert result["reason"] is None and result["squashed"] is False
    assert [c[0] for c in api.calls] == ["info"]


def test_a_full_space_is_squashed_and_loses_only_unserved_files(site):
    api = _Api(990_000_000, 40, ["a", "c"], [("a", 5), ("b", 7), ("c", 1), ("d", 9)])
    result = publisher.squash_nightly_if_needed(site, "o/s", api=api)
    assert result["reason"] == "storage" and result["squashed"] is True
    assert result["deleted_files"] == 2 and result["deleted_bytes"] == 16
    assert api.calls[1] == ("squash", "space")
    assert api.calls[-1] == ("delete", ["b", "d"])
    assert "token" not in repr(result).lower()


def test_the_workflow_squashes_before_it_uploads():
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/docs.yml").read_text()
    squash = text.index("publish_docs_channels.py squash-nightly-if-needed")
    upload = text.index("publish_docs_channels.py upload-nightly")
    assert squash < upload
