"""Deleting or renaming a test takes its entries out of the lists that name it.

``tools/prune_test_references.py`` is the edit every test-suite reduction
batch (2026-11-12) makes to the hygiene ratchets and the CI classification
tables. These tests run it on a throwaway repository in a temporary
directory, never on spaCR's own lists.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "prune_test_references", ROOT / "tools" / "prune_test_references.py")
assert _SPEC is not None and _SPEC.loader is not None
prune = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prune
_SPEC.loader.exec_module(prune)

LISTS = textwrap.dedent('''
    """Lists that name tests."""

    #: Files allowed to break a rule.
    ALLOWED = {
        # Kept: this file still exists.
        "test_alive.py",
        # Gone: this comment leaves with its entry.
        "qt/test_gone.py",
    }

    #: Functions per file.
    NODES = {
        "test_alive.py": {
            "test_kept",
            "test_removed",
        },
        "test_partly.py": {
            "test_all_gone",
        },
    }

    PATHS = ["tests/test_alive.py::test_kept", "tests/test_alive.py::test_removed"]


    def helper():
        """Not a module-level list: never edited."""
        example = {"test_gone.py"}
        return example
''')


@pytest.fixture
def repo(tmp_path):
    """A repository with two live test files and one list file."""
    tests = tmp_path / "tests"
    (tests / "qt").mkdir(parents=True)
    (tests / "test_alive.py").write_text(
        "def test_kept():\n    assert True\n\n\n"
        "class TestGroup:\n    def test_method(self):\n        assert True\n",
        encoding="utf-8")
    (tests / "test_partly.py").write_text(
        "def test_other():\n    assert True\n", encoding="utf-8")
    (tests / "lists.py").write_text(LISTS, encoding="utf-8")
    return tmp_path


def _run(repo, *args):
    return prune.main(["--root", str(repo), "--target", "tests/lists.py", *args])


def test_stale_entries_leave_with_their_comments_and_live_ones_stay(repo, capsys):
    assert _run(repo, "--stale", "--apply") == 0

    text = (repo / "tests" / "lists.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    values = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    assert "test_alive.py" in values
    assert "# Kept: this file still exists." in text
    assert "qt/test_gone.py" not in values
    assert "Gone: this comment leaves" not in text
    assert "test_kept" in values
    assert "test_removed" not in values
    assert "tests/test_alive.py::test_kept" in values
    assert "tests/test_alive.py::test_removed" not in values
    assert "test_partly.py" not in values
    assert "test_all_gone" not in values
    assert 'example = {"test_gone.py"}' in text
    printed = capsys.readouterr().out
    assert "REMOVE tests/lists.py" in printed
    assert "wrote tests/lists.py" in printed


def test_without_apply_nothing_is_written(repo, capsys):
    before = (repo / "tests" / "lists.py").read_text(encoding="utf-8")

    assert _run(repo, "--stale") == 0

    assert (repo / "tests" / "lists.py").read_text(encoding="utf-8") == before
    assert "REMOVE tests/lists.py" in capsys.readouterr().out


def test_a_named_deletion_removes_only_that_test(repo):
    assert _run(repo, "--deleted", "tests/test_alive.py::test_kept", "--apply") == 0

    tree = ast.parse((repo / "tests" / "lists.py").read_text(encoding="utf-8"))
    values = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    assert "test_kept" not in values
    assert "tests/test_alive.py::test_kept" not in values
    assert "test_removed" in values
    assert "qt/test_gone.py" in values


def test_a_rename_keeps_each_entry_spelled_the_way_it_was(repo):
    (repo / "tests" / "test_alive.py").rename(repo / "tests" / "test_renamed.py")

    assert _run(repo, "--renamed", "tests/test_alive.py=tests/test_renamed.py",
                "--renamed", "tests/test_alive.py::test_kept=tests/test_renamed.py::test_new",
                "--apply") == 0

    text = (repo / "tests" / "lists.py").read_text(encoding="utf-8")
    assert '"test_renamed.py"' in text
    assert '"test_alive.py"' not in text
    assert '"test_new"' in text
    assert '"tests/test_renamed.py::test_new"' in text
    assert '"tests/test_renamed.py::test_removed"' in text
    ast.parse(text)


def test_a_method_inside_a_class_is_not_stale(repo):
    lists = repo / "tests" / "lists.py"
    lists.write_text('NODES = {"test_alive.py": {"test_method"}}\n', encoding="utf-8")

    assert _run(repo, "--stale") == 0

    assert prune.plan_edits(repo, [lists], stale=True) == []


def test_nothing_to_change_is_said(repo, capsys):
    lists = repo / "tests" / "lists.py"
    lists.write_text('ALLOWED = {"test_alive.py"}\n', encoding="utf-8")

    assert _run(repo, "--stale", "--apply") == 0

    assert "nothing to change" in capsys.readouterr().out
    assert lists.read_text(encoding="utf-8") == 'ALLOWED = {"test_alive.py"}\n'


def test_a_malformed_rename_is_refused(repo, capsys):
    assert _run(repo, "--renamed", "no-equals-sign") == 2
    assert "--renamed needs OLD=NEW" in capsys.readouterr().err


def test_a_workflow_that_names_a_deleted_test_is_reported_not_edited(repo, capsys):
    workflows = repo / ".github" / "workflows"
    workflows.mkdir(parents=True)
    workflow = workflows / "tests.yml"
    body = ("run: |\n  pytest --deselect=tests/test_alive.py::test_kept \\\n"
            "    --exclude-file tests/test_partly.py\n")
    workflow.write_text(body, encoding="utf-8")

    assert _run(repo, "--deleted", "tests/test_alive.py::test_kept",
                "--deleted", "tests/test_partly.py", "--apply") == 0

    printed = capsys.readouterr().out
    assert printed.count("WORKFLOW (edit by hand)") == 2
    assert workflow.read_text(encoding="utf-8") == body


def test_the_default_targets_exist_in_this_repository():
    for target in prune.DEFAULT_TARGETS:
        assert (ROOT / target).is_file(), target
