"""A conftest's fixtures reach every test in its directory, in any order.

pytest scopes a conftest's fixtures to the DIRECTORY the conftest sits in,
and binds that scope to the directory's collection NODE OBJECT. Collect the
same directory twice -- which pytest does when a bare file path follows it on
the command line -- and the second node is one the conftest was never parsed
against, so every fixture defined there, autouse ones included, vanishes for
the tests underneath it.

It vanishes QUIETLY: the tests that ran before the second collection still
pass, so the summary line reads as a partial success while a whole file
errors with "fixture 'qt_theme_applied' not found". The person it bites is
the one running "the files I touched", which is what WORKFLOW.md asks for.

So two things are pinned here. A directory answers with one collection node
for the whole session, and if that ever stops being true the run stops with a
message about the ORDERING instead of leaving a missing-fixture error for
somebody to misread.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    _check_directory_conftest_fixtures,
    _conftest_fixtures_went_missing,
    canonical_directory_children,
    directory_fixture_expectations,
    lost_directory_conftest_fixtures,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Set in the child pytest this file spawns. The child is collected as the
#: NON-Qt file in the middle of two Qt ones -- that interleaving is the whole
#: point -- so it runs this module's fast tests and must not spawn a child of
#: its own.
_CHILD_ENV = "SPACR_CONFTEST_ORDERING_CHILD"

_FIRST = ("tests/qt/test_ambient_none.py"
          "::test_none_is_offered_alongside_the_six_animations")
_MIDDLE = "tests/test_conftest_fixture_visibility.py"
_LAST = ("tests/qt/test_ambient_none.py"
         "::test_none_has_a_label_and_a_note_that_states_the_cost")

_DISABLE_PLUGIN = '''\
"""Unregister the canonicaliser so the ordering hazard is live again."""


def pytest_configure(config):
    plugin = config.pluginmanager.get_plugin("spacr-one-node-per-directory")
    if plugin is not None:
        config.pluginmanager.unregister(plugin)
'''


def _skip_in_the_child():
    if os.environ.get(_CHILD_ENV):
        pytest.skip("this is the child run; it must not spawn another")
    pytest.importorskip("PySide6")
    pytest.importorskip("pytestqt")


def _interleaved_run(tmp_path, disable_canonicaliser=False):
    """Run a Qt file, a non-Qt file and a Qt file in one pytest invocation."""
    env = dict(os.environ)
    env[_CHILD_ENV] = "1"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["CUDA_VISIBLE_DEVICES"] = ""
    argv = [sys.executable, "-m", "pytest", _FIRST, _MIDDLE, _LAST,
            "-q", "-p", "no:randomly", "-p", "no:cacheprovider"]
    if disable_canonicaliser:
        plugin_dir = tmp_path / "mutation"
        plugin_dir.mkdir()
        (plugin_dir / "spacr_undo_directory_nodes.py").write_text(
            _DISABLE_PLUGIN)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(plugin_dir), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        argv += ["-p", "spacr_undo_directory_nodes"]
    return subprocess.run(argv, cwd=str(_REPO_ROOT), env=env,
                          capture_output=True, text=True, timeout=900)


def test_a_qt_file_after_a_non_qt_file_still_sees_the_qt_conftest(tmp_path):
    """The reproduction the hazard was found by, run for real.

    A subprocess rather than an in-process check because the fault is in how
    pytest builds its collection tree from the command line, and only a fresh
    invocation with those arguments builds it that way.
    """
    _skip_in_the_child()
    done = _interleaved_run(tmp_path)
    output = done.stdout + done.stderr
    assert "fixture 'qt_theme_applied' not found" not in output, output[-3000:]
    assert "COLLECTION ORDERING fault" not in output, output[-3000:]
    assert done.returncode == 0, output[-3000:]


def test_without_the_canonical_directory_node_the_run_stops_and_says_why(
        tmp_path):
    """The same arguments, with the fix switched off: loud, and fast.

    This is the mutation proof for the test above -- it shows the hazard is
    still real and that the guard, not luck, is what keeps the run honest.
    A missing-fixture error 29 times over taught nobody what went wrong, so
    what is asserted here is that the message blames the ORDERING.
    """
    _skip_in_the_child()
    done = _interleaved_run(tmp_path, disable_canonicaliser=True)
    output = done.stdout + done.stderr
    assert done.returncode != 0, output[-3000:]
    assert "COLLECTION ORDERING fault" in output, output[-3000:]
    assert "the conftest in tests/qt" in output, output[-3000:]


def test_a_directory_answers_with_one_collection_node(request):
    """No two nodes in this run's tree claim the same directory.

    Held over the session actually collected, so any invocation shape that
    reintroduces a duplicate directory node fails here rather than in
    whichever unlucky test file loses its fixtures.
    """
    by_nodeid = {}
    for item in request.session.items:
        for node in item.listchain():
            if isinstance(node, pytest.Directory):
                first = by_nodeid.setdefault(node.nodeid, node)
                assert first is node, (
                    f"{node.nodeid} has two collection nodes; every conftest "
                    "fixture defined there is invisible to the tests reached "
                    "through the second one")


def test_the_expected_fixtures_are_read_off_the_running_session(request):
    """The check knows what a directory's conftest defines without a list.

    Read back off the fixtures pytest registered, so a fixture added to
    tests/conftest.py tomorrow is covered without anybody remembering to name
    it here.
    """
    manager = request.session._fixturemanager
    expectations = directory_fixture_expectations(manager)
    assert "tests" in expectations, sorted(expectations)
    assert {"rng", "tmp_project_dir", "synth_image_2d",
            "_isolated_qsettings_store"} <= expectations["tests"]


# ---------------------------------------------------------------------------
# The check itself, driven directly
# ---------------------------------------------------------------------------

class _Node:
    """One node of a collection chain: all the check reads is its nodeid."""

    def __init__(self, nodeid):
        self.nodeid = nodeid


class _Item(_Node):
    """A collected test, with the chain of nodes it hangs off."""

    def __init__(self, nodeid, chain):
        super().__init__(nodeid)
        self._chain = list(chain)

    def listchain(self):
        return [*self._chain, self]


class _FixtureManager:
    """Answers fixture lookups from a fixed per-item visibility table."""

    def __init__(self, visible):
        self._visible = visible
        self._arg2fixturedefs = {}

    def getfixturedefs(self, argname, node):
        return self._visible.get(id(node), {}).get(argname, ())


def _chain(directory):
    return [_Node(""), _Node("tests"), directory, _Node("tests/qt/test_x.py")]


def test_a_conftest_fixture_its_own_tests_cannot_request_is_reported():
    """The failure this exists to catch, named by directory and by fixture."""
    directory = _Node("tests/qt")
    item = _Item("tests/qt/test_x.py::test_y", _chain(directory))
    manager = _FixtureManager({id(item): {"rng": ("a fixturedef",)}})

    lost = lost_directory_conftest_fixtures(
        manager, [item],
        {"tests": {"rng"}, "tests/qt": {"qt_theme_applied"}})

    assert lost == [("tests/qt", "qt_theme_applied", item.nodeid)]


def test_a_conftest_whose_fixtures_all_resolve_reports_nothing():
    """Empty is the healthy answer, so the guard cannot cry wolf."""
    directory = _Node("tests/qt")
    item = _Item("tests/qt/test_x.py::test_y", _chain(directory))
    manager = _FixtureManager({id(item): {"rng": ("a fixturedef",),
                                          "qt_theme_applied": ("another",)}})

    lost = lost_directory_conftest_fixtures(
        manager, [item],
        {"tests": {"rng"}, "tests/qt": {"qt_theme_applied"}})

    assert lost == []


def test_two_nodes_for_one_directory_are_both_examined():
    """Checking one test per chain must not mean checking one test per run.

    The whole fault is that SOME tests reach a directory through a second
    node, so a check that stopped after the first healthy chain would report
    a clean run every time.
    """
    healthy_dir = _Node("tests/qt")
    stale_dir = _Node("tests/qt")
    healthy = _Item("tests/qt/test_x.py::test_y", _chain(healthy_dir))
    orphan = _Item("tests/qt/test_z.py::test_w", _chain(stale_dir))
    manager = _FixtureManager({id(healthy): {"qt_theme_applied": ("a def",)},
                               id(orphan): {}})

    lost = lost_directory_conftest_fixtures(
        manager, [healthy, orphan], {"tests/qt": {"qt_theme_applied"}})

    assert lost == [("tests/qt", "qt_theme_applied", orphan.nodeid)]


def test_tests_sharing_a_chain_are_checked_once():
    """One report per collision, not one per test that inherited it.

    29 identical errors is what the fault looked like before; a hundred
    identical lines in the message would be the same mistake again.
    """
    directory = _Node("tests/qt")
    chain = _chain(directory)
    first = _Item("tests/qt/test_x.py::test_a", chain)
    second = _Item("tests/qt/test_x.py::test_b", chain)
    manager = _FixtureManager({})

    lost = lost_directory_conftest_fixtures(
        manager, [first, second], {"tests/qt": {"qt_theme_applied"}})

    assert lost == [("tests/qt", "qt_theme_applied", first.nodeid)]


def test_the_report_blames_the_ordering_rather_than_the_fixture():
    """What the reader has to learn is that the ARGUMENTS were the problem."""
    message = _conftest_fixtures_went_missing(
        [("tests/qt", "qt_theme_applied", "tests/qt/test_x.py::test_y")])

    assert "qt_theme_applied" in message
    assert "COLLECTION ORDERING fault" in message
    assert "collected twice" in message
    assert "pytest tests/qt/test_a.py tests/test_b.py tests/qt/test_c.py" \
        in message


def test_a_long_report_is_cut_off_with_a_count():
    """A hundred lost fixtures must still print a message somebody reads."""
    lost = [("tests/qt", f"fixture_{index}", "tests/qt/test_x.py::test_y")
            for index in range(24)]

    message = _conftest_fixtures_went_missing(lost)

    assert "24 conftest fixture(s)" in message
    assert "... and 14 more" in message
    assert "'fixture_23'" not in message


# ---------------------------------------------------------------------------
# The canonicaliser, and the capability it must not take away
# ---------------------------------------------------------------------------

class _DirectoryStub(pytest.Directory):
    """A directory collector with no session and no config behind it.

    Real ones are built by pytest from a live session; the canonicaliser
    reads nothing but ``path`` and the class, so a stand-in is enough and
    keeps the check testable without a collection tree.
    """

    def collect(self):
        return []

    @property
    def nodeid(self):
        return self._stub_nodeid


def _directory_stub(path, nodeid=""):
    """Build one, past the node metaclass that refuses direct construction."""
    stub = _DirectoryStub.__new__(_DirectoryStub)
    stub.path = Path(path)
    stub._stub_nodeid = nodeid
    return stub


def _fixturedef_of(node):
    """A fixture definition that claims it was found in ``node``."""
    return type("_FixtureDef", (), {"node": node})()


def test_a_directory_met_twice_is_replaced_by_the_node_already_in_use():
    """The fix in one line: the second node for a directory is discarded."""
    registry = {}
    first = _directory_stub("/repo/tests/qt")
    kept = canonical_directory_children(registry, [first])
    assert kept == [first]

    second = _directory_stub("/repo/tests/qt")
    again = canonical_directory_children(registry, [second])

    assert again[0] is first
    assert second not in again


def test_a_file_collected_twice_is_left_alone():
    """Naming a file twice must still collect it twice.

    pytest deliberately does not de-duplicate files given directly on the
    command line, and folding directories together may not quietly take that
    away -- a run of ``pytest test_x.py test_x.py --keepduplicates`` is a
    supported way to ask for a test twice.
    """
    registry = {}
    module = object()

    kept = canonical_directory_children(registry, [module, module])

    assert kept == [module, module]
    assert registry == {}


def test_a_pytest_whose_registry_moved_does_not_stop_the_run():
    """The check reads a pytest internal, and internals move.

    An upgrade that renames the registry must not make every run fail to
    collect. It answers with nothing instead -- and
    ``test_the_expected_fixtures_are_read_off_the_running_session`` above is
    what turns that quiet answer into a red test.
    """
    assert directory_fixture_expectations(object()) == {}


def test_a_run_with_no_directory_conftest_walks_nothing():
    """Nothing to check means no per-item work, on every run that has none.

    The item here refuses to be walked, so the empty answer can only come
    from not walking it -- otherwise this reads as a pass on any
    implementation.
    """
    class _RefusesToBeWalked:
        nodeid = "test_x.py::test_y"

        def listchain(self):
            raise AssertionError(
                "no conftest defines fixtures, so no item should be walked")

    assert lost_directory_conftest_fixtures(
        _FixtureManager({}), [_RefusesToBeWalked()], {}) == []


def test_a_lost_conftest_stops_collection_rather_than_the_test():
    """The run ends at collection, before 29 setups error one at a time."""
    directory = _Node("tests/qt")
    item = _Item("tests/qt/test_x.py::test_y", _chain(directory))
    manager = _FixtureManager({id(item): {}})
    manager._arg2fixturedefs = {
        "qt_theme_applied": [_fixturedef_of(_directory_stub("/repo/tests/qt",
                                                            "tests/qt"))]}

    class _Session:
        _fixturemanager = manager

    with pytest.raises(pytest.UsageError) as raised:
        _check_directory_conftest_fixtures(_Session(), [item])

    assert "COLLECTION ORDERING fault" in str(raised.value)


def test_a_session_with_no_fixture_manager_is_left_alone():
    """Nothing to read means nothing to claim; the run collects as normal."""
    assert _check_directory_conftest_fixtures(object(), []) is None
