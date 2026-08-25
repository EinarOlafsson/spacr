"""The lost-conftest guard names the cause it can show, not the cause it knew.

When a conftest's fixtures stop reaching the tests underneath it, pytest says
``fixture 'qt_theme_applied' not found`` once per test and never says why.
tests/conftest.py replaces that with a message about the cause -- and there is
more than one cause it could be:

* the directory was collected TWICE and the tests were reached through the
  second node, which is the collection-ordering fault this repository was
  bitten by and which ``_OneNodePerDirectory`` prevents;
* the directory was collected ONCE and the conftest was evicted anyway -- taken
  out of ``sys.modules``, reloaded, or unregistered by something in the run.

The second was the first hypothesis when the fault was found, and it was wrong:
the evidence pinned at the bottom of this file shows one conftest module,
registered, unchanged, while ``tests/qt`` answers with two collection nodes. So
the message may not print "collected twice" as an article of faith. It looks at
the collected tree, and a directory that was collected once is reported as an
eviction with the ordering fault ruled out for the reader instead of at them.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    _conftest_fixtures_went_missing,
    directories_collected_twice,
    directory_conftest_parse_nodes,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The shape that loses the fixtures: a Qt file, a bare non-Qt file that makes
#: pytest re-collect ``tests``, then a second Qt file reached through the node
#: that re-collection built.
_INTERLEAVED = ("tests/qt/test_ambient_none.py",
                "tests/test_crop_format.py",
                "tests/qt/test_demo_menu.py")

_DISABLE_PLUGIN = '''\
"""Unregister the canonicaliser so the ordering hazard is live again."""


def pytest_configure(config):
    plugin = config.pluginmanager.get_plugin("spacr-one-node-per-directory")
    if plugin is not None:
        config.pluginmanager.unregister(plugin)
'''

_MODULE_PROBE = '''\
"""Print what the run holds for tests/qt/conftest.py before the guard fires."""
import sys

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    path = str(config.rootpath / "tests" / "qt" / "conftest.py")
    modules = [m for m in list(sys.modules.values())
               if getattr(m, "__file__", None) == path]
    registered = [p for p in config.pluginmanager.get_plugins()
                  if getattr(p, "__file__", None) == path]
    same = bool(modules) and bool(registered) and modules[0] is registered[0]
    nodes = set()
    for item in items:
        for node in item.listchain():
            if not isinstance(node, pytest.Directory):
                continue
            if node.nodeid == "tests/qt":
                nodes.add(id(node))
    print("\\nQTCONFTEST modules=%d registered=%d same=%s qt_nodes=%d"
          % (len(modules), len(registered), same, len(nodes)))
'''


def _collect_interleaved(tmp_path, extra_plugins=(),
                         disable_canonicaliser=False):
    """Collect the interleaving in a child pytest and return what it printed.

    Collection only: the fault is built while pytest turns the command line
    into a tree, so ``--co`` reaches it without paying for a Qt suite.
    """
    pytest.importorskip("PySide6")
    pytest.importorskip("pytestqt")
    for named in _INTERLEAVED:
        assert (_REPO_ROOT / named).exists(), (
            f"{named} is named here to build the interleaving and is gone; "
            "put another Qt file, non-Qt file, Qt file in its place rather "
            "than dropping the shape")
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir(exist_ok=True)
    argv = [sys.executable, "-m", "pytest", "--co", "-q",
            "-p", "no:randomly", "-p", "no:cacheprovider"]
    for name, source in extra_plugins:
        (plugin_dir / f"{name}.py").write_text(source)
        argv += ["-p", name]
    if disable_canonicaliser:
        (plugin_dir / "spacr_undo_directory_nodes.py").write_text(
            _DISABLE_PLUGIN)
        argv += ["-p", "spacr_undo_directory_nodes"]
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(
        [str(plugin_dir), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    done = subprocess.run([*argv, *_INTERLEAVED], cwd=str(_REPO_ROOT),
                          env=env, capture_output=True, text=True,
                          timeout=900)
    return done, done.stdout + done.stderr


# ---------------------------------------------------------------------------
# What the collected tree is asked
# ---------------------------------------------------------------------------

class _Node:
    """A collection node the check reads nothing off but its nodeid."""

    def __init__(self, nodeid):
        self.nodeid = nodeid


class _Directory(pytest.Directory):
    """A directory node stood up without a session behind it.

    Only its class and nodeid are read, and pytest's node metaclass refuses
    direct construction, so this is built past ``__init__`` on purpose.
    """

    def collect(self):
        return []

    @property
    def nodeid(self):
        return self._stub_nodeid


def _directory(nodeid):
    stub = _Directory.__new__(_Directory)
    stub._stub_nodeid = nodeid
    return stub


class _Item(_Node):
    def __init__(self, nodeid, chain):
        super().__init__(nodeid)
        self._chain = list(chain)

    def listchain(self):
        return [*self._chain, self]


def test_two_nodes_for_one_directory_are_reported_as_collected_twice():
    """The direct evidence: one nodeid, two node objects in the tree."""
    tests = _directory("tests")
    first, second = _directory("tests/qt"), _directory("tests/qt")

    twice = directories_collected_twice([
        _Item("tests/qt/test_a.py::test_x", [tests, first]),
        _Item("tests/qt/test_c.py::test_y", [tests, second]),
    ])

    assert twice == {"tests/qt"}


def test_one_node_per_directory_is_reported_as_collected_once():
    """The healthy tree must answer "no duplicate", not "did not look"."""
    tests = _directory("tests")
    directory = _directory("tests/qt")

    twice = directories_collected_twice([
        _Item("tests/qt/test_a.py::test_x", [tests, directory]),
        _Item("tests/qt/test_c.py::test_y", [tests, directory]),
    ])

    assert twice == set()


def test_a_node_the_conftest_was_not_parsed_against_counts_as_a_duplicate():
    """The surviving half of a duplicate is still a duplicate.

    ``-k`` and the file shard both drop tests, and they drop them after the
    tree is built. When everything hanging off the first node goes, the
    second node is all that is left to see -- and it is the node that lost
    the fixtures, so the run must still be told the directory was collected
    twice rather than that its conftest was evicted.
    """
    parsed_against = _directory("tests/qt")
    survivor = _directory("tests/qt")

    twice = directories_collected_twice(
        [_Item("tests/qt/test_c.py::test_y", [survivor])],
        {"tests/qt": parsed_against})

    assert twice == {"tests/qt"}


def test_a_tree_with_no_directory_nodes_answers_that_it_did_not_look():
    """"Nothing to look at" and "nothing wrong" must not be the same answer.

    A caller that read the empty set here would name a cause off evidence it
    never had.
    """
    assert directories_collected_twice(
        [_Item("test_a.py::test_x", [_Node("tests")])]) is None


def test_the_parse_nodes_are_read_off_the_running_session(request):
    """This session's own conftests are bound to the nodes in this chain.

    Both halves matter: the fixtures of ``tests`` are parsed against a real
    directory node, and it is the very node this test hangs off -- which is
    the healthy state the guard exists to keep.
    """
    parse_nodes = directory_conftest_parse_nodes(
        request.session._fixturemanager)

    assert "tests" in parse_nodes, sorted(parse_nodes)
    chain = {id(node) for node in request.node.listchain()}
    assert id(parse_nodes["tests"]) in chain


def test_a_pytest_whose_registry_moved_yields_no_parse_nodes():
    """An internal that moves must leave the run collectable.

    The test above is what turns this quiet answer into a red test.
    """
    assert directory_conftest_parse_nodes(object()) == {}


# ---------------------------------------------------------------------------
# Which cause the message names
# ---------------------------------------------------------------------------

_LOST_QT = ("tests/qt", "qt_theme_applied", "tests/qt/test_x.py::test_y")
_LOST_TESTS = ("tests", "rng", "tests/test_z.py::test_w")


def test_a_directory_collected_twice_is_reported_as_the_ordering_fault():
    """With the duplicate shown, the message says so and names it."""
    message = _conftest_fixtures_went_missing([_LOST_QT], {"tests/qt"})

    assert "Collected twice: tests/qt." in message
    assert "COLLECTION ORDERING fault" in message
    assert "EVICTED" not in message


def test_a_directory_collected_once_is_reported_as_an_eviction():
    """The other cause, and it may not be dressed up as the known one.

    A reader sent after the collection order when the directory was
    collected once loses the time this file's investigation already spent.
    """
    message = _conftest_fixtures_went_missing([_LOST_QT], set())

    assert "Collected once: tests/qt." in message
    assert "EVICTED" in message
    assert "sys.modules" in message
    assert "COLLECTION ORDERING fault" not in message


def test_both_causes_in_one_run_are_reported_apart():
    """Two directories, two causes; neither may borrow the other's."""
    message = _conftest_fixtures_went_missing([_LOST_QT, _LOST_TESTS],
                                              {"tests/qt"})

    assert "Collected twice: tests/qt." in message
    assert "Collected once: tests." in message
    assert "COLLECTION ORDERING fault" in message
    assert "EVICTED" in message


def test_a_caller_that_looked_at_nothing_gets_the_ordering_wording():
    """No evidence either way keeps the message this guard shipped with."""
    message = _conftest_fixtures_went_missing([_LOST_QT])

    assert "COLLECTION ORDERING fault" in message
    assert "Collected twice:" not in message
    assert "EVICTED" not in message


def test_every_lost_fixture_is_still_listed_before_the_cause():
    """Naming the cause may not cost the reader the fixtures it happened to."""
    message = _conftest_fixtures_went_missing([_LOST_QT, _LOST_TESTS], set())

    assert "'qt_theme_applied' comes from the conftest in tests/qt" in message
    assert "'rng' comes from the conftest in tests" in message
    assert "2 conftest fixture(s)" in message


# ---------------------------------------------------------------------------
# The interleaving, run for real
# ---------------------------------------------------------------------------

def test_the_real_interleaving_is_reported_as_a_directory_collected_twice(
        tmp_path):
    """The live fault picks the ordering branch, with the duplicate shown.

    Run with the canonicaliser off, so the hazard is the real one and not a
    fabricated table. What is asserted is the branch: a message that called
    this an eviction would have the diagnosis exactly backwards.
    """
    done, output = _collect_interleaved(tmp_path, disable_canonicaliser=True)

    assert done.returncode != 0, output[-3000:]
    assert "Collected twice: tests/qt." in output, output[-3000:]
    assert "COLLECTION ORDERING fault" in output, output[-3000:]
    assert "EVICTED" not in output, output[-3000:]


def test_the_lost_fixtures_are_not_a_conftest_that_went_missing(tmp_path):
    """The evidence for the branch above: one module, two directory nodes.

    The first hypothesis about this fault was that something in the run --
    the file that walks the package with ast, or the one that clears module
    caches -- evicted tests/qt/conftest.py from ``sys.modules``. It did not.
    In the run that loses the fixtures the conftest is one module object,
    still registered as a plugin, and ``tests/qt`` answers with two
    collection nodes; and with the canonicaliser back on it answers with
    one. If that ever stops being true the guard is naming the wrong cause,
    and this is where it shows.
    """
    probe = [("spacr_qt_conftest_probe", _MODULE_PROBE)]

    _, broken = _collect_interleaved(tmp_path, probe,
                                     disable_canonicaliser=True)
    _, healthy = _collect_interleaved(tmp_path, probe)

    one_module = "QTCONFTEST modules=1 registered=1 same=True"
    assert f"{one_module} qt_nodes=2" in broken, broken[-3000:]
    assert f"{one_module} qt_nodes=1" in healthy, healthy[-3000:]
