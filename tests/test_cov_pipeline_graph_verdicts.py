"""The verdicts a pipeline graph reports, and how it words them.

Instruction 60. Six statements in :mod:`spacr.pipeline_graph` were
uncovered, and every one of them is part of how the graph TELLS A USER
something: whether a node is stale, what to call it, the summary sentence,
and the walk that finds everything downstream of a change.

The reason to assert on the wording rather than only the booleans: this is
the screen a user reads to decide whether to re-run a week of compute. A
node that says "current" when it is stale costs them a wrong conclusion; one
that says "stale" when it is fine costs them the week.
"""
from __future__ import annotations

import pytest

from spacr import pipeline_graph as pg


def _node(**over):
    """A Node with every required field filled, overridable per test."""
    base = dict(
        artifact_id="a1", project="/screens/plate1", kind="measurements-db",
        role="measurements", module="measure", path="/screens/plate1/m.db",
        run_id="run-1", settings_hash="abc", spacr_version="1.5.0",
        created_utc="2026-01-01T00:00:00Z", created_ns=1, size_bytes=10,
        n_files=1, status="complete", exists=True, state=pg.STATE_CURRENT,
        reasons=(), causes=(), depth=0, inputs=(),
    )
    base.update(over)
    return pg.Node(**base)


# --------------------------------------------------------------------------- #
#  Node.stale -- two ways to be stale, and they are not the same way
# --------------------------------------------------------------------------- #

def test_a_node_in_the_stale_state_is_stale():
    assert _node(state=pg.STATE_STALE).stale is True


def test_a_current_node_with_causes_is_still_stale():
    """The second half of the `or`, and the one worth having.

    An artifact can be on disk, complete, and NEWER than nothing -- and still
    be stale because something upstream of it moved. Reporting only `state`
    would call that current.
    """
    node = _node(state=pg.STATE_CURRENT, causes=("upstream-newer",))
    assert node.stale is True


def test_a_current_node_with_no_causes_is_not_stale():
    assert _node().stale is False


# --------------------------------------------------------------------------- #
#  Node.label -- what the graph draws
# --------------------------------------------------------------------------- #

def test_the_label_names_the_module_and_the_file():
    """Both halves: a graph column full of `measurements.db` with no module
    is unreadable, and so is one full of module names with no file."""
    label = _node(path="/screens/plate1/measurements.db").label
    assert "measure" in label
    assert "measurements.db" in label


def test_a_trailing_separator_does_not_produce_an_empty_name():
    """A folder artifact is often registered with a trailing slash, and
    basename() of that is the empty string -- which would draw a blank."""
    assert _node(path="/screens/plate1/merged/").label.endswith("merged")


def test_a_path_that_is_only_a_separator_falls_back_to_itself():
    """Nothing legible to shorten to, so show the path rather than nothing.
    This is the `or self.path` half of the expression."""
    assert _node(path="/").label.endswith("/")


# --------------------------------------------------------------------------- #
#  __str__ -- the one-line form that ends up in logs
# --------------------------------------------------------------------------- #

def test_the_string_form_carries_state_module_kind_and_path():
    """All four, because a log line with three of them cannot be acted on."""
    text = str(_node(state=pg.STATE_STALE))

    assert pg.STATE_STALE in text
    assert "measure" in text
    assert "measurements-db" in text
    assert "/screens/plate1/m.db" in text


# --------------------------------------------------------------------------- #
#  stale_nodes -- everything that is not current
# --------------------------------------------------------------------------- #

def test_stale_nodes_returns_missing_and_stale_but_not_current():
    """Missing counts. An artifact that is gone is not 'fine', and a caller
    asking what needs attention needs both in one list."""
    graph = pg.PipelineGraph(
        nodes=(_node(artifact_id="ok", state=pg.STATE_CURRENT),
               _node(artifact_id="old", state=pg.STATE_STALE),
               _node(artifact_id="gone", state=pg.STATE_MISSING)),
        edges=(),
    )

    ids = {node.artifact_id for node in graph.stale_nodes()}
    assert ids == {"old", "gone"}


def test_stale_nodes_is_empty_on_a_clean_graph():
    graph = pg.PipelineGraph(nodes=(_node(),), edges=())
    assert graph.stale_nodes() == ()


# --------------------------------------------------------------------------- #
#  stale_summary -- the sentence a screen shows
# --------------------------------------------------------------------------- #

def test_the_summary_counts_missing_and_stale_separately():
    """They need different actions -- re-run versus re-register -- so a
    summary that adds them together tells the user to do the wrong thing."""
    graph = pg.PipelineGraph(
        nodes=(_node(artifact_id="ok"),
               _node(artifact_id="old", state=pg.STATE_STALE),
               _node(artifact_id="gone", state=pg.STATE_MISSING),
               _node(artifact_id="gone2", state=pg.STATE_MISSING)),
        edges=(),
    )

    summary = pg.stale_summary(graph)

    assert summary["n_missing"] == 2
    assert summary["n_stale"] == 1
    verdict = str(summary.get("verdict", ""))
    assert "2" in verdict and "1" in verdict


def test_the_summary_of_a_clean_graph_does_not_read_as_a_problem():
    graph = pg.PipelineGraph(nodes=(_node(), _node(artifact_id="a2")),
                             edges=())

    summary = pg.stale_summary(graph)

    assert summary["n_missing"] == 0
    assert summary["n_stale"] == 0
