"""The pipeline graph's failure paths: what it reports when things go wrong.

Every path here is one the module was written to survive rather than to
raise from, because the caller is a drawing routine and an exception there
is a blank screen where a user expected to read whether a week of compute
still counts. So each test asserts the *answer* the graph gives — the note,
the node, the rendered line — not merely that nothing blew up:

* a registry file that disappears between the existence check and the open;
* a registry that cannot answer about one artifact while a job holds it;
* an input that was forgotten, so an edge points at nothing;
* an edge naming a node that is not being laid out;
* a layer naming an artifact the graph does not hold.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import artifacts, pipeline_graph  # noqa: E402
from spacr.pipeline_graph import (STATE_CURRENT, STATE_STALE,  # noqa: E402
                                  Edge, ModuleGraph, Node, PipelineGraph,
                                  build_graph, format_graph)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """No test may inherit a shared-registry override from the environment."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


def _write(path: Path, text: str = "x") -> str:
    """Create a file with content and return its absolute path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def project(tmp_path):
    """A project root with a real mask -> measure -> classify chain."""
    root = tmp_path / "plate9"
    root.mkdir()
    registry = artifacts.Registry(project=str(root))

    masks = registry.register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(root / "masks" / "cell.npy"),
        settings={"diameter": 30}, run_id="run-mask-1")
    # created_ns is the ordering key; two registrations inside one nanosecond
    # would make "produced after" unassertable.
    time.sleep(0.002)
    db = registry.register(
        module="measure", kind="measurements-db", role="merged",
        path=_write(root / "measurements" / "measurements.db"),
        settings={"channels": [0, 1]}, inputs=[masks], run_id="run-measure-1")
    time.sleep(0.002)
    preds = registry.register(
        module="classify", kind="predictions", role="predictions",
        path=_write(root / "results" / "predictions.csv"),
        settings={"epochs": 5}, inputs=[db], run_id="run-classify-1")
    return {"root": root, "registry": registry, "masks": masks, "db": db,
            "preds": preds}


# --------------------------------------------------------------------------- #
#  A registry that goes away, or will not answer
# --------------------------------------------------------------------------- #

def test_a_registry_that_vanishes_after_the_check_is_a_note_not_an_exception(
        tmp_path, monkeypatch):
    """The registry is shared with running jobs, so it can be removed between
    ``os.path.isfile`` saying yes and ``Registry`` opening it. That race must
    come back as the same "nothing to show" answer as a project that never
    had a registry, because the caller draws the result either way."""
    root = tmp_path / "raced"
    root.mkdir()
    target = os.path.join(str(root), artifacts.ARTIFACTS_DB_NAME)
    real_isfile = os.path.isfile
    seen = []

    def vanishing_isfile(path):
        """True the first time the registry is asked about, False after —
        exactly what a concurrent delete looks like from in here."""
        if os.fspath(path) == target:
            seen.append(path)
            return len(seen) == 1
        return real_isfile(path)

    monkeypatch.setattr(os.path, "isfile", vanishing_isfile)
    graph = build_graph(root)

    assert len(seen) == 2, (
        "build_graph must check, then hand the same path to Registry")
    assert len(graph) == 0
    assert graph.edges == ()
    assert graph.registry_file == target
    assert graph.project == str(root)
    assert len(graph.notes) == 1
    assert graph.notes[0].startswith("Could not open the artifact registry: ")
    assert target in graph.notes[0], (
        "the note must name the file the user has to go look at")
    assert "mask" in graph.modules.modules, (
        "the declared pipeline is still drawable when the registry is not")


def test_a_registry_that_cannot_answer_about_one_artifact_keeps_the_graph(
        project):
    """A row can be locked away by a running job mid-walk. Losing the whole
    graph over one artifact would be the wrong trade: the other verdicts are
    still true, so the unanswerable one is reported as unanswerable."""
    raced_id = project["db"].artifact_id

    class RacedRegistry(artifacts.Registry):
        def is_stale(self, artifact, *, settings=None):
            if artifact == raced_id:
                raise sqlite3.OperationalError("database is locked")
            return super().is_stale(artifact, settings=settings)

    registry = RacedRegistry(path=project["registry"].path,
                             project=str(project["root"]), create=False)
    graph = build_graph(project["root"], registry=registry)

    assert len(graph) == 3, "one unanswerable row must not drop the other two"
    raced = graph.node(raced_id)
    assert raced.reasons == ("Could not check this artifact's provenance.",)
    assert raced.causes == ()
    assert raced.state == STATE_CURRENT, (
        "an unchecked artifact must not be accused of being stale")
    assert raced.stale is False
    # The neighbours were still asked, and answered for real.
    assert graph.node(project["masks"].artifact_id).state == STATE_CURRENT
    assert "! Could not check this artifact's provenance." in format_graph(
        graph), "the rendering must show that this one was not checked"


def test_a_real_stale_verdict_still_reaches_the_node_when_nothing_races(
        project):
    """The other side of the guarded call: an artifact the registry *can*
    answer about gets the registry's real reasons, not the fallback sentence."""
    time.sleep(0.002)
    project["registry"].register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(project["root"] / "masks" / "cell.npy", "re-run"),
        settings={"diameter": 45})

    graph = build_graph(project["root"])
    measure = next(n for n in graph.nodes if n.module == "measure")

    assert measure.state == STATE_STALE
    assert measure.causes, "a real verdict carries machine causes"
    assert "Could not check this artifact's provenance." not in measure.reasons


# --------------------------------------------------------------------------- #
#  Walking past what is no longer there
# --------------------------------------------------------------------------- #

def test_upstream_reports_the_ancestors_that_exist_and_skips_the_forgotten_one(
        project):
    """The edge to a forgotten input is deliberately kept — it is what makes
    the target stale — so the reachability walk meets an id with no node.
    That id must be stepped over, not reported as an ancestor and not
    crashed on."""
    project["registry"].forget(project["masks"])

    graph = build_graph(project["root"])
    ancestors = graph.upstream(project["preds"].artifact_id)

    assert [n.artifact_id for n in ancestors] == [project["db"].artifact_id], (
        "measure is a real ancestor; the forgotten mask is not a node")
    assert project["masks"].artifact_id not in {n.artifact_id
                                                for n in ancestors}
    assert any(e.dangling and e.source == project["masks"].artifact_id
               for e in graph.edges), (
        "the walk must have been offered the forgotten id to step over")
    assert graph.upstream(project["db"].artifact_id) == (), (
        "an artifact whose only input was forgotten has no drawable ancestor")


def test_layering_ignores_an_edge_that_names_a_node_it_is_not_placing():
    """``_layer`` is handed edges from a graph that may be a subset of the
    one they came from. An edge to an id outside the node list has no column
    to point at, so it must be dropped rather than raise, and the nodes that
    are being placed must be laid out as if it were not there."""
    layers = pipeline_graph._layer(
        ["a", "b"], [("a", "b"), ("a", "zz"), ("zz", "b")])

    assert layers == (("a",), ("b",)), (
        "the unknown endpoint must neither be placed nor shift a real node")


# --------------------------------------------------------------------------- #
#  Building without a project root
# --------------------------------------------------------------------------- #

def test_build_graph_with_no_project_covers_the_registry_it_was_given(project):
    """The shared-registry case: no project root is named, so the graph takes
    its identity from the registry rather than from an absolute path that was
    never supplied."""
    graph = build_graph(registry=project["registry"], all_projects=True)

    assert graph.project == str(project["root"]), (
        "with no root given, the registry's own project names the graph")
    assert graph.registry_file == project["registry"].path
    assert sorted(n.module for n in graph.nodes) == ["classify", "mask",
                                                     "measure"]
    assert len(graph.edges) == 2
    assert format_graph(graph).startswith(
        f"Pipeline graph — {project['root']}")


# --------------------------------------------------------------------------- #
#  Rendering a graph whose layers and nodes disagree
# --------------------------------------------------------------------------- #

def test_rendering_skips_a_layer_entry_the_graph_has_no_node_for():
    """A ``PipelineGraph`` can be assembled by a caller — from ``to_dict``
    output, or by filtering a bigger graph — with a layer still naming an id
    whose node was dropped. The renderer prints what it has and steps over
    the rest rather than dying halfway down the page."""
    real = Node(artifact_id="keep", project="/p", kind="masks",
                role="cell_mask", module="mask", path="/p/masks/cell.npy",
                run_id="run-1", spacr_version="1.5.0", created_ns=1,
                status="complete", depth=0)
    graph = PipelineGraph(
        project="/p", nodes=(real,),
        edges=(Edge(source="ghost", target="keep", dangling=True),),
        layers=(("ghost", "keep"),),
        modules=ModuleGraph(modules=("mask",), layers=(("mask",),)),
        registry_file="/p/artifacts.db")

    text = format_graph(graph)

    drawn = [line for line in text.splitlines() if line.startswith("    [")]
    assert drawn == ["    [ok] mask -> masks (cell_mask)"], (
        "exactly one node line: none may be drawn for an id with no node")
    assert "/p/masks/cell.npy" in text
    assert "run run-1" in text
    assert "<- ghost (forgotten)" in text, (
        "the surviving node still shows where its input went")
    assert text.count("Step 1") == 1 and "Step 2" not in text
