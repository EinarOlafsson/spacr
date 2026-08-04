"""Real tests for :mod:`spacr.pipeline_graph`.

Every assertion is about a question a user asks of the graph: what produced
what, which of it can still be believed, and what a re-run would invalidate.
The registry underneath is a real SQLite file with real artifacts registered
into it in the real order, because the whole point of the module is that the
staleness verdicts come from provenance rather than from a heuristic.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import artifacts, pipeline_graph  # noqa: E402
from spacr.pipeline_graph import (STATE_CURRENT, STATE_MISSING,  # noqa: E402
                                  STATE_STALE, build_graph, format_graph,
                                  module_graph, stale_summary, to_dot)


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
    """A project root with a registry and a mask -> measure -> classify chain."""
    root = tmp_path / "plate7"
    root.mkdir()
    registry = artifacts.Registry(project=str(root))

    masks = registry.register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(root / "masks" / "cell.npy"),
        settings={"diameter": 30}, run_id="run-mask-1")
    # time_ns is the ordering key, and two registrations inside the same
    # nanosecond would make "produced after" untestable.
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


# ---------------------------------------------------------------------------
# the static module DAG
# ---------------------------------------------------------------------------

def test_the_module_graph_puts_mask_before_measure_before_classify():
    graph = module_graph()

    assert ("mask", "measure") in graph.edges
    assert ("measure", "classify") in graph.edges

    column = {module: index
              for index, row in enumerate(graph.layers) for module in row}
    assert column["mask"] < column["measure"] < column["classify"], (
        f"declared order came out as {graph.layers}")


def test_the_module_graph_covers_every_module_exactly_once():
    graph = module_graph()

    flattened = [module for row in graph.layers for module in row]
    assert sorted(flattened) == sorted(graph.modules)
    assert len(flattened) == len(set(flattened))


def test_module_graph_records_which_modules_actually_ran():
    graph = module_graph(ran=["mask", "measure", "not_a_module"])

    assert graph.ran == ("mask", "measure"), (
        "a module key that is not in PORTS must not be reported as having run")


def test_module_graph_neighbours_agree_with_ports():
    from spacr import ports

    graph = module_graph()
    known = set(graph.modules)
    for module in graph.modules:
        assert graph.next_of(module) == tuple(
            m for m in ports.next_modules(module) if m in known)
        assert graph.previous_of(module) == tuple(
            m for m in ports.upstream_modules(module) if m in known), (
            "the drawn edges must be exactly the ones ports declares")


def test_layering_survives_a_cycle_instead_of_hanging():
    layers = pipeline_graph._layer(
        ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "b")])

    flattened = [n for row in layers for n in row]
    assert sorted(flattened) == ["a", "b", "c"], (
        "every node must still be placed when the edges hold a cycle")


# ---------------------------------------------------------------------------
# the artifact DAG
# ---------------------------------------------------------------------------

def test_a_project_with_no_registry_is_an_empty_graph_with_a_note(tmp_path):
    graph = build_graph(tmp_path / "never-run")

    assert len(graph) == 0
    assert graph.notes and "registry" in graph.notes[0].lower()
    assert graph.modules.modules, (
        "an empty project must still show the pipeline it is about to run")
    assert bool(graph) is True


def test_the_graph_reproduces_the_chain_that_was_registered(project):
    graph = build_graph(project["root"])

    assert len(graph) == 3
    assert {n.module for n in graph.nodes} == {"mask", "measure", "classify"}

    depth = {n.module: n.depth for n in graph.nodes}
    assert depth["mask"] == 0
    assert depth["measure"] == 1
    assert depth["classify"] == 2

    assert [n.module for n in graph.roots()] == ["mask"]
    assert [n.module for n in graph.leaves()] == ["classify"]


def test_every_edge_names_a_real_input(project):
    graph = build_graph(project["root"])

    pairs = {(e.source, e.target) for e in graph.edges}
    assert (project["masks"].artifact_id, project["db"].artifact_id) in pairs
    assert (project["db"].artifact_id, project["preds"].artifact_id) in pairs
    assert not any(e.dangling for e in graph.edges)


def test_nothing_is_stale_until_something_is_re_run(project):
    graph = build_graph(project["root"])

    assert [n.state for n in graph.nodes] == [STATE_CURRENT] * 3
    assert stale_summary(graph)["verdict"] == "All 3 artifact(s) current."


def test_re_running_mask_makes_everything_downstream_stale(project):
    registry = project["registry"]
    time.sleep(0.002)
    registry.register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(project["root"] / "masks" / "cell.npy", "changed"),
        settings={"diameter": 45}, run_id="run-mask-2")

    graph = build_graph(project["root"])
    by_module = {n.module: n for n in graph.nodes if n.module != "mask"}

    assert by_module["measure"].state == STATE_STALE
    assert by_module["classify"].state == STATE_STALE
    assert by_module["measure"].causes, "a stale node must say why"
    assert artifacts.CAUSE_UPSTREAM_SUPERSEDED in by_module["measure"].causes
    assert artifacts.CAUSE_UPSTREAM_STALE in by_module["classify"].causes
    assert any("was re-produced by mask" in reason
               for reason in by_module["measure"].reasons)


def test_downstream_answers_what_a_re_run_invalidates(project):
    graph = build_graph(project["root"])

    downstream = graph.downstream(project["masks"].artifact_id)
    assert [n.module for n in downstream] == ["measure", "classify"], (
        "downstream must be transitive, not just the immediate child")

    upstream = graph.upstream(project["preds"].artifact_id)
    assert [n.module for n in upstream] == ["mask", "measure"], (
        "both directions are returned in pipeline order, shallowest first")

    assert graph.downstream(project["preds"].artifact_id) == ()
    assert graph.upstream(project["masks"].artifact_id) == ()


def test_the_graph_walk_agrees_with_the_registrys_own(project):
    """The in-memory walk is a shortcut, not a second opinion.

    ``Registry.downstream_of`` / ``upstream_of`` re-open the database per
    call; the graph answers the same questions off the edges it already has.
    They have to give the same answer or the picture and the registry
    disagree about what a re-run invalidates.
    """
    registry = project["registry"]
    graph = build_graph(project["root"])

    for artifact in (project["masks"], project["db"], project["preds"]):
        assert ({n.artifact_id for n in graph.downstream(artifact.artifact_id)}
                == {a.artifact_id for a in registry.downstream_of(artifact)})
        assert ({n.artifact_id for n in graph.upstream(artifact.artifact_id)}
                == {a.artifact_id
                    for a in registry.upstream_of(artifact, transitive=True)})


def test_a_deleted_file_is_missing_and_not_merely_stale(project):
    os.remove(project["db"].path)

    graph = build_graph(project["root"])
    node = next(n for n in graph.nodes if n.module == "measure")

    assert node.state == STATE_MISSING
    assert node.exists is False
    summary = stale_summary(graph)
    assert summary["n_missing"] == 1
    assert "missing" in summary["verdict"]


def test_a_forgotten_input_is_a_dangling_edge_and_a_note(project):
    project["registry"].forget(project["masks"])

    graph = build_graph(project["root"])

    dangling = [e for e in graph.edges if e.dangling]
    assert len(dangling) == 1
    assert dangling[0].target == project["db"].artifact_id
    assert any("no longer in the registry" in note for note in graph.notes)
    measure = next(n for n in graph.nodes if n.module == "measure")
    assert measure.state == STATE_STALE
    assert artifacts.CAUSE_UPSTREAM_MISSING in measure.causes
    # The measure node lost its only parent, so it is now a root of the graph.
    assert measure.depth == 0


def test_changed_settings_make_the_node_stale_before_anything_is_overwritten(project):
    graph = build_graph(project["root"], settings={"diameter": 999})

    mask = next(n for n in graph.nodes if n.module == "mask")
    assert mask.state == STATE_STALE
    assert artifacts.CAUSE_SETTINGS_CHANGED in mask.causes


def test_nodes_carry_the_provenance_a_methods_section_needs(project):
    graph = build_graph(project["root"])
    mask = next(n for n in graph.nodes if n.module == "mask")

    assert mask.run_id == "run-mask-1"
    assert mask.settings_hash, "the settings digest must reach the node"
    assert mask.spacr_version, "the producing version must reach the node"
    assert mask.kind == "masks"
    assert mask.role == "cell_mask"
    assert mask.status == artifacts.STATUS_COMPLETE
    assert mask.to_dict()["artifact_id"] == mask.artifact_id


def test_node_lookup_and_grouping(project):
    graph = build_graph(project["root"])

    assert graph.node("no-such-id") is None
    assert graph.node(project["db"].artifact_id).module == "measure"
    grouped = graph.by_module()
    assert set(grouped) == {"classify", "mask", "measure"}
    assert len(grouped["mask"]) == 1


def test_the_graph_records_which_modules_have_produced_something(project):
    graph = build_graph(project["root"])

    assert graph.modules.ran == ("classify", "mask", "measure")


def test_limit_reads_only_the_newest_artifacts(project):
    graph = build_graph(project["root"], limit=1)

    assert len(graph) == 1
    assert graph.nodes[0].module == "classify"


def test_an_explicit_registry_is_used_as_given(project, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()

    graph = build_graph(other, registry=project["registry"])

    assert len(graph) == 0, (
        "a registry filtered to another project root must come back empty")

    everything = build_graph(other, registry=project["registry"],
                             all_projects=True)
    assert len(everything) == 3


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_the_text_rendering_names_every_module_and_marks_the_stale_one(project):
    time.sleep(0.002)
    project["registry"].register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(project["root"] / "masks" / "cell.npy", "changed"),
        settings={"diameter": 45})

    text = format_graph(build_graph(project["root"]))

    for module in ("mask", "measure", "classify"):
        assert module in text
    assert "STALE" in text
    assert "Step 1" in text and "Step 2" in text
    assert not text.endswith("\n")


def test_the_text_rendering_of_an_empty_project_shows_the_declared_order(tmp_path):
    text = format_graph(build_graph(tmp_path / "nothing"))

    assert "Declared module order" in text
    assert "mask" in text


def test_dot_output_is_well_formed_and_colours_the_states(project):
    os.remove(project["preds"].path)

    dot = to_dot(build_graph(project["root"]))

    assert dot.startswith("digraph spacr {")
    assert dot.rstrip().endswith("}")
    assert dot.count("->") == 2
    assert "#f4cccc" in dot, "a missing artifact must be coloured differently"
    assert "#d9ead3" in dot


def test_a_long_path_is_elided_from_the_middle():
    text = pipeline_graph._elide("/very/long/" + "a" * 200 + "/file.csv", 40)

    assert len(text) <= 40
    assert text.startswith("/very/long/")
    assert text.endswith("file.csv")
    assert "..." in text


def test_stale_summary_tallies_causes(project):
    time.sleep(0.002)
    project["registry"].register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(project["root"] / "masks" / "cell.npy", "changed"),
        settings={"diameter": 45})

    summary = stale_summary(build_graph(project["root"]))

    assert summary["n_nodes"] == 4
    assert summary["n_stale"] == 2
    assert summary["causes"], "the tally must name the codes"
    assert set(summary["causes"]) <= {
        artifacts.CAUSE_UPSTREAM_SUPERSEDED, artifacts.CAUSE_UPSTREAM_STALE,
        artifacts.CAUSE_UPSTREAM_NEWER}
    assert summary["modules"] == ["classify", "mask", "measure"]


def test_the_graph_is_json_serializable(project):
    import json

    payload = json.dumps(build_graph(project["root"]).to_dict())

    assert "artifact_id" in payload
    restored = json.loads(payload)
    assert len(restored["nodes"]) == 3
    assert restored["modules"]["ran"] == ["classify", "mask", "measure"]
