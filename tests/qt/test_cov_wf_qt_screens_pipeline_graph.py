"""The Pipeline Graph screen where the recorded provenance is thin or absent.

``tests/qt/test_pipeline_graph_screen.py`` and
``tests/qt/test_cov_pipeline_graph_screen.py`` drive this screen against a
registry that recorded everything: a run id, a settings digest, a spaCR
version, a timestamp, a port role. What neither reaches is the *other* shape
of a real project — an artifact registered by an older spaCR, or by hand, or
by a run that died before it could stamp its metadata. Every optional line of
:meth:`PipelineGraphScreen.describe` is skipped for such a node, and the
detail pane has to stay a readable block rather than collapse into a heading.

The rest is the two "the platform gave me nothing" seams: a clipboard the
window system will not hand over, and a widget with no style to repolish.
Both are silent-by-design, and a screen that raises out of a button press
leaves a window that no longer redraws.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel                      # noqa: E402

from spacr import artifacts                               # noqa: E402
from spacr.pipeline_graph import (STATE_CURRENT,          # noqa: E402
                                  STATE_STALE, Edge, Node, PipelineGraph)
from spacr.qt.screens import pipeline_graph as screen_module  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Never let a developer's real artifact database into these tests."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


def _bare_node(artifact_id: str = "bare", depth: int = 0,
               state: str = STATE_CURRENT) -> Node:
    """An artifact the registry knows the path of and nothing else.

    No role, no run id, no timestamp, no version, no settings digest — the
    row a pre-provenance spaCR, or an interrupted run, leaves behind.
    """
    return Node(artifact_id=artifact_id, project="/p", kind="masks",
                role="", module="mask", path=f"/p/masks/{artifact_id}.npy",
                state=state, depth=depth)


def _full_node(artifact_id: str = "full", depth: int = 1,
               state: str = STATE_STALE) -> Node:
    """The same artifact with every optional field the screen can print."""
    return Node(
        artifact_id=artifact_id, project="/p", kind="measurements-db",
        role="measurements", module="measure",
        path=f"/p/measure/{artifact_id}.db", state=state, depth=depth,
        run_id="run-77", settings_hash="0123456789abcdef0123456789abcdef",
        spacr_version="1.2.3", created_utc="2026-01-02T03:04:05Z",
        size_bytes=2048, n_files=3,
        reasons=("its cell masks were written again afterwards",),
        causes=("upstream-newer",))


def _graph(*nodes: Node, edges=()) -> PipelineGraph:
    """A graph holding exactly these nodes, one layer per depth."""
    layers: dict = {}
    for node in nodes:
        layers.setdefault(node.depth, []).append(node.artifact_id)
    return PipelineGraph(
        project="/p", nodes=tuple(nodes), edges=tuple(edges),
        layers=tuple(tuple(layers[depth]) for depth in sorted(layers)))


@pytest.fixture
def canvas(qtbot):
    """A bare canvas, sized so every box of a small graph is on it."""
    widget = screen_module.GraphCanvas()
    qtbot.addWidget(widget)
    widget.resize(900, 400)
    return widget


@pytest.fixture
def screen(qtbot):
    """The whole screen, running its jobs inline so a test can drive it."""
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# A redraw that keeps the box the user was reading about
# ---------------------------------------------------------------------------

def test_a_redraw_that_still_shows_the_box_keeps_it_selected(canvas):
    """Ticking a filter box re-pushes the whole graph into the canvas. If
    that dropped the selection every time, the ring would jump off the box
    the user is reading about in the detail pane — the pane would still
    describe an artifact that no longer looks chosen, which is how somebody
    reads one artifact's provenance under another's name. The selection may
    only be cleared when the box it points at has actually gone."""
    graph = _graph(_bare_node("a"), _full_node("b"))
    canvas.set_graph(graph)
    canvas.select("b")
    assert canvas.selected == "b"

    # Same graph, both boxes still shown: the selection survives.
    canvas.set_graph(graph, visible={"a", "b"})
    assert canvas.selected == "b"
    assert set(canvas.node_rects()) == {"a", "b"}

    # Now hide it, and only now is it dropped.
    canvas.set_graph(graph, visible={"a"})
    assert canvas.selected == ""
    assert set(canvas.node_rects()) == {"a"}


def test_a_filtered_redraw_keeps_the_selection_when_the_filter_spares_it(
        screen):
    """Same guarantee through the real filter toggles rather than the canvas
    API: unticking "Stale" must not un-select a current box that is still on
    screen, or the panel and the picture disagree after every filter click."""
    graph = _graph(_bare_node("a"), _full_node("b"))
    screen._on_graph_ready(graph)
    screen._canvas.select("a")
    assert screen._canvas.selected == "a"

    screen._filters[STATE_STALE].setChecked(False)
    assert set(screen._canvas.node_rects()) == {"a"}
    assert screen._canvas.selected == "a"


# ---------------------------------------------------------------------------
# describe() for an artifact the registry recorded almost nothing about
# ---------------------------------------------------------------------------

def test_an_artifact_with_no_recorded_provenance_still_describes_itself(
        screen):
    """An artifact registered by an older spaCR — or by a run that died
    before stamping its metadata — has no role, run id, timestamp, version
    or settings digest. Every one of those lines is optional, and if the
    pane printed the labels with nothing after them the user would read
    "run id:" as "this was produced by a run with an empty id" instead of
    "nobody recorded which run made this". What must survive is the part
    that is always known: what it is, where it is, and how big it is."""
    graph = _graph(_bare_node("a"), _full_node("b"),
                   edges=[Edge(source="a", target="b", kind="masks")])
    screen._on_graph_ready(graph)

    thin = screen.describe("a")
    lines = thin.splitlines()
    assert lines[:3] == ["mask → masks", "state: current",
                         "path: /p/masks/a.npy"]
    assert "size: 0 bytes in 0 file(s)" in lines
    # None of the five optional lines may be printed empty.
    assert not [line for line in lines
                if line.startswith(("role:", "run id:", "produced:",
                                    "spaCR:", "settings digest:"))]
    # ...and the same pane does print all five for the node that has them.
    rich = screen.describe("b").splitlines()
    assert "role: measurements" in rich
    assert "run id: run-77" in rich
    assert "produced: 2026-01-02T03:04:05Z" in rich
    assert "spaCR: 1.2.3" in rich
    assert "settings digest: 0123456789abcdef" in rich


def test_the_thin_artifact_still_names_what_it_fed_and_what_fed_it(screen):
    """Provenance is the whole point of the screen, and it is recorded on
    the edge, not on the node's metadata. A node with no run id must still
    say what re-running it would invalidate — otherwise the artifacts spaCR
    knows least about are exactly the ones it warns least about."""
    graph = _graph(_bare_node("a"), _full_node("b"),
                   edges=[Edge(source="a", target="b", kind="masks")])
    screen._on_graph_ready(graph)

    upstream_end = screen.describe("a")
    assert "Re-running this would invalidate:" in upstream_end
    assert "  • measure measurements-db at /p/measure/b.db" in upstream_end

    downstream_end = screen.describe("b")
    assert "Made from:" in downstream_end
    assert "  • mask masks (current)" in downstream_end
    assert "Nothing was derived from this." in downstream_end
    assert "Why it is flagged:" in downstream_end
    assert ("  • its cell masks were written again afterwards"
            in downstream_end)


def test_clicking_the_thin_artifact_fills_the_pane_rather_than_blanking_it(
        screen):
    """The detail pane is the screen's only prose. A click that produced an
    empty block would read as "this box has no provenance at all", so the
    describe() text has to reach the widget verbatim; only a click on the
    background falls back to the invitation."""
    screen._on_graph_ready(_graph(_bare_node("a")))
    screen._on_node_clicked("a")
    shown = screen._details.toPlainText()
    assert shown == screen.describe("a")
    assert "path: /p/masks/a.npy" in shown

    screen._on_node_clicked("")
    assert screen._details.toPlainText() == "Click a box for its provenance."


# ---------------------------------------------------------------------------
# A clipboard the window system will not hand over
# ---------------------------------------------------------------------------

class _NoClipboardApp:
    """Stands in for ``QApplication`` on a session that has no clipboard."""

    @staticmethod
    def clipboard():
        """What Qt returns before a display connection exists."""
        return None


def test_copying_the_graph_without_a_clipboard_says_nothing_and_raises_none(
        screen):
    """"Copy Graphviz" is a button on a screen whose promise is that it
    never writes anything. On a headless or freshly-started session
    ``QApplication.clipboard()`` can be ``None``, and an unguarded
    ``setText`` there is an AttributeError raised inside a button handler —
    which in Qt tears down the click, not the window, and leaves a screen
    that looks alive and no longer responds. Nothing is copied and the
    previous banner has to stay, because claiming a copy that did not
    happen sends the user to paste an empty methods figure."""
    from PySide6 import QtWidgets

    graph = _graph(_bare_node("a"), _full_node("b"),
                   edges=[Edge(source="a", target="b", kind="masks")])
    screen._on_graph_ready(graph)

    # With a real clipboard the DOT text really is copied and announced.
    screen._on_copy_dot()
    assert screen._verdict.text() == "2 node(s) copied as Graphviz DOT."
    copied = QtWidgets.QApplication.clipboard().text()
    assert copied.startswith("digraph spacr {")
    assert '"a" -> "b"' in copied

    # With no clipboard at all: no exception, no claim, banner untouched.
    # Restored by hand rather than with monkeypatch, because pytest-qt pumps
    # the event loop through ``QApplication.instance()`` after the call phase
    # and before a fixture finalizer would put the real class back.
    screen._set_verdict("2 of 2 artifact(s) stale.", problem=True)
    real_app = QtWidgets.QApplication
    QtWidgets.QApplication = _NoClipboardApp
    try:
        screen._on_copy_dot()
    finally:
        QtWidgets.QApplication = real_app
    assert screen._verdict.text() == "2 of 2 artifact(s) stale."


def test_copying_before_a_project_is_open_says_there_is_no_graph(screen):
    """The button is enabled from the first paint, so it will be pressed
    before a project is chosen. Silence there reads as a successful copy of
    an empty graph; the banner has to say the copy did not happen."""
    assert screen.graph() is None
    screen._on_copy_dot()
    assert screen._verdict.text() == "There is no graph to copy yet."
    assert screen._verdict.property("problem") == "true"


# ---------------------------------------------------------------------------
# A verdict label with no style to repolish
# ---------------------------------------------------------------------------

class _StylelessLabel(QLabel):
    """A label that reports no style, as one does while being torn down."""

    def style(self):
        """Qt hands back the application style; a dying widget need not."""
        return None


def test_a_verdict_survives_a_widget_that_has_no_style_to_repolish(screen):
    """The banner's colour comes from a dynamic ``problem`` property, which
    only takes effect after the style is unpolished and polished again. The
    repolish is the cosmetic half; the text is the half that carries the
    warning. If a missing style took the whole method down with it, the
    error the user needs to read — "could not read that project" — would
    never be written at all, and the screen would sit on its previous, now
    wrong, verdict."""
    real = screen._verdict
    screen._set_verdict("Could not read that project: no registry",
                        problem=True)
    assert real.text() == "Could not read that project: no registry"
    assert real.property("problem") == "true"

    screen._verdict = _StylelessLabel("")
    screen._set_verdict("All 3 artifact(s) current.", problem=False)
    assert screen._verdict.text() == "All 3 artifact(s) current."
    assert screen._verdict.property("problem") == "false"


def test_a_background_failure_lands_in_the_banner_not_in_a_dialog(screen):
    """A modal dialog on a headless run hangs the process forever, so the
    job runner's failures have to end up as text on the screen and on
    ``last_error`` where a test — or the next load — can read them."""
    screen._on_job_failed("registry is locked")
    assert screen.last_error == "registry is locked"
    assert screen._verdict.text() == (
        "Could not read that project: registry is locked")
    assert screen._verdict.property("problem") == "true"
