from __future__ import annotations

import base64
import xml.etree.ElementTree as ET

import pytest

from spacr import flowview
from spacr.flowview.export import (
    _edge_width,
    _number,
    _thumbnail_uri,
    export,
    export_graph,
    render_html,
    render_svg,
)
from spacr.flowview.model import Edge, Node, NodeKind, NodeState, RunGraph
from spacr.flowview.thumbs import thumbnail_png


def rich_graph(thumbnail: str, missing: str) -> RunGraph:
    nodes = {
        "raw<&\"": Node(
            "raw<&\"",
            "Raw <images> & metadata",
            NodeKind.INPUT,
            state=NodeState.RUNNING,
            started_at=1.0,
            progress=(-1, 10),
            metrics={"objects": 1000, "quality": 0.123456789, "note": "ok"},
            thumbnail=thumbnail,
            params={"path": "a&b"},
        ),
        "model": Node(
            "model",
            "A model label long enough to wrap cleanly across two lines",
            NodeKind.PROCESS,
            state=NodeState.DONE,
            started_at=2.0,
            ended_at=4.5,
            progress=(12, 10),
            metrics={"z-last": 4, "a-first": "yes", "middle": 2.0, "zz-unused": 9},
            thumbnail=missing,
            params={"family": "torch", "folds": [1, 2]},
        ),
        "failed": Node(
            "failed",
            "Scores",
            NodeKind.OUTPUT,
            state=NodeState.FAILED,
            ended_at=5.0,
            progress=(1, 0),
            error="bad <trace>\nsecond line",
        ),
        "empty": Node(
            "empty",
            "",
            NodeKind.OUTPUT,
            state=NodeState.SKIPPED,
        ),
    }
    return RunGraph(
        run_id='run<&"',
        started_at=1.0,
        nodes=nodes,
        edges=[
            Edge("model", "failed"),
            Edge("raw<&\"", "model", label="1,000 files & rows", volume=1_000_000),
            Edge("model", "empty", label="zero", volume=0),
        ],
        spacr_version="1.5.0.4",
        settings_digest="abc<&",
    )


def test_svg_is_valid_deterministic_editable_and_encodes_visual_semantics(tmp_path):
    png = thumbnail_png([[0, 1], [2, 3]])
    thumb = tmp_path / "preview.png"
    thumb.write_bytes(png)
    graph = rich_graph(str(thumb), str(tmp_path / "missing.png"))

    first = render_svg(graph)
    second = render_svg(graph)
    root = ET.fromstring(first)
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    assert first == second
    assert root.tag.endswith("svg")
    assert root.findall(".//svg:text", namespace)
    assert root.findall(".//svg:path", namespace)
    assert all(element.text != "Raw <images> & metadata" for element in root.findall(".//svg:path", namespace))
    assert "Raw &lt;images&gt; &amp; metadata" in first
    assert "data:image/png;base64," + base64.b64encode(png).decode("ascii") in first
    assert str(thumb) not in first
    assert "stroke-dasharray=\"7 6\"" in first
    assert ">RUNNING<" in first and ">FAILED<" in first and ">SKIPPED<" in first
    assert "C " in first
    assert "1,000,000 transferred" in first
    assert "bad &lt;trace&gt;" not in first  # full errors live in the inspector
    assert first.count("zz-unused: 9") == 0  # only three quiet metrics per card


def test_html_is_one_file_with_same_svg_and_complete_escaped_inspector(tmp_path):
    thumb = tmp_path / "sample.jpg"
    thumb.write_bytes(b"not-a-real-jpeg-but-embeddable")
    graph = rich_graph(str(thumb), "absent.png")

    svg = render_svg(graph)
    document = render_html(graph)

    assert svg in document
    assert "data:image/jpeg;base64," in document
    assert str(thumb) not in document
    assert "<table>" in document
    assert "Identifier" in document and "Parameters" in document
    assert "bad &lt;trace&gt;\nsecond line" in document
    assert "2.5" in document  # duration
    assert "run&lt;&amp;&quot;" in document
    assert "abc&lt;&amp;" in document


def test_export_dispatches_all_formats_and_is_byte_identical(tmp_path):
    graph = rich_graph("missing.png", "also-missing.png")
    svg_path = tmp_path / "run.svg"
    html_path = tmp_path / "run.html"
    json_path = tmp_path / "run.json"

    assert export(graph, svg_path) == svg_path
    assert flowview.export(graph, svg_path) == svg_path
    first_svg = svg_path.read_bytes()
    export_graph(graph, svg_path, ".SVG")
    assert svg_path.read_bytes() == first_svg
    assert export(graph, html_path, "HTML") == html_path
    assert html_path.read_text(encoding="utf-8") == render_html(graph)
    assert export(graph, json_path, "json") == json_path
    assert json_path.read_text(encoding="utf-8") == graph.to_json()

    untouched = tmp_path / "unknown.out"
    with pytest.raises(ValueError, match="fmt must"):
        export(graph, untouched, "pdf")
    assert not untouched.exists()


def test_renderer_helpers_cover_fallbacks_and_numeric_widths(tmp_path):
    assert _number(4.0) == "4"
    assert _number(1.23456) == "1.235"
    assert _edge_width(None) == 1.0
    assert _edge_width(0) == 1.0
    assert 1.0 < _edge_width(10) < 6.0
    assert _edge_width(10**20) == 6.0
    assert _thumbnail_uri(None) is None
    assert _thumbnail_uri(str(tmp_path / "absent")) is None

    gif = tmp_path / "x.GIF"
    gif.write_bytes(b"gif")
    webp = tmp_path / "x.webp"
    webp.write_bytes(b"webp")
    svg = tmp_path / "x.svg"
    svg.write_bytes(b"svg")
    assert _thumbnail_uri(str(gif)).startswith("data:image/gif;base64,")
    assert _thumbnail_uri(str(webp)).startswith("data:image/webp;base64,")
    assert _thumbnail_uri(str(svg)).startswith("data:image/svg+xml;base64,")
