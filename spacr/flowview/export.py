"""Deterministic, Qt-free SVG, HTML, and JSON exports for FlowView."""

from __future__ import annotations

import base64
import html
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any

from .layout import GraphLayout, NodeLayout, layout_graph
from .model import Edge, Node, NodeState, RunGraph
from .theme import (
    CANVAS,
    CARD,
    CORNER_RADIUS,
    FONT_FAMILY,
    LABEL_SIZE,
    METRIC_SIZE,
    STATE_SIZE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    THUMBNAIL_SIZE,
    node_accent,
    state_label,
)


def _escape(value: object, *, quote: bool = False) -> str:
    return html.escape(str(value), quote=quote)


def _number(value: float) -> str:
    rounded = round(float(value), 3)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def _edge_width(volume: int | None) -> float:
    if volume is None or volume <= 0:
        return 1.0
    return min(6.0, 1.0 + 0.8 * math.log10(volume + 1.0))


def _thumbnail_uri(path: str | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError:
        return None
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }.get(source.suffix.casefold(), "image/png")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _edge_key(edge: Edge) -> tuple[str, str, str, int]:
    return (edge.src, edge.dst, edge.label or "", edge.volume or 0)


def _render_edge(edge: Edge, layout: GraphLayout, graph: RunGraph) -> str:
    source = layout[edge.src]
    target = layout[edge.dst]
    start_x = source.x + source.width
    start_y = source.centre_y
    end_x = target.x
    end_y = target.centre_y
    bend = max(32.0, abs(end_x - start_x) * 0.45)
    path = (
        f"M {_number(start_x)} {_number(start_y)} "
        f"C {_number(start_x + bend)} {_number(start_y)}, "
        f"{_number(end_x - bend)} {_number(end_y)}, "
        f"{_number(end_x)} {_number(end_y)}"
    )
    running = graph.nodes[edge.src].state is NodeState.RUNNING
    dash = ' stroke-dasharray="7 6"' if running else ""
    title_parts = [edge.label] if edge.label else []
    if edge.volume is not None:
        title_parts.append(f"{edge.volume:,} transferred")
    title = " · ".join(title_parts) or f"{edge.src} to {edge.dst}"
    pieces = [
        f'<g class="edge" data-src="{_escape(edge.src, quote=True)}" '
        f'data-dst="{_escape(edge.dst, quote=True)}">',
        f"<title>{_escape(title)}</title>",
        f'<path d="{path}" fill="none" stroke="{TEXT_SECONDARY}" '
        f'stroke-opacity="0.72" stroke-width="{_number(_edge_width(edge.volume))}"'
        f'{dash} marker-end="url(#arrow)"/>',
    ]
    if title_parts:
        label_x = (start_x + end_x) / 2.0
        label_y = (start_y + end_y) / 2.0 - 7.0
        pieces.append(
            f'<text x="{_number(label_x)}" y="{_number(label_y)}" '
            f'fill="{TEXT_SECONDARY}" font-size="{METRIC_SIZE}" '
            f'text-anchor="middle">{_escape(" · ".join(title_parts))}</text>'
        )
    pieces.append("</g>")
    return "".join(pieces)


def _label_lines(label: str) -> list[str]:
    lines = textwrap.wrap(
        label,
        width=27,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return lines or [""]


def _metric_text(name: str, value: float | int | str) -> str:
    if isinstance(value, float):
        rendered = f"{value:.6g}"
    elif isinstance(value, int):
        rendered = f"{value:,}"
    else:
        rendered = str(value)
    return f"{name}: {rendered}"


def _render_node(node: Node, box: NodeLayout) -> str:
    accent = node_accent(node.kind, node.state)
    state = state_label(node.state)
    state_width = max(45.0, len(state) * 6.5 + 14.0)
    pieces = [
        f'<g class="node node-{node.kind.value}" '
        f'data-node-id="{_escape(node.id, quote=True)}">',
        f"<title>{_escape(node.label)} — {state}</title>",
        f'<rect x="{_number(box.x)}" y="{_number(box.y)}" '
        f'width="{_number(box.width)}" height="{_number(box.height)}" '
        f'rx="{CORNER_RADIUS}" fill="{CARD}" stroke="#FFFFFF" '
        'stroke-opacity="0.10"/>',
        f'<rect x="{_number(box.x)}" y="{_number(box.y)}" width="4" '
        f'height="{_number(box.height)}" rx="2" fill="{accent}"/>',
        f'<rect x="{_number(box.x + box.width - state_width - 12.0)}" '
        f'y="{_number(box.y + 12.0)}" width="{_number(state_width)}" height="20" '
        f'rx="{CORNER_RADIUS}" fill="none" stroke="{accent}"/>',
        f'<text x="{_number(box.x + box.width - state_width / 2.0 - 12.0)}" '
        f'y="{_number(box.y + 26.0)}" fill="{TEXT_PRIMARY}" '
        f'font-size="{STATE_SIZE}" text-anchor="middle">{state}</text>',
    ]

    label_x = box.x + 16.0
    label_y = box.y + 26.0
    pieces.append(
        f'<text x="{_number(label_x)}" y="{_number(label_y)}" '
        f'fill="{TEXT_PRIMARY}" font-size="{LABEL_SIZE}" font-weight="600">'
    )
    for index, line in enumerate(_label_lines(node.label)[:2]):
        x = _number(label_x)
        dy = "0" if index == 0 else "17"
        pieces.append(
            f'<tspan x="{x}" dy="{dy}">{_escape(line)}</tspan>'
        )
    pieces.append("</text>")

    cursor_y = box.y + 58.0
    thumbnail_uri = _thumbnail_uri(node.thumbnail)
    if thumbnail_uri is not None:
        thumb_size = min(THUMBNAIL_SIZE, box.width - 32.0)
        pieces.append(
            f'<image x="{_number(box.x + 16.0)}" y="{_number(cursor_y)}" '
            f'width="{_number(thumb_size)}" height="{_number(thumb_size)}" '
            f'preserveAspectRatio="xMidYMid meet" href="{thumbnail_uri}"/>'
        )
        cursor_y += thumb_size + 16.0

    for name, value in sorted(node.metrics.items())[:3]:
        pieces.append(
            f'<text x="{_number(box.x + 16.0)}" y="{_number(cursor_y)}" '
            f'fill="{TEXT_SECONDARY}" font-size="{METRIC_SIZE}">'
            f"{_escape(_metric_text(name, value))}</text>"
        )
        cursor_y += 16.0

    if node.progress is not None and node.progress[1] > 0:
        fraction = max(0.0, min(1.0, node.progress[0] / node.progress[1]))
        pieces.append(
            f'<line x1="{_number(box.x + 4.0)}" '
            f'y1="{_number(box.y + box.height - 2.0)}" '
            f'x2="{_number(box.x + 4.0 + (box.width - 8.0) * fraction)}" '
            f'y2="{_number(box.y + box.height - 2.0)}" '
            f'stroke="{accent}" stroke-width="2"/>'
        )
    pieces.append("</g>")
    return "".join(pieces)


def render_svg(graph: RunGraph) -> str:
    """Render *graph* as a standalone SVG string with editable text."""

    layout = layout_graph(graph)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{_number(layout.width)}" height="{_number(layout.height)}" '
        f'viewBox="0 0 {_number(layout.width)} {_number(layout.height)}" '
        f'role="img" aria-label="FlowView run {_escape(graph.run_id, quote=True)}">',
        "<defs>",
        f'<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
        f'refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" '
        f'fill="{TEXT_SECONDARY}"/></marker>',
        "</defs>",
        f'<rect width="100%" height="100%" fill="{CANVAS}"/>',
        f'<g font-family="{FONT_FAMILY}" font-variant-numeric="tabular-nums">',
    ]
    for edge in sorted(graph.edges, key=_edge_key):
        pieces.append(_render_edge(edge, layout, graph))
    node_order = sorted(
        graph.nodes,
        key=lambda node_id: (
            layout[node_id].layer,
            layout[node_id].order,
            node_id,
        ),
    )
    for node_id in node_order:
        pieces.append(_render_node(graph.nodes[node_id], layout[node_id]))
    pieces.extend(("</g>", "</svg>"))
    return "".join(pieces)


def _json_value(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _inspector(graph: RunGraph) -> str:
    sections: list[str] = []
    for node_id, node in sorted(graph.nodes.items()):
        duration = (
            node.ended_at - node.started_at
            if node.started_at is not None and node.ended_at is not None
            else None
        )
        fields: tuple[tuple[str, object], ...] = (
            ("Identifier", node.id),
            ("Kind", node.kind.value),
            ("State", state_label(node.state)),
            ("Started", node.started_at if node.started_at is not None else "—"),
            ("Ended", node.ended_at if node.ended_at is not None else "—"),
            ("Duration", duration if duration is not None else "—"),
            ("Progress", _json_value(node.progress) if node.progress is not None else "—"),
            ("Metrics", _json_value(node.metrics)),
            ("Parameters", _json_value(node.params)),
            ("Error", node.error if node.error is not None else "—"),
        )
        rows = "".join(
            f'<tr><th scope="row">{_escape(name)}</th><td><pre>{_escape(value)}</pre></td></tr>'
            for name, value in fields
        )
        sections.append(
            f'<section id="inspect-{_escape(node_id, quote=True)}">'
            f"<h2>{_escape(node.label)}</h2><table><tbody>{rows}</tbody></table></section>"
        )
    return "".join(sections)


def render_html(graph: RunGraph) -> str:
    """Render one self-contained HTML record containing SVG and inspector."""

    svg = render_svg(graph)
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>FlowView — {_escape(graph.run_id)}</title>"
        "<style>"
        f":root{{color-scheme:dark}}body{{margin:0;background:{CANVAS};color:{TEXT_PRIMARY};"
        f"font-family:{FONT_FAMILY}}}main{{padding:24px}}.canvas{{overflow:auto;"
        f"border:1px solid #FFFFFF1A;background:{CANVAS}}}svg{{display:block;max-width:none}}"
        "h1{font-size:20px}h2{font-size:16px;margin-top:28px}"
        "table{border-collapse:collapse;width:100%;table-layout:fixed}"
        "th,td{border-bottom:1px solid #FFFFFF1A;padding:7px 9px;text-align:left;vertical-align:top}"
        f"th{{width:130px;color:{TEXT_SECONDARY}}}pre{{margin:0;white-space:pre-wrap;"
        "overflow-wrap:anywhere;font:inherit;font-variant-numeric:tabular-nums}}"
        "</style></head><body><main>"
        f"<h1>FlowView run {_escape(graph.run_id)}</h1>"
        f"<p>spaCR {_escape(graph.spacr_version)} · settings {_escape(graph.settings_digest)}</p>"
        f'<div class="canvas">{svg}</div><div class="inspector">{_inspector(graph)}</div>'
        "</main></body></html>"
    )


def export(
    graph: RunGraph,
    path: str | os.PathLike[str],
    fmt: str = "svg",
) -> Path:
    """Write a deterministic SVG, HTML, or JSON representation of *graph*."""

    format_name = fmt.casefold().lstrip(".")
    if format_name == "svg":
        payload = render_svg(graph)
    elif format_name == "html":
        payload = render_html(graph)
    elif format_name == "json":
        payload = graph.to_json()
    else:
        raise ValueError("fmt must be 'svg', 'html', or 'json'")
    target = Path(path)
    target.write_bytes(payload.encode("utf-8"))
    return target


export_graph = export

__all__ = ["export", "export_graph", "render_html", "render_svg"]
