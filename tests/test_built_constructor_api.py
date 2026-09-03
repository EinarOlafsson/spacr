"""Keep constructor parameter documentation visible in generated API pages.

Sphinx AutoAPI can obtain constructor prose from either a class docstring or
``__init__``.  Source-level docstring checks cannot prove that either form
survives AutoAPI's class rendering, so this gate reads the fresh HTML tree
produced by the documentation workflow.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


BUILT = Path(__file__).resolve().parents[1] / "docs" / "_build" / "html"


def _class_parameters(page: Path, symbol: str) -> set[str]:
    """Return parameter names rendered in one class's introductory block."""
    source = page.read_text(encoding="utf-8")
    marker = f'id="{symbol}"'
    assert marker in source, f"{symbol} is absent from {page}"

    prologue = source.split(marker, 1)[1]
    prologue = re.split(
        r'<dl class="py (?:attribute|class|exception|function|method|property)">',
        prologue,
        maxsplit=1,
    )[0]
    field = re.search(
        r'<dt[^>]*>Parameters.*?</dt>\s*<dd[^>]*>(.*?)</dd>',
        prologue,
        flags=re.S,
    )
    assert field, f"{symbol} has no rendered Parameters field list"
    return set(re.findall(r"<strong>([A-Za-z_]\w*)</strong>", field.group(1)))


def test_parameter_reader_accepts_a_rendered_class_field_list(tmp_path):
    """The HTML reader must succeed on the shape emitted by Sphinx."""
    page = tmp_path / "index.html"
    page.write_text(
        '<dl class="py class"><dt id="pkg.Example">Example</dt><dd>'
        '<dl class="field-list simple"><dt>Parameters:</dt><dd>'
        '<ul><li><strong>first</strong> – one.</li>'
        '<li><strong>second_2</strong> – two.</li></ul></dd></dl>'
        '<dl class="py method"><dt id="pkg.Example.run">run</dt></dl>'
        "</dd></dl>",
        encoding="utf-8",
    )

    assert _class_parameters(page, "pkg.Example") == {"first", "second_2"}


def test_parameter_reader_rejects_a_class_without_rendered_parameters(tmp_path):
    """A class signature alone must not be mistaken for parameter prose."""
    page = tmp_path / "index.html"
    page.write_text(
        '<dl class="py class"><dt id="pkg.Example">'
        '<strong>signature_only</strong></dt><dd><p>Summary.</p>'
        '<dl class="py method"><dt id="pkg.Example.run">run</dt></dl>'
        "</dd></dl>",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="no rendered Parameters"):
        _class_parameters(page, "pkg.Example")


@pytest.mark.skipif(
    not os.environ.get("SPACR_DOCS_BUILT"),
    reason="requires the fresh HTML tree produced by the docs workflow",
)
def test_class_and_init_parameter_docs_reach_generated_class_pages():
    """Both supported constructor-docstring locations must reach the API."""
    assert _class_parameters(
        BUILT / "api" / "spacr" / "active_learning" / "index.html",
        "spacr.active_learning.StoppingVerdict",
    ) == {
        "stop",
        "reason",
        "gain",
        "labels_in_window",
        "window_from",
        "confident",
        "noise",
        "trend",
    }
    assert _class_parameters(
        BUILT / "api" / "spacr" / "flowview" / "feeder" / "index.html",
        "spacr.flowview.feeder.MultiprocessingFeeder",
    ) == {"source", "collector", "poll_interval", "max_event_bytes"}
