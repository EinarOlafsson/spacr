"""A prior export is evidence of panel reuse only while its hash holds."""
from __future__ import annotations

import json
import os

from matplotlib.figure import Figure

from spacr import plot
from spacr.run_journal import hash_file


def test_replaced_prior_figure_does_not_claim_duplicate_pixels(tmp_path):
    previous = tmp_path / "previous.png"
    destination = tmp_path / "current.png"
    figure = Figure(figsize=(2, 2), dpi=50)
    axis = figure.add_subplot()
    line, = axis.plot([0, 1], [0, 1])
    figure.savefig(previous)

    panel = {"panel": 0, "displayed_sha256": "same-panel-pixels", "source": []}
    sidecar = tmp_path / (previous.name + plot._PROVENANCE_SUFFIX)
    sidecar.write_text(json.dumps({
        "schema": plot._PROVENANCE_SCHEMA,
        "panels": [panel],
        "figure_sha256": hash_file(previous, full=True),
    }), encoding="utf-8")
    assert [finding["prior_figure"] for finding in
            plot._prior_figure_findings([panel], destination)] == [previous.name]

    line.set_ydata([1, 0])
    figure.savefig(previous)
    before_sidecar = sidecar.stat().st_mtime_ns - 1_000_000_000
    os.utime(previous, ns=(before_sidecar, before_sidecar))
    assert previous.stat().st_mtime_ns < sidecar.stat().st_mtime_ns
    assert hash_file(previous, full=True) != json.loads(
        sidecar.read_text(encoding="utf-8"))["figure_sha256"]
    assert plot._prior_figure_findings([panel], destination) == []
