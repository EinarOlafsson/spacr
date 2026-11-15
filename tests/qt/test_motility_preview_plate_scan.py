"""What the motility preview does with a plate scan that comes back.

Only the newest scan may change the panel; a scan that failed puts its own
sentence on the status line; a plate whose groups hold no time series says
what a motility preview needs.  One general test walking a case table with
pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.motility_preview import MotilityPreviewPanel  # noqa: E402

pytestmark = pytest.mark.qt

_NO_SERIES = ("No (plate, well, field) group has two or more time points — "
              "a motility preview needs a time series.")


def test_a_scanned_plate_is_installed_only_when_current_and_usable(qtbot, subtests):
    """Stale scans change nothing; errors and group-less plates say exactly why."""
    cases = [
        ("stale scan", -1, {"merged": "/somewhere", "groups": {"g": [1, 2]}}, None),
        ("not a payload", 0, "not a payload", None),
        ("failed scan", 0, {"error": "Load failed: no merged folder"},
         "Load failed: no merged folder"),
        ("no time series", 0, {"merged": "/plate/merged", "groups": {}}, _NO_SERIES),
    ]
    for case_id, offset, payload, expected in cases:
        with subtests.test(msg=case_id):
            panel = MotilityPreviewPanel(threaded=False)
            qtbot.addWidget(panel)
            try:
                panel._load_token = 5
                before = panel._status.text()
                panel._on_plate_scanned(5 + offset, payload)
                assert panel._status.text() == (before if expected is None else expected)
            finally:
                panel.shutdown()
