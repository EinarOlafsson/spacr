"""An empty barcode-regex field is not a broken expression.

The inline field validates on every keystroke, including the first one, and
the row is drawn before anybody has typed anything. Reporting "Regex error"
over an empty box tells somebody who has written nothing that what they wrote
is wrong.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_an_empty_field_asks_for_an_expression_rather_than_failing_it(blank):
    from spacr.qt.widgets.barcode_regex import evaluate_barcode_regex

    result = evaluate_barcode_regex(blank)

    assert result.valid is False
    assert result.message == "Enter a regular expression."
    assert result.captures == {}
