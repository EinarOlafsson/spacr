"""The QC dashboard's image-quality card says when screening was off and
when the saved report cannot be read, rather than reporting a clean run.

A report written with ``image_qc_mode = 'off'`` screened nothing, so the
card is 'missing' with a sentence saying screening is off. A report that is
not JSON, or lacks the fields the card reads, gives an 'error' card naming
what went wrong.
"""
import json

from spacr.image_quality import REPORT
from spacr.qt.widgets.qc_summary import _read_image_quality


def _write(root, payload):
    path = root / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_screening_that_was_off_is_reported_as_missing(tmp_path):
    _write(tmp_path, {"fields": [], "excluded_fields": [],
                      "policy": {"image_qc_mode": "off"}})
    card = _read_image_quality(str(tmp_path))
    assert card.verdict == "missing"
    assert card.headline == "Image-quality screening is off."
    assert card.detail[0].startswith("Policy: off.")


def test_a_report_that_is_not_json_is_an_error_card(tmp_path):
    _write(tmp_path, "{ not json")
    card = _read_image_quality(str(tmp_path))
    assert card.verdict == "error"
    assert card.headline.startswith("Could not read image-quality report")


def test_a_report_without_its_policy_is_an_error_card(tmp_path):
    _write(tmp_path, {"fields": [], "excluded_fields": []})
    card = _read_image_quality(str(tmp_path))
    assert card.verdict == "error"
    assert "policy" in card.headline


def test_a_project_without_a_report_has_no_card(tmp_path):
    assert _read_image_quality(str(tmp_path)) is None
