"""Settings files written by the synthetic and large-site writers carry a version."""
import csv
import json

from spacr.settings import SETTINGS_VERSION_KEY, SETTINGS_SCHEMA_VERSION
from spacr.qt.synthetic import save_settings_csv


def test_synthetic_settings_csv_is_stamped(tmp_path):
    path = save_settings_csv(tmp_path / "s.csv", {"a": 1})
    rows = list(csv.reader(open(path, encoding="utf-8")))
    assert [SETTINGS_VERSION_KEY, str(SETTINGS_SCHEMA_VERSION)] in rows


def test_large_site_score_only_settings_json_is_stamped(tmp_path, monkeypatch):
    import spacr.ops_large_sites as ols
    monkeypatch.setattr(ols, "_agreement_rows", lambda *a, **k: [])
    monkeypatch.setattr(ols, "_write_agreement_report", lambda rows, dst, agree: [])
    monkeypatch.setattr(ols, "_saved_label_sites", lambda d: [])
    labels = tmp_path / "labels"
    labels.mkdir()
    settings = {"seg_dst": str(tmp_path / "out"), "seg_report_reference": "r"}
    ols._score_saved_labels(settings, {"ops_mask_agreement_labels": str(labels)})
    saved = json.load(open(tmp_path / "out" / "settings.json"))
    assert saved[SETTINGS_VERSION_KEY] == SETTINGS_SCHEMA_VERSION
