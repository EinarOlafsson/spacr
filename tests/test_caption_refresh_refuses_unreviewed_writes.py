"""Caption refresh refuses incomplete evidence before touching its fixture."""
import gzip
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_i18n_catalogs as builder
import i18n_caption_arrivals as arrivals


@pytest.mark.parametrize("language", arrivals.LANGUAGES)
@pytest.mark.parametrize("defect", ["missing_review", "source_hash"])
@pytest.mark.parametrize("change", ["arrival", "changed_body"])
def test_refresh_refuses_incomplete_locale_evidence_without_writing(
        tmp_path, monkeypatch, language, defect, change):
    baseline_source = "Object diameter"
    row = {"table": "SETTING_LABELS", "key": "diameter",
           "source": baseline_source, "source_sha256": arrivals._sha(baseline_source)}
    previous = {"schema": 1, "baseline": [row], "added": [],
                "retired": [], "changed": []}
    fixture = tmp_path / "arrivals.json.gz"
    fixture.write_bytes(gzip.compress(json.dumps(previous).encode(), mtime=0))
    original = fixture.read_bytes()
    signature = arrivals._sha(
        "SETTING_LABELS\0diameter\0" + row["source_sha256"])
    monkeypatch.setattr(arrivals, "PINNED_BASELINE_SOURCE_HASH_SHA256", signature)
    monkeypatch.setattr(arrivals, "ROOT", tmp_path)
    monkeypatch.setattr(arrivals, "FIXTURE", fixture)
    source = "Reviewed new caption." if change == "arrival" else "Model diameter"
    table, section, key = (("UI", "ui", source) if change == "arrival" else
                           ("SETTING_LABELS", "setting_labels", "diameter"))
    canonical = {"setting_labels": {"diameter": baseline_source},
                 "setting_tooltips": {}, "categories": (), "ui": (),
                 "module_summaries": {}}
    canonical[section] = (source,) if section == "ui" else {key: source}
    monkeypatch.setattr(builder, "canonical_sources", lambda: canonical)
    monkeypatch.setattr(builder, "reviewed_runtime_translations", lambda locale: {})
    reviews = tmp_path / "reviews"
    monkeypatch.setattr(builder, "REVIEWED_RUNTIME_DIR", reviews)
    catalogs = {}
    for locale in arrivals.LANGUAGES:
        target = "Reviewed target " + locale
        identity = (table, key)
        hashes = {identity: arrivals._sha(source)}
        if defect == "source_hash" and locale == language:
            hashes[identity] = "0" * 64
        catalogs[locale] = SimpleNamespace(SOURCE_HASHES=hashes, **{table: {key: target}})
        directory = reviews / locale
        directory.mkdir(parents=True)
        if defect == "missing_review" and locale == language:
            continue
        record = {"table": section, "key": key, "source": source,
                  "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "translation": target}
        (directory / "review.json").write_text(json.dumps(
            {"schema": 1, "language": locale, "records": [record]}), encoding="utf-8")
    original_import = importlib.import_module
    def load_catalog(name, *args, **kwargs):
        prefix = "spacr.qt.i18n_catalogs."
        if name.startswith(prefix):
            return catalogs[name.removeprefix(prefix)]
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", load_catalog)
    message = ("lacks explicit reviewed record" if defect == "missing_review"
               else "stale caption arrival catalog hash")
    with pytest.raises(AssertionError, match=message):
        arrivals.refresh_arrival_contract()
    assert fixture.read_bytes() == original
    assert arrivals.load_arrival_contract()["baseline"] == [row]


def test_refresh_refuses_a_rewritten_baseline_before_changing_the_file(tmp_path, monkeypatch):
    import i18n_caption_arrivals as arrivals

    candidate = arrivals.load_arrival_contract()
    entry = next(row for row in candidate['baseline'] if row['table'] == 'SETTING_LABELS')
    entry['source'] += ' silently repinned'
    entry['source_sha256'] = hashlib.sha256(entry['source'].encode()).hexdigest()
    fixture = tmp_path / 'arrivals.json.gz'
    fixture.write_bytes(gzip.compress(json.dumps(candidate).encode(), mtime=0))
    before = fixture.read_bytes()
    monkeypatch.setattr(arrivals, 'FIXTURE', fixture)
    with pytest.raises(AssertionError, match='historical caption bodies changed'):
        arrivals.refresh_arrival_contract()
    assert fixture.read_bytes() == before
