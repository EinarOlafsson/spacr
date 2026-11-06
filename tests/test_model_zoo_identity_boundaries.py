"""Catalogue identity and acquisition failures never manufacture a usable pin."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    payload = b'small local checkpoint fixture'
    raw = {'key': 'boundary-model', 'name': 'boundary.pt', 'kind': 'cellpose',
           'sha256': hashlib.sha256(payload).hexdigest(), 'version': '001'}
    real_fetch = zoo.fetch

    def forbidden(*args, **kwargs):
        pytest.fail('identity handling attempted provider or remote acquisition')

    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, 'catalogue', forbidden)
    monkeypatch.setattr(zoo, 'shared_catalogue', forbidden)
    monkeypatch.setattr(zoo, 'fetch', forbidden)
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '1')
    return raw, payload, real_fetch


def _settings(raw, requested):
    return {'cell_model_name': requested, 'model_metadata': {
        'schema_version': 1, 'settings': {'cell_model_name':
            zoo._model_reference_from_record(requested, raw)}}}


def test_capture_deduplicates_identical_declarations_without_changing_settings(catalogue, monkeypatch):
    raw, _, _ = catalogue
    supplied = {'cell_model_name': raw['key']}
    original = deepcopy(supplied)
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw, deepcopy(raw)])

    captured = zoo._capture_model_metadata('mask', supplied)

    assert captured == _settings(raw, raw['key'])
    captured['model_metadata']['settings']['cell_model_name']['key'] = 'changed copy'
    assert supplied == original


def test_capture_refuses_conflicting_identities_for_one_selected_key(catalogue, monkeypatch):
    raw, _, _ = catalogue
    supplied = {'cell_model_name': raw['key']}
    original = deepcopy(supplied)
    declarations = [raw, {**raw, 'sha256': '2' * 64}]
    before = deepcopy(declarations)
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', declarations)

    with pytest.raises(zoo.ModelZooError, match='one exact catalogue identity'):
        zoo._capture_model_metadata('mask', supplied)

    assert supplied == original
    assert declarations == before


@pytest.mark.parametrize('validator, field', [
    ('_validate_model_metadata', 'settings'),
    ('_model_reference_from_record', 'catalogue'),
    ('_match_model_reference', 'model_metadata'),
])
def test_identity_boundaries_refuse_nonmapping_inputs(catalogue, validator, field):
    supplied = ['not', 'a', 'mapping']
    original = deepcopy(supplied)
    call = getattr(zoo, validator)

    with pytest.raises(zoo.ModelZooError, match=field + '.*mapping'):
        if validator == '_model_reference_from_record':
            call('requested', supplied)
        else:
            call(supplied)

    assert supplied == original


@pytest.mark.parametrize('invalid_name', ['', 7])
def test_recorded_setting_names_must_be_nonempty_text(catalogue, invalid_name):
    raw, _, _ = catalogue
    supplied = _settings(raw, 'requested')
    records = supplied['model_metadata']['settings']
    records[invalid_name] = records.pop('cell_model_name')
    original = deepcopy(supplied)

    with pytest.raises(zoo.ModelZooError, match='setting key'):
        zoo._validate_model_metadata(supplied)

    assert supplied == original


def test_empty_identity_text_is_refused_without_repairing_it(catalogue):
    raw, _, _ = catalogue
    supplied = _settings(raw, 'requested')
    supplied['model_metadata']['settings']['cell_model_name']['sha256'] = ''
    original = deepcopy(supplied)

    with pytest.raises(zoo.ModelZooError, match='sha256.*string'):
        zoo._validate_model_metadata(supplied)

    assert supplied == original


def test_one_allowed_kind_as_text_preserves_the_exact_record(catalogue):
    raw, _, _ = catalogue
    supplied = _settings(raw, 'requested')
    original = deepcopy(supplied)

    validated = zoo._validate_model_metadata(supplied, model_kinds={'cell_model_name': 'cellpose'})

    assert validated == supplied['model_metadata']
    validated['settings'].clear()
    assert supplied == original


def test_numeric_catalogue_version_cannot_silently_become_text(catalogue):
    raw, _, _ = catalogue
    malformed = {**raw, 'version': 1}
    original = deepcopy(malformed)

    with pytest.raises(zoo.ModelZooError, match='catalogue.version.*string'):
        zoo._model_reference_from_record('requested', malformed)

    assert malformed == original


def test_acquired_file_disappearing_before_verification_never_reaches_a_consumer(catalogue, tmp_path, monkeypatch):
    raw, payload, _ = catalogue
    raw['uri'] = 'https://fixture.invalid/checkpoint'
    supplied = _settings(raw, str(tmp_path / 'missing.pt'))
    original = deepcopy(supplied)
    acquired = []

    def disappear(entry, destination):
        target = destination / entry.name
        target.write_bytes(payload)
        acquired.append(target)
        target.unlink()
        return target

    monkeypatch.setattr(zoo, 'fetch', disappear)
    with pytest.raises(zoo.ModelZooError, match=raw['sha256']):
        zoo._resolve_pinned_model_file(supplied, 'cell_model_name')

    assert len(acquired) == 1 and not acquired[0].exists()
    assert supplied == original
    assert not list((tmp_path / 'cache').iterdir())


def test_offline_resolution_deduplicates_cached_declarations_without_providers(catalogue, monkeypatch):
    raw, _, _ = catalogue
    entry = zoo._entry_from_mapping(raw)
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'off')
    monkeypatch.delenv(zoo.CATALOGUE_ENV_VAR, raising=False)
    monkeypatch.setitem(zoo._SHARED_CATALOGUE_CACHE, 'entries', [entry, entry])
    original = deepcopy(raw)

    resolved = zoo._resolution_model_entries(download=True)

    assert resolved == [entry]
    assert raw == original
    assert zoo._SHARED_CATALOGUE_CACHE['entries'] == [entry, entry]


def test_wrong_versioned_cache_bytes_are_preserved_during_verified_local_acquisition(catalogue, tmp_path, monkeypatch):
    raw, payload, real_fetch = catalogue
    mirror = tmp_path / 'mirror.pt'
    mirror.write_bytes(payload)
    raw['uri'] = mirror.as_uri()
    cache = tmp_path / 'cache'
    cache.mkdir()
    wrong = cache / 'boundary_v1.pt'
    wrong.write_bytes(b'unrelated previous bytes')
    monkeypatch.setattr(zoo, 'fetch', real_fetch)

    resolved = zoo._ensure_catalogue_model_file(zoo._entry_from_mapping(raw))

    assert resolved == cache / raw['name']
    assert resolved.read_bytes() == mirror.read_bytes() == payload
    assert wrong.read_bytes() == b'unrelated previous bytes'
    assert sorted(path.name for path in cache.iterdir()) == ['boundary.pt', 'boundary_v1.pt']


def test_acquisition_without_a_checksum_refuses_before_reading_a_source(catalogue, tmp_path, monkeypatch):
    raw, _, real_fetch = catalogue
    raw = {**raw, 'sha256': None, 'uri': 'https://fixture.invalid/checkpoint'}
    monkeypatch.setattr(zoo, 'fetch', real_fetch)
    monkeypatch.setattr(zoo, 'open_uri', lambda *a, **k: pytest.fail('unverified source was read'))

    with pytest.raises(zoo.ModelZooError, match='no sha256 was published'):
        zoo._ensure_catalogue_model_file(zoo._entry_from_mapping(raw))

    assert not list((tmp_path / 'cache').iterdir())


def test_picker_verification_reports_a_file_that_became_unreadable(catalogue, tmp_path, monkeypatch):
    raw, payload, _ = catalogue
    checkpoint = tmp_path / raw['name']
    checkpoint.write_bytes(payload)
    original_open = zoo.os.open
    attempts = []

    def unreadable(path, *args, **kwargs):
        if Path(path) == checkpoint:
            attempts.append(path)
            raise PermissionError('checkpoint read permission was revoked')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(zoo.os, 'open', unreadable)
    with pytest.raises(zoo.ModelZooError, match=raw['sha256']):
        zoo._verified_model_reference(zoo._entry_from_mapping(raw), str(checkpoint))

    assert attempts == [checkpoint]
    assert checkpoint.read_bytes() == payload


def test_lock_refresh_stops_when_the_owned_lock_disappears(tmp_path):
    import threading

    missing = tmp_path / 'released.lock'
    stop = threading.Event()
    worker = threading.Thread(target=zoo._refresh_model_download_lock,
                              args=(missing, stop, 0.001))
    worker.start()
    try:
        worker.join(timeout=1)
        assert not worker.is_alive(), 'refresh kept retrying a lock it no longer owns'
        assert not stop.is_set()
    finally:
        stop.set()
        worker.join(timeout=1)
    assert not missing.exists()


def test_missing_packaged_scorecard_preserves_existing_measurements(tmp_path, monkeypatch):
    from spacr import model_cards

    key = next(iter(zoo._CURATED_SCORECARD_BINDINGS))
    raw = next(row for row in zoo.BUNDLED_REMOTE_MODELS if row['key'] == key)
    original = deepcopy(raw)
    monkeypatch.setattr(model_cards, '_CARD_DIR', tmp_path)

    entry = zoo._builtin_entry_from_mapping(raw)

    assert entry.metrics == raw['metrics']
    assert not entry._curated_card
    assert raw == original
    assert list(tmp_path.iterdir()) == []


def test_a_missing_stock_score_is_not_rendered_as_zero_or_a_difference(monkeypatch):
    from spacr.qt import i18n

    monkeypatch.setattr(i18n, 'current_language', lambda: 'en')
    metrics = {key: {'finetuned': 0.25, 'vanilla': None, 'delta': None}
               for _, key in zoo._curated_scorecard_rows()}
    metrics.update(holdout='fixture fields', n_fields=2, n_objects=3,
                   scorecard_provenance={'boundary_tolerance_px': 2, 'stock_model': 'fixture stock'})
    original = deepcopy(metrics)

    lines = zoo._curated_scorecard_lines(metrics)

    assert len(lines) == 7
    assert lines[0] == 'F1 @ IoU 0.5: 0.2500'
    assert all('stock' not in line and 'difference' not in line for line in lines[:6])
    assert 'fixture fields; 2 fields, 3 objects' in lines[-1]
    assert metrics == original


def test_default_scorecard_export_uses_the_current_bundled_entries(catalogue):
    raw, _, _ = catalogue
    original = deepcopy(raw)

    rows = zoo.scorecard_export_rows()

    assert len(rows) == 1
    assert rows[0]['key'] == raw['key'] and rows[0]['name'] == raw['name']
    assert rows[0]['metric'] == '' and rows[0]['sha256'] == raw['sha256']
    assert rows[0]['this_model'] is None and rows[0]['stock'] is None
    assert raw == original


def test_missing_lock_has_no_owner_signature_and_is_not_recreated(tmp_path):
    missing = tmp_path / 'released.lock'

    assert zoo._model_lock_signature(missing) is None

    assert list(tmp_path.iterdir()) == []


def test_verified_scorecard_without_optional_counts_does_not_invent_them(tmp_path, monkeypatch):
    from spacr import model_cards

    key = next(iter(zoo._CURATED_SCORECARD_BINDINGS))
    binding = zoo._CURATED_SCORECARD_BINDINGS[key]
    raw = next(row for row in zoo.BUNDLED_REMOTE_MODELS if row['key'] == key)
    original = deepcopy(raw)
    card = json.loads((model_cards._CARD_DIR / binding['resource']).read_bytes())
    card['scores'].pop('counts', None)
    payload = json.dumps(card).encode('utf-8')
    fixture = tmp_path / binding['resource']
    fixture.write_bytes(payload)
    monkeypatch.setattr(model_cards, '_CARD_DIR', tmp_path)
    monkeypatch.setitem(zoo._CURATED_SCORECARD_BINDINGS, key,
                        {**binding, 'resource_sha256': hashlib.sha256(payload).hexdigest()})

    entry = zoo._builtin_entry_from_mapping(raw)

    assert entry._curated_card
    assert not {'true_positives', 'false_positives', 'false_negatives'} & entry.metrics.keys()
    expected = next(row for row in card['scores']['rows'] if row['key'] == 'object_f1')
    assert entry.metrics['f1'] == {'finetuned': expected['this'], 'vanilla': expected['stock'],
                                   'delta': expected['this'] - expected['stock']}
    assert entry.metrics['n_objects'] == card['scores']['holdout']['n_objects']
    assert entry.metrics['historical_scorecards'][0]['metrics'] == original['metrics']
    assert raw == original and fixture.read_bytes() == payload
