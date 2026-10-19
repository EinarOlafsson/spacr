"""Pipeline settings snapshots validate supplied pins before replacing outputs."""
import ast
from copy import deepcopy
import json

import pandas as pd
import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    from spacr import utils

    settings = {'src': str(tmp_path), 'base_model': '~origin/models/checkpoint.pt',
                'model_metadata': {'schema_version': 1, 'settings': {'base_model': {
                    'requested_value': '~origin/models/checkpoint.pt', 'key': 'private-fixture',
                    'filename': 'nested/checkpoint.pt', 'kind': 'cellpose',
                    'version': '001', 'version_status': 'declared', 'sha256': '1' * 64}}}}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [])
    outputs = tmp_path / 'settings'
    outputs.mkdir()
    csv = outputs / 'pipeline.csv'
    js = outputs / 'pipeline.json'
    csv.write_bytes(b'previous native CSV snapshot')
    js.write_bytes(b'previous native JSON snapshot')
    return utils, settings, csv, js


@pytest.mark.parametrize('problem', ['schema', 'binding', 'hash', 'filename', 'version', 'kind', 'null'])
def test_malformed_supplied_pin_preserves_both_existing_native_outputs(snapshot, problem):
    utils, settings, csv, js = snapshot
    record = settings['model_metadata']['settings']['base_model']
    if problem == 'schema':
        settings['model_metadata']['schema_version'] = True
    elif problem == 'binding':
        record['requested_value'] = 'different-original-request'
    elif problem == 'hash':
        record['sha256'] = 'not-a-checksum'
    elif problem == 'filename':
        record['filename'] = '.'
    elif problem == 'version':
        record['version_status'] = 'unspecified'
    elif problem == 'kind':
        record['kind'] = 'unknown-model-kind'
    else:
        settings['model_metadata'] = None
    original = deepcopy(settings)
    refused = False
    try:
        utils.save_settings(settings, name='pipeline')
    except zoo.ModelZooError:
        refused = True
    assert csv.read_bytes() == b'previous native CSV snapshot'
    assert js.read_bytes() == b'previous native JSON snapshot'
    assert refused
    assert settings == original


def test_valid_supplied_metadata_roundtrips_without_enrichment_or_model_io(snapshot, monkeypatch):
    utils, settings, csv, js = snapshot
    for name in ('fetch', 'catalogue', 'shared_catalogue', 'sha256_file', '_capture_model_metadata'):
        monkeypatch.setattr(zoo, name, lambda *a, **k: pytest.fail('snapshot performed model I/O/enrichment'))
    original = deepcopy(settings)
    utils.save_settings(settings, name='pipeline')
    assert json.loads(js.read_text(encoding="utf-8"))['model_metadata'] == original['model_metadata']
    rows = pd.read_csv(csv, dtype=str).set_index('Key')['Value']
    assert ast.literal_eval(rows['model_metadata']) == original['model_metadata']
    assert settings == original


def test_legacy_settings_without_model_metadata_never_call_pin_validator(snapshot, monkeypatch):
    utils, settings, csv, js = snapshot
    del settings['model_metadata']
    original = deepcopy(settings)
    monkeypatch.setattr(zoo, '_validate_model_metadata', lambda *a, **k: pytest.fail('legacy snapshot validated absent pins'))
    utils.save_settings(settings, name='pipeline')
    assert json.loads(js.read_text(encoding="utf-8")) == original
    assert 'model_metadata' not in pd.read_csv(csv)['Key'].tolist()
    assert settings == original
