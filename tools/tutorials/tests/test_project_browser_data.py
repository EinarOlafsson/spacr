"""Project tutorial claims must agree with independently known files and records."""
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_browser_data import verify_summary


@pytest.fixture
def example():
    inventory = {'image.tif': {'bytes': 80, 'sha256': 'measured'}}
    stamp = 1700000000_000000000
    records = [{'created_ns': stamp}]
    summary = SimpleNamespace(root='/private/example', name='example', exists=True,
        known=True, staleness_known=True, n_artifacts=1, n_files=1, size_bytes=80,
        last_run_ns=stamp, last_run_utc=datetime.fromtimestamp(stamp/1e9, tz=timezone.utc).isoformat(timespec='seconds'),
        last_run_source='registry', stage_label='measure', stale=(), missing=(),
        unregistered_files=0, unregistered_bytes=0, staleness_note=lambda: 'current')
    return summary, inventory, records


@pytest.mark.parametrize('field,value', [('root','/other'),('name','other'),('exists',False),
    ('known',False),('staleness_known',False),('n_artifacts',2),('n_files',2),('size_bytes',81)])
def test_independent_identity_files_and_registry_facts_cannot_drift(example, field, value):
    summary, inventory, records = example
    assert verify_summary(summary, '/private/example', inventory, records)['bytes'] == 80
    setattr(summary, field, value)
    with pytest.raises(ValueError, match='independently measured'):
        verify_summary(summary, '/private/example', inventory, records)


@pytest.mark.parametrize('field,value', [('last_run_ns',1),('last_run_utc','wrong'),('last_run_source','filesystem')])
def test_registered_time_must_come_from_actual_registry(example, field, value):
    summary, inventory, records = example
    assert verify_summary(summary, '/private/example', inventory, records)['last_run_source'] == 'registry'
    setattr(summary, field, value)
    with pytest.raises(ValueError, match='recorded last-run timestamp'):
        verify_summary(summary, '/private/example', inventory, records)


@pytest.mark.parametrize('field,value', [('staleness_note',lambda:'current'),('stale',('invented',)),
    ('missing',('invented',)),('last_run_source','registry'),('last_run_ns',1),('last_run_utc','invented'),
    ('unregistered_files',0),('unregistered_bytes',0)])
def test_raw_conversion_is_unknown_not_proof_of_a_clean_or_missing_run(example, field, value):
    summary, inventory, _ = example
    summary.known = summary.staleness_known = False
    summary.n_artifacts = summary.last_run_ns = 0
    summary.last_run_source = summary.last_run_utc = ''
    summary.staleness_note = lambda: 'unknown — nothing recorded'
    summary.stage_label = 'nothing run'
    summary.unregistered_files = 1
    summary.unregistered_bytes = 80
    assert verify_summary(summary, '/private/example', inventory, [])['known'] is False
    setattr(summary, field, value)
    with pytest.raises(ValueError, match='unknown provenance'):
        verify_summary(summary, '/private/example', inventory, [])
