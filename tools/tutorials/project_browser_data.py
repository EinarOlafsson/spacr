"""Real registered and unregistered tutorial inputs; never invent project history."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from manager_data import prepare as prepare_registered
from capture_report import copy_private_source, snapshot_source, require_unchanged


def prepare(stage):
    inputs = prepare_registered(stage)
    original = Path(stage)/'batch_runs/real-channels-u0zjboui/converted_01'
    search = Path(inputs['root'])/'search'
    search.mkdir()
    copied = search/'converted_images'
    before, after = copy_private_source(original, copied)
    if (len(before) != 7 or sum(name.endswith('.tif') for name in before) != 4 or
            set(name for name in before if not name.endswith('.tif')) !=
            {'.spacr_conversion.checkpoint.json', 'conversion_map.csv', 'conversion_map.run_status.json'}):
        raise ValueError('Expected four actual channel images and all three conversion records, including the hidden checkpoint')
    inputs.update(conversion_original=str(original), conversion_copy=str(copied),
                  conversion_search=str(search), conversion_before=before, conversion_copied=after)
    return inputs


def verify_originals(inputs):
    from manager_data import verify_original
    registered = verify_original(inputs)
    require_unchanged(inputs['conversion_before'], snapshot_source(inputs['conversion_original']))
    require_unchanged(inputs['conversion_copied'], snapshot_source(inputs['conversion_copy']))
    return {'registered_original': registered, 'unregistered_original_and_copy_files': len(inputs['conversion_before']),
            'conversion_bytes_unchanged': True}


def verify_summary(summary, root, inventory, records):
    """Compare measured facts with independent disk bytes and original SQL rows."""
    known = bool(records)
    if (summary.root != str(root) or summary.name != Path(root).name or not summary.exists or
            summary.known != known or summary.staleness_known != known or
            summary.n_artifacts != len(records) or summary.n_files != len(inventory) or
            summary.size_bytes != sum(row['bytes'] for row in inventory.values())):
        raise ValueError('Project summary differs from independently measured identity, files or registry')
    if known:
        expected_ns = max(r['created_ns'] for r in records)
        expected_stamp = datetime.fromtimestamp(expected_ns/1e9, tz=timezone.utc).isoformat(timespec='seconds')
        if (summary.last_run_ns != expected_ns or summary.last_run_utc != expected_stamp or
                summary.last_run_source != 'registry'):
            raise ValueError('The recorded last-run timestamp differs from the real artifact registry')
    elif (summary.staleness_note() != 'unknown — nothing recorded' or summary.stale or summary.missing or
          summary.last_run_source != '' or summary.last_run_ns != 0 or summary.last_run_utc != '' or
          summary.unregistered_files != len(inventory) or
          summary.unregistered_bytes != summary.size_bytes):
        raise ValueError('This raw conversion has unknown provenance and no declared-output run timestamp')
    return {'root': str(root), 'files': summary.n_files, 'bytes': summary.size_bytes,
            'known': known, 'artifact_rows': summary.n_artifacts,
            'last_run_source': summary.last_run_source, 'last_run_utc': summary.last_run_utc,
            'stage_displayed_not_independently_rederived': summary.stage_label,
            'state_displayed': summary.staleness_note()}
