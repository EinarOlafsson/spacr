"""Per-field versus closed fixed_map pooled-normalisation latency, real preprocessing."""
import json, os, shutil, threading, time

import numpy as np
import pytest
import tifffile

from spacr import convert, core
from tests.test_watch_folder_and_analyse import MASK, _channels, real_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
WELLS = ('A01', 'A02', 'A03', 'A04', 'A05', 'A06')
INTERVAL = float(os.environ.get('POOL_INTERVAL', '3'))


@pytest.mark.parametrize('mode', ['per_field', 'fixed_map'])
def test_pool_latency(tmp_path, real_pipeline, mode):
    raw, converted = tmp_path / 'raw', tmp_path / 'converted'
    for index, well in enumerate(WELLS):
        (raw / well).mkdir(parents=True)
        for channel, image in enumerate(_channels(index * 7), start=1):
            tifffile.imwrite(raw / well / f'field01_C{channel}.tif', image)
    result = convert.convert_folder({'src': str(raw), 'dst': str(converted), 'preview_rows': 0})
    rows = result.rows()
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(converted / convert.MAP_FILENAME, watched / convert.MAP_FILENAME)
    arrivals = {}

    def arrive():
        for well in WELLS:
            for row in rows:
                if row['well'] == well:
                    shutil.copy2(converted / row['target'], watched / row['target'])
            arrivals[well] = time.time()
            time.sleep(INTERVAL)

    thread = threading.Thread(target=arrive)
    settings = dict(MASK, src=str(watched), batch_size=len(WELLS), keep_intermediate=True,
                    watch_folder=True, watch_pipeline='mask',
                    watch_normalization_pool=mode, watch_settle_seconds=0.5,
                    watch_poll_seconds=0.2, watch_idle_minutes=(len(WELLS) * INTERVAL + 5) / 60)
    thread.start()
    outcome = core._watch_folder_and_analyse(settings)
    thread.join()
    assert len(outcome['done']) == len(WELLS)
    ledger = json.load(open(watched / 'spacr_watch/watch_ledger.json'))
    report = {'mode': mode, 'interval_s': INTERVAL, 'fields': {}}
    for key, entry in sorted(ledger['fields'].items()):
        well = next(w for w in WELLS if f'_{w}_' in key)
        finished = entry.get('finished')
        if finished is None:
            cohort = ledger['normalization_cohorts'][entry['normalization_cohort']]
            finished = cohort.get('finished')
        report['fields'][well] = {
            'result_after_arrival_s': round(finished - arrivals[well], 3)}
    report['max_s'] = max(v['result_after_arrival_s'] for v in report['fields'].values())
    with open(os.path.join(HERE, f'pool_{mode}.json'), 'w') as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1))
