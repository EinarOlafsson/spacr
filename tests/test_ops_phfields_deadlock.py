"""Real spawned measure-only parity and bounded randomized staging without deadlocks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('pyarrow')

from spacr import ops_engine
from spacr.ops_run_infra import FieldStager
from tests.test_ops_phenotype_run_v2 import PLATE, SETTINGS, _placed, _write_tiles, models


_DRIVER = r'''
"""Run only the real measurement path and record actual spawned worker overlap."""
import json
import os
from pathlib import Path
import sys
import time
from spacr import ops_engine

_REAL_MEASURE = ops_engine._ph_measure_field


def overlapping_measure(task):
    """Hold each real worker until both processes have started, then measure unchanged pixels."""
    folder = Path(os.environ['SPACR_DEADLOCK_WORKER_MARKERS'])
    (folder / ('worker-' + str(os.getpid()))).touch()
    deadline = time.monotonic() + 30.
    while len(list(folder.glob('worker-*'))) < 2:
        if time.monotonic() >= deadline:
            raise RuntimeError('configured measurement workers did not overlap')
        time.sleep(.01)
    return _REAL_MEASURE(task)


def main():
    """Call the production field coordinator and write only diagnostic process metadata."""
    settings = json.loads(Path(sys.argv[1]).read_text())
    ops_engine._ph_measure_field = overlapping_measure
    files = ops_engine._index_tiles(settings['phenotype_source'], 'uncycled')['A1']
    report = ops_engine._ph_fields(str(Path(settings['dst_root']) / 'measurements.db'),
        settings['plate'], 'A1', files, settings, False, settings['dst_root'], segment=False, measure=True)
    payload = dict(report=report, driver=os.getpid(),
        workers=sorted(int(path.name.split('-')[1]) for path in Path(os.environ['SPACR_DEADLOCK_WORKER_MARKERS']).glob('worker-*')))
    Path(sys.argv[2]).write_text(json.dumps(payload, default=str))


if __name__ == '__main__':
    main()
'''


def _prepared(root):
    """Create real deterministic raw fields and segment them using the existing threshold fixture."""
    _write_tiles(root / 'tiles', sites=(0, 1, 2, 3))
    (root / 'run').mkdir()
    _placed(str(root / 'run' / 'measurements.db'), sites=(0, 1, 2, 3))
    settings = dict(SETTINGS, dst_root=str(root / 'run'), phenotype_source=str(root / 'tiles'),
                    ops_ph_measure_workers=2, randomize=True, mask_batch_seed=42)
    ops_engine.run_ops(settings, phases=('ph_segment',))
    return settings


def _measure_only(settings, root, *, staged):
    """Bound the entire driver and its own spawned process group by a sixty-second deadlock guard."""
    script = root / 'measure_driver.py'; script.write_text(_DRIVER)
    markers = root / 'workers'; markers.mkdir()
    configured = dict(settings)
    if staged:
        configured.update(ops_ph_stage_dir=str(root / 'stage'), ops_ph_stage_ahead=2, ops_ph_stage_threads=2)
    options = root / 'settings.json'; options.write_text(json.dumps(configured))
    output = root / 'report.json'
    environment = dict(os.environ, PYTHONPATH=str(Path(ops_engine.__file__).resolve().parents[1]),
                       PYTHONHASHSEED='0', SPACR_DEADLOCK_WORKER_MARKERS=str(markers))
    process = subprocess.Popen([sys.executable, str(script), str(options), str(output)],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate(timeout=10)
        pytest.fail('measure-only staging exceeded the sixty-second deadlock guard\n' + stdout)
    assert process.returncode == 0, stdout
    return json.loads(output.read_text())


@pytest.mark.parametrize('resume', [False, True], ids=['fresh', 'resumed'])
def test_staged_random_measure_only_finishes_in_spawn_workers_with_identical_tables(tmp_path, models, resume):
    """Seed42 stages 2,1,3,0 with two slots, so sorted0 cannot start; require completion under a sixty-second safety guard, two overlapping spawned workers and exact per-field/well Parquet bytes, also when one field is already committed."""
    unstaged = tmp_path / 'unstaged'; staged = tmp_path / 'staged'
    first = _prepared(unstaged); second = _prepared(staged)
    baseline = _measure_only(first, unstaged, staged=False)
    if resume:
        source = unstaged / 'run' / 'A1' / 'phenotype_cells'
        target = staged / 'run' / 'A1' / 'phenotype_cells'
        shutil.copytree(source, target, dirs_exist_ok=True)
        for site in (0, 2, 3):
            (target / f'site-{site}.parquet').unlink()
    result = _measure_only(second, staged, staged=True)
    for observed in (baseline, result):
        assert len(observed['workers']) == 2
        assert observed['driver'] not in observed['workers']
        assert observed['report']['ph_measure']['workers'] == 2
    assert result['report']['ph_measure']['measured'] == (3 if resume else 4)
    assert result['report']['ph_measure']['resumed'] == (1 if resume else 0)
    assert result['report']['ph_stream']['pipelined'] is False
    assert result['report']['ph_stream']['stage_peak_files'] <= 2
    for site in (0, 1, 2, 3):
        relative = Path('run/A1/phenotype_cells') / f'site-{site}.parquet'
        assert (staged / relative).read_bytes() == (unstaged / relative).read_bytes()
    first_table = Path(ops_engine._ph_cells_path(str(unstaged / 'run'), 'A1'))
    second_table = Path(ops_engine._ph_cells_path(str(staged / 'run'), 'A1'))
    assert second_table.read_bytes() == first_table.read_bytes()
    assert list((staged / 'stage' / PLATE / 'A1').iterdir()) == []


def test_stager_rejects_a_field_outside_its_queue_with_a_clear_error(tmp_path):
    """An invalid field must raise a named queue error immediately, with cleanup guaranteed."""
    def read(site):
        """Stage one tiny deterministic field through the production stager."""
        return np.full((2, 2, 1), site, dtype=np.float32), []

    stager = FieldStager(str(tmp_path / 'stage'), [2, 1, 3, 0], read, ahead=2, threads=2)
    try:
        with pytest.raises(ValueError, match='99.*staging queue'):
            stager.path(99)
    finally:
        stager.close()
