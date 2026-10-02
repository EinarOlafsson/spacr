"""The Slurm wrapper makes both settings files visible before running Mask."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / 'packaging/apptainer/spacr_slurm.sh'


def _run_wrapper(tmp_path, *, separate=True, linked=False, missing=False, absolute=False):
    """Use a fake Apptainer to enforce explicit settings-directory visibility."""
    work = tmp_path / 'job'
    work.mkdir()
    plate = tmp_path / 'plate one'
    plate.mkdir()
    (work / 'plates.txt').write_text(str(plate) + '\n')
    mask_folder = tmp_path / 'mask settings'
    measure_folder = tmp_path / 'measure settings' if separate else mask_folder
    mask_folder.mkdir()
    if separate:
        measure_folder.mkdir()
    mask = mask_folder / 'mask.csv'
    measure = measure_folder / 'measure.csv'
    mask.write_text('Key,Value\ncell_channel,1\n')
    if not missing:
        measure.write_text('Key,Value\ncell_channel,1\n')
    mask_argument = os.path.relpath(mask, work)
    measure_argument = os.path.relpath(measure, work)
    if absolute:
        mask_argument, measure_argument = str(mask), str(measure)
    if linked:
        (work / 'mask-link.csv').symlink_to(mask)
        (work / 'measure-link.csv').symlink_to(measure)
        mask_argument, measure_argument = 'mask-link.csv', 'measure-link.csv'
    fake = tmp_path / 'bin'
    fake.mkdir()
    calls = tmp_path / 'calls.jsonl'
    executable = fake / 'apptainer'
    executable.write_text(
        '#!' + sys.executable + '\n'
        'import json, os, pathlib, sys\n'
        'args = sys.argv[1:]\n'
        'with open(os.environ["F586_CALLS"], "a") as stream:\n'
        '    stream.write(json.dumps(args) + "\\n")\n'
        'settings = pathlib.Path(args[args.index("--settings") + 1])\n'
        'binds = args[args.index("--bind") + 1].split(",")\n'
        'if not settings.is_absolute() or str(settings.resolve().parent) not in binds:\n'
        '    raise SystemExit("settings target is not explicitly visible in container")\n')
    executable.chmod(0o755)
    environment = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                       F586_CALLS=str(calls), MASK_SETTINGS=mask_argument,
                       MEASURE_SETTINGS=measure_argument, SIF='/unused/image.sif',
                       PLATES='plates.txt', SLURM_ARRAY_TASK_ID='0',
                       SLURM_CPUS_PER_TASK='2')
    result = subprocess.run(['bash', str(SCRIPT)], cwd=work, env=environment,
                            text=True, capture_output=True, check=False)
    records = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, records, mask, measure, plate


@pytest.mark.parametrize('separate,linked,absolute', [
    (True, False, False), (False, False, False), (True, True, False), (True, False, True),
])
def test_both_settings_directories_are_visible(tmp_path, separate, linked, absolute):
    result, calls, mask, measure, plate = _run_wrapper(
        tmp_path, separate=separate, linked=linked, absolute=absolute)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    for args, mode, settings in zip(calls, ('mask', 'measure'), (mask, measure)):
        assert args[:2] == ['run', '--nv']
        assert args[args.index('spacr-run') + 1] == mode
        assert args[args.index('--settings') + 1] == str(settings.resolve())
        binds = args[args.index('--bind') + 1].split(',')
        assert set(binds) == {str(plate), str(mask.parent), str(measure.parent)}
        assert 'n_jobs=2' in args and f'src={plate}' in args
    assert mask.read_text() == measure.read_text() == 'Key,Value\ncell_channel,1\n'


def test_missing_measure_settings_refuses_before_mask(tmp_path):
    result, calls, mask, measure, plate = _run_wrapper(tmp_path, missing=True)
    assert result.returncode != 0
    assert 'settings file not found:' in result.stderr
    assert calls == []
    assert not measure.exists()
    assert list(plate.iterdir()) == []
    assert mask.read_text() == 'Key,Value\ncell_channel,1\n'
