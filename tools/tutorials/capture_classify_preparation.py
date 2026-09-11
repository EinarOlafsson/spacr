"""Show the real metadata-preserving export command in a private terminal."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from capture_cli import accepted_command, capture_terminal
from stage_lesson import DEFAULT_STAGE, REPO, read, write

NAME = 'classify_canonical_preparation'


def terminal_driver(stage):
    from classify_split_evidence import inspect_inputs
    capture = stage / 'captures' / NAME
    work = stage / 'classify_preparation_work'
    work.mkdir(exist_ok=False)
    helper = Path(__file__).with_name('prepare_classify_split.py')
    shutil.copy2(helper, work / helper.name)
    destination = stage / 'classify_canonical_split_v2'
    source = stage / 'annotate_fresh/example_data/plate1'
    commands = [
        ('01_source_example', ['python', '-c',
          'import sqlite3; from pathlib import Path; p=Path(' + repr(str(source / 'measurements/measurements.db')) + '); '
          'c=sqlite3.connect(p.as_uri()+"?mode=ro",uri=True); '
          'print("Existing example labels by actual well:"); '
          '[print(r) for r in c.execute("SELECT plateID,rowID,columnID,infected,COUNT(*) FROM png_list GROUP BY 1,2,3,4 ORDER BY 1,2,3,4")]; c.close()'], 'Existing example labels'),
        ('02_prepare_existing_split', ['python', helper.name, '--source', str(source),
                                      '--destination', str(destination)], '"objects": 64'),
        ('03_exported_manifest', ['python', '-c',
          'import json; from pathlib import Path; p=Path(' + repr(str(destination)) + '); '
          'm=json.loads((p/"tutorial_input_manifest.json").read_text()); '
          'print("Actual training wells:",m["train_wells"]); print("Actual test wells:",m["test_wells"]); '
          'print("Selection:",m["selection"]); print("Example original:",m["records"][0]["source"]); '
          'print("Canonical copy:",m["records"][0]["target"]); '
          'print("Image bytes preserved; labels are not newly invented."); '
          'print("A deliberately balanced example does not preserve population prevalence."); '
          'print("Model started:",m["model_started"]); print("App filename parser fixed:",m["filename_parser_fixed"])'], 'Model started: False'),
    ]
    outcomes = []
    for scene, command, expected in commands:
        print('\033[2J\033[3J\033[H', end='', flush=True)
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=60)
        print(result.stdout, end='', flush=True)
        print(f'\nExit status: {result.returncode}', flush=True)
        passed = accepted_command(result, 0, expected)
        outcomes.append(dict(scene=scene, command=command, output=result.stdout,
                             returncode=result.returncode, accepted=passed))
        write(capture / 'commands.json', outcomes)
        write(capture / 'terminal_ready.json', dict(scene=scene, accepted=passed))
        input('\nPress Enter to continue the recording. ')
        if not passed:
            return 1
    proof = inspect_inputs(destination)
    write(capture / 'scientific_acceptance.json', dict(accepted=True, inputs=proof,
          helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
          destination=str(destination), preparation_only=True, model_started=False,
          filename_parser_fixed=False, published=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--terminal-driver', action='store_true')
    args = parser.parse_args(); stage = args.stage.resolve()
    if args.terminal_driver:
        return terminal_driver(stage)
    if args.inside:
        return capture_terminal(stage, driver=Path(__file__).resolve(), capture_name=NAME,
              window_title='spaCR explicit CV dataset preparation', expected_scenes=3,
              module='classify_merged', pipeline_requested=False)
    if (stage / 'captures' / NAME).exists():
        raise FileExistsError('Preserve the earlier terminal capture')
    env = dict(os.environ)
    for key, name in [('XDG_CONFIG_HOME','config'), ('XDG_DATA_HOME','data'),
                      ('XDG_CACHE_HOME','cache'), ('XDG_RUNTIME_DIR','runtime')]:
        path = stage / 'desktop' / NAME / name
        path.mkdir(parents=True, exist_ok=True, mode=0o700); env[key] = str(path)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1', QT_QPA_PLATFORM='xcb',
               PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'], PYTHONPATH=str(REPO),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return subprocess.run(['xvfb-run','-a','-s','-screen 0 3840x2160x24','dbus-run-session','--',
          sys.executable,str(Path(__file__).resolve()),'--stage',str(stage),'--inside'],
          env=env, timeout=300).returncode


if __name__ == '__main__':
    raise SystemExit(main())
