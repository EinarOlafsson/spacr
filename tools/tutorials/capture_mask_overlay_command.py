"""Record the actual explicit Mask plotting command on a private terminal."""
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

NAME = 'mask_explicit_overlay_command_opaque'


def terminal_driver(stage):
    capture = stage / 'captures' / NAME
    work = stage / 'mask_overlay_command_work_opaque'; work.mkdir(exist_ok=False)
    helper = Path(__file__).with_name('export_mask_overlays.py')
    shutil.copy2(helper, work / helper.name)
    source = stage / 'mask_fresh_v1/example_data/plate1/test/merged'
    destination = stage / 'mask_fresh_v1/overlay_api_recorded_opaque'
    commands = [
        ('01_explicit_numeric_inputs', ['python', '-c',
          'from pathlib import Path; import json; p=Path(' + repr(str(source)) + '); '
          'print("Merged numeric inputs:"); [print(f.name) for f in sorted(p.glob("*.npy"))]; '
          'print("Layout is metadata, NOT an image:"); print((p/".spacr_plane_layout.json").read_text()); '
          'print("The native overlay loop remains unfixed.")'], 'Layout is metadata, NOT an image:'),
        ('02_explicit_plotting_api', ['python', helper.name, '--source', str(source),
                                    '--destination', str(destination)], '"plots": 2'),
        ('03_verified_saved_overlays', ['python', '-c',
          'import json,csv; from pathlib import Path; p=Path(' + repr(str(destination)) + '); '
          'r=json.loads((p/"overlay_checks.json").read_text()); '
          'print("Actual saved plots and post-adjustment label counts:"); '
          '[print(Path(x["output"]).name,x["counts"]) for x in r["reports"]]; '
          'print("Channel RGB values checked:",sum(x["channel_rgb_values_checked"] for x in r["reports"])); '
          'print("Combined foreground pixels checked:",sum(x["combined_foreground_pixels_checked"] for x in r["reports"])); '
          'print("Source arrays unchanged:",r["original_inputs_unchanged"]); '
          'q=Path(' + repr(str(source.parent/'qc/segmentation_qc_pathogen.csv')) + '); '
          'rows=list(csv.DictReader(q.open())); '
          '[print("QC warning retained:",x["field"],x["flags"]) for x in rows if x["severity"]!="ok"]; '
          'print("Native plotting loop fixed:",r["native_plotting_loop_fixed"]); '
          'print("Segmentation accuracy certified:",r["segmentation_accuracy_certified"])'], 'Source arrays unchanged: True'),
    ]
    outcomes = []
    for scene, command, expected in commands:
        print('\033[2J\033[3J\033[H', end='', flush=True)
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        print(result.stdout, end='', flush=True)
        print(f'\nExit status: {result.returncode}', flush=True)
        passed = accepted_command(result, 0, expected)
        outcomes.append(dict(scene=scene, command=command, output=result.stdout,
                             returncode=result.returncode, accepted=passed))
        write(capture / 'commands.json', outcomes)
        write(capture / 'terminal_ready.json', dict(scene=scene, accepted=passed))
        input('\nPress Enter to continue the recording. ')
        if not passed: return 1
    proof = read(destination / 'overlay_checks.json')
    for path, digest in proof['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A plotted source changed after the command')
    if Path(proof['plotting_api_source']['path']) != REPO / 'spacr/plot.py':
        raise ValueError('The recorded API must come from this checkout')
    write(capture / 'scientific_acceptance.json', dict(accepted=True, overlay=proof,
          helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
          native_plotting_loop_fixed=False, segmentation_accuracy_certified=False,
          destination=str(destination), published=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--terminal-driver', action='store_true')
    args = parser.parse_args(); stage = args.stage.resolve()
    if args.terminal_driver: return terminal_driver(stage)
    if args.inside:
        return capture_terminal(stage, driver=Path(__file__).resolve(), capture_name=NAME,
              window_title='spaCR explicit Mask overlay export', expected_scenes=3,
              module='mask', pipeline_requested=False)
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
          env=env, timeout=480).returncode


if __name__ == '__main__':
    raise SystemExit(main())
