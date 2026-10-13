"""Record an explicit training API workaround in an actual private terminal."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from capture_cli import accepted_command, capture_terminal
from stage_lesson import DEFAULT_STAGE, REPO, read, write

NAME = 'train_cellpose_explicit_api_v1'


def terminal_driver(stage):
    capture = stage/'captures'/NAME
    work = stage/'cellpose_training_command_v1'; work.mkdir(exist_ok=False)
    helper = Path(__file__).with_name('train_cellpose_example.py')
    shutil.copy2(helper, work/helper.name)
    source = stage/'derived/train_cellpose_corrected'
    destination = stage/'cellpose_training_runs/explicit_api_two_epoch_v1'
    commands = [
        ('01_verified_cell_pairs', ['python','-c',
          'import json; from pathlib import Path; p=Path('+repr(str(source))+'); '
          'm=json.loads((p/"source_manifest.json").read_text()); '
          'print("Explicit API route: the GUI source control remains unfixed."); '
          'print("Recorded cell intensity channel:",m["image_channel"],"cell mask plane:",m["mask_plane"]); '
          '[print(x["file"],x["source"],"existing labels:",x["objects"]) for x in m["pairs"]]; '
          'print("Six real 512 x 512 pairs; labels not independently reviewed."); '
          'print("Two epochs, minibatch 1, target size 512, learning rate 1e-5."); '
          'print("No held-out accuracy will be claimed.")'], 'Six real 512 x 512 pairs'),
        ('02_actual_two_epoch_api', ['python',helper.name,'--source',str(source),
                                    '--destination',str(destination)], '"accepted": true'),
        ('03_actual_checkpoint_checks', ['python','-c',
          'import json; from pathlib import Path; p=Path('+repr(str(destination))+'); '
          'r=json.loads((p/"training_checks.json").read_text()); '
          'print("Actual checkpoint:",r["result"]["checkpoint"]); '
          'print("SHA256:",r["checkpoint_sha256"]); '
          'print("Training losses, NOT held-out accuracy:",r["result"]["training_losses"]); '
          'print("Actual warm-up learning rates:",r["result"]["learning_rates"]); '
          'print("Actual internal training patch size:",r["result"]["internal_patch_size"]); '
          'print("Trainable parameters checked:",r["trainable_parameters_checked"]); '
          'print("Trainable parameters changed:",len(r["changed_trainable_parameters"])); '
          'print("All six pairs used:",r["result"]["actual_nimg"]==6); '
          'print("Original inputs preserved:",r["original_inputs_preserved"]); '
          'print("GUI source control fixed:",r["gui_source_control_fixed"]); '
          'print("Held-out accuracy validated:",r["held_out_accuracy_validated"]); '
          'print("Zero test-loss slots are placeholders; no test set was supplied.")'],
         'Original inputs preserved: True'),
    ]
    outcomes = []
    for scene, command, expected in commands:
        print('\033[2J\033[3J\033[H',end='',flush=True)
        print('$ '+shlex.join(command),flush=True)
        result = subprocess.run(command,cwd=work,text=True,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,timeout=480)
        print(result.stdout,end='',flush=True)
        print(f'\nExit status: {result.returncode}',flush=True)
        passed = accepted_command(result,0,expected)
        outcomes.append(dict(scene=scene,command=command,output=result.stdout,
                             returncode=result.returncode,accepted=passed))
        write(capture/'commands.json',outcomes)
        write(capture/'terminal_ready.json',dict(scene=scene,accepted=passed))
        input('\nPress Enter to continue the recording. ')
        if not passed:return 1
    proof=read(destination/'training_checks.json')
    if not proof['accepted'] or Path(proof['api_source']['path']) != REPO/'spacr/submodules.py':
        raise ValueError('The recorded training must use the real API from this checkout')
    write(capture/'scientific_acceptance.json',dict(accepted=True,training=proof,
          gui_source_control_fixed=False,held_out_accuracy_validated=False,published=False))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,default=DEFAULT_STAGE)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--terminal-driver',action='store_true')
    args=parser.parse_args();stage=args.stage.resolve()
    if args.terminal_driver:return terminal_driver(stage)
    if args.inside:
        return capture_terminal(stage,driver=Path(__file__).resolve(),capture_name=NAME,
              window_title='spaCR explicit Cellpose training API',expected_scenes=3,
              module='train_cellpose',pipeline_requested=True)
    if (stage/'captures'/NAME).exists():raise FileExistsError('Preserve the earlier recording')
    env=dict(os.environ)
    for key,name in [('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),
                     ('XDG_CACHE_HOME','cache'),('XDG_RUNTIME_DIR','runtime')]:
        path=stage/'desktop'/NAME/name;path.mkdir(parents=True,exist_ok=True,mode=0o700)
        env[key]=str(path)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1',GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory',GTK_USE_PORTAL='0',XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1',QT_QPA_PLATFORM='xcb',PYTHONPATH=str(REPO),
               PATH=str(Path(sys.executable).parent)+os.pathsep+env['PATH'],
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    return subprocess.run(['xvfb-run','-a','-s','-screen 0 3840x2160x24','dbus-run-session','--',
          sys.executable,str(Path(__file__).resolve()),'--stage',str(stage),'--inside'],
          env=env,timeout=600).returncode


if __name__=='__main__':raise SystemExit(main())
