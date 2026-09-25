"""Launch the Cellpose Workbench training/apply recording in a fresh neutral stage.

Extracts the example ZIP into ``<stage>/Cellpose_training_images_masks`` as a
user would, gives the recording a private HOME that links only the stock
Cellpose-SAM weights, and runs ``capture_refresh.py --cellpose-training-gui``.
Run it through ``tools/gpu_turn.sh`` unless ``--stop-before-training``.
"""
import argparse
import os
from pathlib import Path
import zipfile

from capture_barcode_saved_plots import launch

CELLPOSE_WEIGHTS = Path('/home/olafsson/.cellpose/models')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True, help='New neutral directory, e.g. /tmp/spacr-train-cellpose')
    parser.add_argument('--example-zip', type=Path, required=True)
    parser.add_argument('--capture-name', default='train_cellpose_gui_run')
    parser.add_argument('--stop-before-training', action='store_true')
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    stage = args.stage.absolute()
    stage.mkdir(parents=True)
    with zipfile.ZipFile(args.example_zip) as bundle:
        if bundle.testzip():
            raise ValueError('The example ZIP failed its CRC check')
        bundle.extractall(stage / 'Cellpose_training_images_masks')
    home = stage / 'home'
    models = home / '.cellpose/models'
    models.mkdir(parents=True)
    for name in ('cpsam', 'cpsam_v2'):
        if (CELLPOSE_WEIGHTS / name).exists():
            (models / name).symlink_to(CELLPOSE_WEIGHTS / name)
    os.environ.update(HOME=str(home), PYTHONUNBUFFERED='1')
    os.environ.pop('CELLPOSE_LOCAL_MODELS_PATH', None)
    if args.stop_before_training:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    options = ['--cellpose-training-gui']
    if args.stop_before_training:
        options.append('--stop-before-training')
    print('STAGE', stage, flush=True)
    return launch('train_cellpose', args.capture_name, options, args.timeout, stage=stage)


if __name__ == '__main__':
    raise SystemExit(main())
