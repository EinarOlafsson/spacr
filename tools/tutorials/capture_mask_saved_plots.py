"""Show the explicitly regenerated Mask overlays in a real external viewer."""
import hashlib
from pathlib import Path

from capture_barcode_saved_plots import launch
from capture_saved_plots import show_saved_plots
from stage_lesson import read


def record(app, window, stage, captures, capture, settle, write_json):
    stage = Path(stage)
    command = read(stage / 'captures/mask_explicit_overlay_command_opaque/scientific_acceptance.json')
    if command.get('accepted') is not True or command.get('native_plotting_loop_fixed') is not False:
        raise ValueError('Expected the explicit recorded API workaround, not a repaired native loop')
    proof = command['overlay']
    for path, digest in proof['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A verified merged source changed')
    paths = []
    for row in proof['reports']:
        path = Path(row['output'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['png_sha256']:
            raise ValueError('A verified overlay PNG changed')
        paths.append(path)
    if len(paths) != 2:
        raise ValueError('The recorded example requires both field overlays')
    capture('02_actual_mask_before_external_viewer')
    viewer = show_saved_plots(app, window, stage, capture, settle, paths)
    write_json(captures / 'scientific_acceptance.json', dict(accepted=True,
        command=command, viewer=viewer, native_plotting_loop_fixed=False,
        segmentation_accuracy_certified=False, published=False))


if __name__ == '__main__':
    raise SystemExit(launch('mask', 'mask_explicit_overlay_viewer_opaque',
                           ['--mask-saved-plots', '--ai-controls'], 200))
