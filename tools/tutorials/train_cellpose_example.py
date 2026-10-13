"""Run a bounded, explicit API demonstration; the Train GUI remains unfixed.

This helper is for the six source-pinned tutorial cell pairs, not a recommended
training recipe. Its profiler observes the real Cellpose call without replacing
the model, training implementation, data or optimiser. Existing labels are not
independently reviewed ground truth; no held-out evaluation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def check_arrays(images, labels, expected):
    """Match every actual training pair to independently read source pixels."""
    if len(images) != len(expected) or len(labels) != len(expected):
        raise ValueError('The actual training call does not contain every pair')
    unused = set(expected)
    order = []
    for image, label in zip(images, labels):
        candidates = [name for name in unused
                      if image.shape == expected[name][0].shape
                      and label.shape == expected[name][1].shape
                      and np.array_equal(label, expected[name][1])
                      and np.allclose(image, expected[name][0], rtol=0, atol=1e-7)]
        if len(candidates) != 1:
            raise ValueError('Training image/mask pixels do not match one unique source pair')
        name = candidates[0]
        unused.remove(name)
        order.append(name)
    return order


def tensor_hash(value):
    array = value.detach().to(device='cpu').float().contiguous().numpy()
    if not np.isfinite(array).all():
        raise ValueError('A trainable parameter is nonfinite')
    return hashlib.sha256(array.tobytes()).hexdigest()


def changed_trainable(initial, state):
    """A changed non-trainable diameter or dtype alone is not training proof."""
    if not initial or not set(initial).issubset(state):
        raise ValueError('Checkpoint does not cover the initial trainable parameters')
    changed = [name for name, before in initial.items()
               if tensor_hash(state[name]) != before]
    if not changed:
        raise ValueError('No actual trainable weight changed')
    return changed


def run(source, destination):
    import tifffile
    import torch
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from cellpose import train as cp_train
    from spacr import submodules
    from spacr.settings import get_train_cellpose_default_settings

    source = Path(source).resolve()
    destination = Path(destination).absolute()
    if destination.is_relative_to(source):
        raise ValueError('Use a new destination outside the preserved source')
    manifest_path = source / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('image_channel') != 1 or manifest.get('mask_plane') != 4
            or manifest.get('crop_size') != 512 or len(manifest.get('pairs', [])) != 6):
        raise ValueError('The six verified cell-compartment pairs are required')
    originals = {str(manifest_path): digest(manifest_path)}
    expected = {}
    for row in manifest['pairs']:
        name = row['file']
        if Path(name).name != name or name in expected:
            raise ValueError('Training filenames must be unique basenames')
        paths = [source/'train'/role/name for role in ('images', 'masks')]
        for path, key in zip(paths, ('image_sha256', 'mask_sha256')):
            if path.is_symlink() or not path.is_file() or digest(path) != row[key]:
                raise ValueError('A verified source image or mask changed')
            originals[str(path)] = row[key]
        image, mask = [tifffile.imread(path) for path in paths]
        if (image.shape != (512, 512) or mask.shape != image.shape
                or image.dtype != np.uint16 or mask.dtype != np.uint16):
            raise ValueError('Expected paired uint16 512 by 512 source pixels')
        values = image.astype(np.float32)
        if values.max() > 1:
            values /= values.max()
        expected[name] = (values, mask)
    if not torch.cuda.is_available():
        raise ValueError('This bounded recorded example requires CUDA')
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    torch.set_num_threads(2)
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    destination.mkdir(parents=True)
    for role in ('images', 'masks'):
        folder = destination/'train'/role; folder.mkdir(parents=True)
        for name in expected:
            shutil.copy2(source/'train'/role/name, folder/name)
    settings = get_train_cellpose_default_settings(dict(
        src=str(destination), model_name='tutorial_cells_two_epoch_API_demo',
        n_epochs=2, batch_size=1, target_size=512, normalize=False,
        augment=False, learning_rate=1e-5, weight_decay=1e-5,
        max_train_images=None))
    save(destination/'requested_settings.json', settings)
    proof = dict(accepted=False, scope='Explicit training API and saved checkpoint mechanics only',
        source=str(source), destination=str(destination), source_hashes=originals,
        seed_at_entry=19, gpu=torch.cuda.get_device_name(0), gpu_memory_fraction=.65,
        settings=settings, gui_source_control_fixed=False,
        independent_annotation_review=False, held_out_accuracy_validated=False,
        application_source_modified=False, published=False,
        api_source=dict(path=str(Path(submodules.__file__).resolve()),
                        sha256=digest(submodules.__file__)),
        cellpose_training_source=dict(path=str(Path(cp_train.__file__).resolve()),
                                      sha256=digest(cp_train.__file__)))
    initial = {}; observations = []; before_figures = set(plt.get_fignums())
    original_profile = sys.getprofile()
    started = time.monotonic()

    def observe(frame, event, arg):
        if frame.f_code is not cp_train.train_seg.__code__:
            return
        values = frame.f_locals
        if event == 'call':
            order = check_arrays(values['train_data'], values['train_labels'], expected)
            initial.update({name: tensor_hash(parameter)
                            for name, parameter in values['net'].named_parameters()
                            if parameter.requires_grad})
            proof['actual_training_input'] = dict(order=order, pairs=len(order),
                image_and_label_values_checked=6*512*512*2,
                actual_device=str(values['net'].device),
                all_six_pairs_passed_to_cellpose=True,
                no_extra_spacr_augmentations=settings['augment'] is False,
                cellpose_internal_random_rotation_and_resize_still_enabled=True)
            plots = [plt.figure(n) for n in set(plt.get_fignums())-before_figures]
            if len(plots) != 1 or len(plots[0].axes) != 12:
                raise ValueError('The actual six-pair preview figure is missing')
            figure = plots[0]
            plotted_images = [np.asarray(axis.images[0].get_array()) for axis in figure.axes[:6]]
            plotted_masks = [np.asarray(axis.images[0].get_array()) for axis in figure.axes[6:]]
            if check_arrays(plotted_images, plotted_masks, expected) != order:
                raise ValueError('The real preview order differs from the training input')
            figure.savefig(destination/'actual_training_pairs.png', dpi=150,
                           transparent=False, facecolor='black')
            proof['actual_preview'] = dict(pairs=6, arrays_checked=12,
                path=str(destination/'actual_training_pairs.png'),
                sha256=digest(destination/'actual_training_pairs.png'))
            save(destination/'training_checks.json', proof)
            print('Verified actual API input: six cell pairs; no held-out accuracy claimed.', flush=True)
        elif event == 'return' and arg is not None:
            path, losses, test_losses = arg
            observations.append(dict(checkpoint=str(path), training_losses=np.asarray(losses).tolist(),
                reported_test_loss_slots=np.asarray(test_losses).tolist(),
                actual_nimg=int(values['nimg']), nimg_per_epoch=int(values['nimg_per_epoch']),
                learning_rates=np.asarray(values['LR'][:values['n_epochs']]).tolist(),
                internal_patch_size=int(values['bsize']),
                no_test_data=values['test_data'] is None and values['test_files'] is None))

    try:
        sys.setprofile(observe)
        submodules.train_cellpose(settings)
        sys.setprofile(original_profile)
        if len(observations) != 1:
            raise ValueError('The actual Cellpose training call did not return once')
        result = observations[0]
        losses = np.asarray(result['training_losses'])
        if (result['actual_nimg'] != 6 or result['nimg_per_epoch'] != 6
                or losses.shape != (2,) or not np.isfinite(losses).all()
                or not result['no_test_data'] or result['internal_patch_size'] != 256
                or result['learning_rates'][0] != 0 or result['learning_rates'][1] <= 0):
            raise ValueError('The bounded two-epoch training contract was not met')
        checkpoint = Path(result['checkpoint']).resolve()
        if not checkpoint.is_relative_to(destination) or not checkpoint.is_file():
            raise ValueError('The checkpoint was not saved inside the new destination')
        # This is our just-created checkpoint, not an arbitrary downloaded pickle.
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        changed = changed_trainable(initial, state)
        proof.update(result=result, checkpoint_sha256=digest(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size,
            trainable_parameters_checked=len(initial), changed_trainable_parameters=changed,
            saved_checkpoint_loads_with_weights_only=True,
            test_loss_zero_slots_are_not_test_results=True, accepted=True)
        print('Saved checkpoint has changed trainable weights; no held-out accuracy validated.', flush=True)
        return proof
    except Exception as error:
        proof['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        sys.setprofile(original_profile)
        proof['elapsed_seconds'] = time.monotonic()-started
        proof['original_inputs_preserved'] = all(digest(path)==value for path,value in originals.items())
        proof['private_inputs_preserved'] = all(
            digest(destination/'train'/role/name)==originals[str(source/'train'/role/name)]
            for role in ('images','masks') for name in expected)
        if not proof['original_inputs_preserved'] or not proof['private_inputs_preserved']:
            proof['accepted'] = False
        save(destination/'training_checks.json', proof)
        plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.destination)
    if not result['accepted']:
        raise SystemExit('Source preservation checks failed')
    print(json.dumps(dict(accepted=result['accepted'],
        changed_trainable_parameters=len(result['changed_trainable_parameters']),
        original_inputs_preserved=result['original_inputs_preserved'],
        gui_source_control_fixed=False, held_out_accuracy_validated=False), indent=2))
