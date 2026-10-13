#!/usr/bin/env python3
"""A bounded, real-pretrained Embeddings API example; never changes spaCR.

Input is a directory of existing spaCR crop PNGs. Output must be a NEW
directory. Filenames identify this demonstration's rows, not database keys.
No annotations, treatment labels, training, biological claims or GUI injection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_crops(folder, count=16):
    """Read a deterministic bounded sample without resizing or changing channels."""
    from spacr.crops import read_crop_png

    if not 4 <= count <= 64:
        raise ValueError('Choose 4 to 64 crops for this small demonstration')
    folder = Path(folder).resolve(strict=True)
    paths = sorted(folder.glob('*.png'))[:count]
    if len(paths) != count:
        raise ValueError(f'Need {count} PNGs directly in {folder}; found {len(paths)}')
    records, images = [], []
    for path in paths:
        before = digest(path)
        image = np.asarray(read_crop_png(str(path)))
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 32:
            raise ValueError(f'Expected a crop with 3 stored slots and size >=32: {path.name}')
        if not np.isfinite(image).all():
            raise ValueError(f'Non-finite crop pixels: {path.name}')
        records.append({'name': path.name, 'sha256': before, 'shape': list(image.shape),
                        'dtype': str(image.dtype)})
        images.append(image)
    if len({tuple(image.shape) for image in images}) != 1:
        raise ValueError('Crops must have equal shapes; no hidden resizing is performed')
    return np.stack(images), records


def verify_saved(output, values, names, identities):
    """Reopen every float and row/column identity, not just the matrix shape."""
    import pandas as pd

    output = Path(output)
    saved = np.load(output / 'vectors.npy', allow_pickle=False)
    np.testing.assert_array_equal(saved, values)
    frame = pd.read_csv(output / 'vectors.csv', float_precision='round_trip')
    if frame['object_id'].tolist() != identities or list(frame.columns[1:]) != list(names):
        raise ValueError('Saved object/feature identities differ from the encoded rows')
    np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy(dtype=np.float32), values)
    return {'matrix_cells_checked': int(values.size), 'ordered_identities_match': True,
            'npy_exact': True, 'csv_float32_exact': True}


def run(folder, output, *, count=16, policy='per_channel', device='cpu'):
    """Use embed_array's real encoder and verify the newly written artifacts."""
    from spacr import __version__
    from spacr.embeddings import EmbeddingSpec, embed_array, encoder_entry
    import torch

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing destination: {output}')
    folder = Path(folder).resolve(strict=True)
    crops, sources = load_crops(folder, count)
    torch.set_num_threads(2)
    spec = EmbeddingSpec(backbone='resnet18', channel_policy=policy,
                         batch_size=4, device=device)
    print(f'spaCR {__version__}; real pretrained resnet18; device={device}', flush=True)
    print(f'Input: {crops.shape}; policy={policy}; no crop resizing', flush=True)
    print('Slot indices describe saved PNG channels, NOT independently verified stain names.', flush=True)
    start = time.monotonic()
    result = embed_array(crops, spec)  # Deliberately no injected encoder.
    entry = encoder_entry(spec)
    if not entry.sha256:
        raise ValueError('Cannot document this example without a cached weights checksum')
    if not np.isfinite(result.values).all() or not np.any(np.std(result.values, axis=0) > 0):
        raise ValueError('Expected finite, nonconstant vectors for the actual images')
    identities = [record['name'] for record in sources]
    frame = result.to_frame(identities)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'vectors.npy', result.values, allow_pickle=False)
    frame.to_csv(output / 'vectors.csv', index=False)
    check = verify_saved(output, result.values, result.columns, identities)
    # PCA is a display of this tiny sample, not a trained classifier or a test.
    centered = result.values.astype(np.float64) - result.values.mean(axis=0, dtype=np.float64)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    np.save(output / 'pca_coordinates.npy', coordinates, allow_pickle=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if index < min(len(crops), 16):
            ax.imshow(crops[index])
            ax.set_title(f'Crop {index + 1}', fontsize=14)
    figure.suptitle('Real downloaded crop PNGs — stored display slots', fontsize=20)
    figure.savefig(output / 'input_crops.png', dpi=180)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], color='#367c9b', s=65)
    for index, (x, y) in enumerate(coordinates):
        ax.annotate(str(index + 1), (x, y), xytext=(5, 5), textcoords='offset points')
    ax.set(xlabel='Principal component 1', ylabel='Principal component 2',
           title=f'{count} crops, {policy}: exploratory PCA, NOT phenotype validation')
    figure.savefig(output / 'pca.png', dpi=180)
    plt.close(figure)
    for record in sources:
        if digest(folder / record['name']) != record['sha256']:
            raise ValueError('An original crop changed during the run')
    packages = {name: importlib.metadata.version(name)
                for name in ('numpy', 'pandas', 'torch', 'timm', 'Pillow')}
    report = {'accepted': True, 'version': __version__, 'source_folder': str(folder),
              'sources': sources, 'source_unchanged': True, 'shape': list(result.values.shape),
              'spec': asdict(spec), 'spec_fingerprint': spec.fingerprint(),
              'weights_sha256': entry.sha256, 'packages': packages,
              'helper_sha256': digest(__file__), 'elapsed_seconds': time.monotonic() - start,
              'verification': check, 'gui_workflow_completed': False,
              'biology_validated': False, 'stain_mapping_verified': False,
              'artifacts': {path.name: digest(path) for path in sorted(output.iterdir())}}
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Vectors: {result.values.shape}; saved and independently reopened', flush=True)
    print(f'Weights SHA256: {entry.sha256}', flush=True)
    print(frame.iloc[:3, :5].to_string(index=False), flush=True)
    print(f'Original crops unchanged. Outputs: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--policy', choices=('per_channel', 'project'), default='per_channel')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    args = parser.parse_args()
    run(args.crops, args.output, count=args.count, policy=args.policy, device=args.device)


if __name__ == '__main__':
    main()
