"""Independent pixel and predicted-class checks for the activation tutorial.

The reference differentiates selected logits directly, without calling spaCR's
saliency generator. It is software evidence, never a model-validation claim.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile

import numpy as np
from PIL import Image

from capture_database import _digest, file_bundle, require_unchanged_source, _readonly


def encode_gradient(gradient, method):
    """Encode absolute CHW derivatives into the expected saved PNG pixels."""
    a = np.asarray(gradient)
    if a.ndim != 3 or a.shape[0] != 3 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Expected finite nonnegative three-channel derivatives')
    if method not in ('saliency_image', 'saliency_channel'):
        raise ValueError('Only the two measured saliency methods are accepted')
    planes = [a.sum(axis=0)] if method == 'saliency_image' else list(a)
    encoded = []
    for plane in planes:
        low, high = plane.min(), plane.max()
        scaled = (plane-low)/(high-low) if high > low else np.zeros_like(plane)
        encoded.append((255*scaled).astype(np.uint8))
    return encoded[0] if method == 'saliency_image' else np.stack(encoded, axis=-1)


def check_pixels(actual, expected, *, tolerance=1):
    """Refuse wrong modes, shapes or values; report any rounding disagreement."""
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('Saved activation image has the wrong shape or pixel type')
    error = np.abs(a.astype(np.int16)-b.astype(np.int16))
    maximum = int(error.max())
    if maximum > tolerance:
        raise ValueError(f'Saved activation pixels disagree by {maximum}, allowed {tolerance}')
    return dict(pixels=int(a.size), max_absolute_error=maximum,
                differing_pixels=int(np.count_nonzero(error)), tolerance=tolerance)


def check_outputs(paths, expected_names, predictions):
    """Require one saved map for every source, under its predicted class."""
    names = [Path(p).name for p in paths]
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise ValueError('Saved maps duplicate, omit or substitute a source crop')
    for p in paths:
        if Path(p).parent.parent.parent.name != f'class_{predictions[Path(p).name]}':
            raise ValueError('Saved map is assigned to the wrong predicted class')


def check_database(records, crops, paths):
    """Verify original full cell identities, not just plausible map filenames."""
    source = {r['name']:r['identity'] for r in crops}
    if len(records) != len(source) or {r['png_path'] for r in records} != {str(p) for p in paths}:
        raise ValueError('Database map records disagree with the actual output files')
    for row in records:
        original = source.get(row['file_name'])
        if original is None or any(row[k] != original[k] for k in ('plateID','rowID','columnID','fieldID','prcfo')) or row['object'] != original['cell_id']:
            raise ValueError('Database map identity differs from its original cell')


def check_native_runs(runs):
    """Saved PNGs alone cannot certify two successful, visible GUI results."""
    if len(runs) != 2 or {r['method'] for r in runs} != {'saliency_image','saliency_channel'}:
        raise ValueError('Both distinct saliency runs must be recorded')
    if any(not r['outcome']['finished'] or not r['outcome']['ok'] or r['outcome']['errors'] or
           r['gui_figure_count'] < 1 or not r['figures_card_visible'] or r['settings_errors'] for r in runs):
        raise ValueError('Activation outputs exist but native figures or settings validation failed')


def prepare(stage):
    stage = Path(stage); base = stage/'annotate_fresh/example_data'
    database = base/'plate1/measurements/measurements.db'; bundle = file_bundle(database)
    if _digest(database) != '7b18161f0161d39b3ecedf92cfb0ccf9fee2328980da8e43167555a8f6fd27cd':
        raise ValueError('The previously downloaded tutorial database changed')
    parent = stage/'activation_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='real-crops-trained-model-', dir=parent))
    project = work/'project'; (project/'data').mkdir(parents=True)
    (project/'measurements').mkdir()
    with _readonly(database) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM png_list ORDER BY png_path LIMIT 4')]
    crops = []
    for row in rows:
        source = base/Path(row['png_path']).relative_to('/home/olafsson/.cache/spacr/example_data')
        if not source.resolve(strict=True).is_relative_to(base.resolve()):
            raise ValueError('Real crop escaped the downloaded dataset')
        crops.append(dict(source=str(source), name=source.name, sha256=_digest(source), identity=row))
    marker = Path(crops[0]['source']).parent/'.spacr_crop_format.json'
    fmt = json.loads(marker.read_text())
    if fmt.get('spacr_crop_format') != 3 or fmt.get('channel_order') != 'declared_rgb':
        raise ValueError('The current RGB crop format must be preserved explicitly')
    archive = project/'data/real_cell_crops.tar'
    with tarfile.open(archive, 'x') as stream:
        for record in crops: stream.add(record['source'], arcname=record['name'])
        stream.add(marker, arcname=marker.name)
    with tarfile.open(archive) as stream:
        for record in crops:
            if hashlib.sha256(stream.extractfile(record['name']).read()).hexdigest() != record['sha256']:
                raise ValueError('Archive does not contain the original downloaded crop bytes')
    source_model = base/'plate1/datasets/training_1/model/resnet18/rgb/epochs_1/resnet18_best_channels_rgb.pth'
    sources = {str(source_model): _digest(source_model), str(marker): _digest(marker)}
    model = project/source_model.name; shutil.copy2(source_model, model)
    card_source = source_model.with_suffix('.card.json'); sources[str(card_source)] = _digest(card_source)
    card = json.loads(card_source.read_text()); card_copy = model.with_suffix('.card.json')
    shutil.copy2(card_source, card_copy)
    if card['epochs'] != 1 or card['extra']['image_size'] != 128 or card['extra']['channels'] != ['r','g','b']:
        raise ValueError('The trusted locally generated demonstration model changed')
    for record in crops: sources[record['source']] = record['sha256']
    private = {str(p): _digest(p) for p in (archive, model, card_copy)}
    for source, target in ((source_model,model),(card_source,card_copy)):
        if _digest(source) != _digest(target): raise ValueError('Private checkpoint or card differs')
    return dict(project=str(project), archive=str(archive), model=str(model), crops=crops,
                original_hashes=sources, private_input_hashes=private, database=str(database),
                database_bundle=bundle, crop_format=fmt, model_classes=card['classes'],
                model_epochs=1, model_validation_claim=False, biological_claim=False,
                limitation='The earlier demonstration split shares wells; no independent accuracy claim.')


def reference(prepared, *, device='cpu'):
    """Check training input transforms, then differentiate on the named device."""
    import torch
    from spacr.torch_artifacts import load_model_artifact
    from spacr.io import _classification_transform, TarImageDataset
    transform = _classification_transform(128, [1,2,3], True)
    dataset = TarImageDataset(prepared['archive'], transform)
    if len(dataset) != 4 or dataset.crop_format != 3:
        raise ValueError('Archive metadata was miscounted as an image or channel format lost')
    inputs = []; names = []; maximum = 0.0
    for i, record in enumerate(prepared['crops']):
        with Image.open(record['source']) as image:
            pixels = np.asarray(image.convert('RGB'))
        height, width = pixels.shape[:2]
        top, left = round((height-128)/2), round((width-128)/2)
        independent = pixels[top:top+128,left:left+128].transpose(2,0,1).astype(np.float32)/255
        independent = (independent-np.float32(.5))/np.float32(.5)
        actual, name = dataset[i]
        if name != record['name'] or not np.array_equal(actual.numpy(), independent):
            raise ValueError('Training loader differs from independent RGB/crop/symmetric input')
        inputs.append(torch.from_numpy(independent)); names.append(name)
    # This is a trusted checkpoint generated by the earlier local tutorial run,
    # not an arbitrary downloaded pickle. No training or app generator is used.
    model, _ = load_model_artifact(prepared['model'], map_location=device)
    for module in model.modules():
        if hasattr(module, 'use_checkpoint'): module.use_checkpoint = False
    model.to(device).eval(); x = torch.stack(inputs).to(device).requires_grad_(True); logits = model(x)
    if logits.shape != (4,2): raise ValueError('Expected the recorded two-class model head')
    targets = logits.detach().argmax(1)
    score = sum(logits[i,int(targets[i])] for i in range(len(names)))
    gradient = torch.autograd.grad(score,x)[0].abs().detach().cpu().numpy()
    return dict(predictions={name:int(targets[i]) for i,name in enumerate(names)},
                gradients={name:gradient[i] for i,name in enumerate(names)},
                proof=dict(reference_device=str(next(model.parameters()).device), source_names=names, logits=logits.detach().tolist(),
                           prediction_indices=targets.tolist(), training_input_pixels=4*3*128*128,
                           training_input_max_error=maximum, autograd_target='predicted class logit',
                           original_channel_order_preserved=True))


def verify_saved(prepared, method, reference):
    archive = Path(prepared['archive']); root = archive.parent/archive.stem/method
    maps = sorted(p for p in root.rglob('*.png') if 'batch_grids' not in p.parts)
    check_outputs(maps, reference['predictions'], reference['predictions'])
    rows = []
    for path in maps:
        expected = encode_gradient(reference['gradients'][path.name], method)
        with Image.open(path) as image: check = check_pixels(np.asarray(image), expected)
        rows.append(dict(path=str(path), sha256=_digest(path), **check))
    dbpath = Path(prepared['project'])/'measurements'/f'{archive.stem}.db'
    with _readonly(dbpath) as db:
        db.row_factory = sqlite3.Row
        records = [dict(r) for r in db.execute(f'SELECT * FROM "{method}_list"')]
    check_database(records,prepared['crops'],maps)
    return dict(method=method, maps=rows, database=str(dbpath), database_rows=records,
                independent_saved_pixels_checked=sum(r['pixels'] for r in rows))


def preserved(prepared):
    require_unchanged_source(prepared['database'], prepared['database_bundle'])
    for key in ('original_hashes','private_input_hashes'):
        if any(not Path(p).is_file() or _digest(p) != sha for p,sha in prepared[key].items()):
            raise ValueError('An original or private tutorial input changed')
    return dict(original_database_and_sidecars_unchanged=True,
                original_crops_model_card_and_marker_unchanged=True,
                private_archive_model_and_card_unchanged=True)
