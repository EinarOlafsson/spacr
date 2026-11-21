"""Give array-based mask inputs the same memory limits as prepared batches."""

from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np

from . import _mask_batches as batches


def _tile_override(settings, key):
    """A per-path tile batch override, 0 when it is unset.

    :param settings: the run settings.
    :param key: the override setting, such as ``ops_ph_tile_batch``.
    :returns: the tile count, or 0 to use ``mask_eval_batch_size``.
    :raises ValueError: when the value is negative or not a whole number.
    """
    value = (settings or {}).get(key)
    if value in (None, ''):
        return 0
    try:
        tiles = int(value)
    except (TypeError, ValueError):
        tiles = -1
    if tiles < 0 or isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()):
        from .qt.i18n import tr

        raise ValueError(tr('{key} must be 0, which uses mask_eval_batch_size, or a '
                            'positive whole number of tiles, got {value}.',
                            key=key, value=repr(value)))
    return tiles


def _batch_limit(settings, image, remaining, gpu):
    """Resolve the shared field count before loading the rest of one batch.

    :param settings: shared field, tile and GPU memory settings.
    :param image: the first input, used to estimate the resident bytes.
    :param remaining: records still to process, including this input.
    :param gpu: whether the caller is using a CUDA model.
    :returns: the requested field count limited by available RAM and VRAM.
    """
    requested = batches._batch_size_request(settings.get('batch_size', 0))
    if requested == batches._AUTO_VRAM:
        return _auto_vram_limit(settings, image, remaining, gpu)
    limit = batches._largest_fitting_size(
        remaining, image.nbytes, image.nbytes * 6,
        free_vram=batches._free_vram_bytes() if gpu else None,
        free_ram=batches._available_ram_bytes(),
        margin_gb=float(settings.get('gpu_memory_margin_gb') or 1.5),
        tile_batch=batches._resolved_tiles(settings, 0),
        model_loaded=batches._model_resident())
    return min(requested or remaining, limit)


def _auto_vram_limit(settings, image, remaining, gpu):
    """The most fields of one ``'auto_vram'`` batch: calibrated, RAM-limited (item 1011).

    :param settings: shared settings holding ``_mask_calibration`` once calibrated.
    :param image: the first input, used for the RAM estimate.
    :param remaining: records still to process, including this input.
    :param gpu: whether the caller is using a CUDA model.
    :returns: a whole number from 1 to ``remaining``.
    """
    calibration = settings.get('_mask_calibration') or {}
    ram = batches._largest_fitting_size(
        remaining, 0, image.nbytes * 6, free_vram=None,
        free_ram=batches._available_ram_bytes(), margin_gb=0.0, tile_batch=0,
        model_loaded=True)
    limit = min(remaining, ram)
    if gpu and calibration.get('max_fields'):
        limit = min(limit, int(calibration['max_fields']))
    if not gpu or calibration.get('budget_source', 'cpu') == 'cpu':
        limit = min(limit, batches._LEGACY_BATCH_SIZE)
    return max(1, limit)


def _batch_plan(path, settings):
    """Read a persisted grouping so resumed fields retain their tile count.

    :param path: optional plan path beside the pipeline checkpoints.
    :param settings: shared field and tile settings identifying the grouping.
    :returns: plan path and mutable grouping record.
    :raises ValueError: when a saved plan uses different grouping settings.
    """
    path = None if path is None else Path(path)
    request = batches._batch_size_request(settings.get('batch_size', 0))
    basis = {'batch_size': request,
             'mask_eval_batch_size': batches._resolved_tiles(settings, 0)}
    if request == batches._AUTO_VRAM or batches._wants_auto_tiles(settings):
        basis['mask_batch_seed'] = batches._batch_seed(settings)
        basis['budget_source'] = (settings.get('_mask_calibration') or {}).get('budget_source', 'cpu')
    plan = {'format': 1, 'settings': basis, 'groups': []}
    if path is not None and path.is_file():
        plan = json.loads(path.read_text(encoding='utf-8'))
        if plan.get('format') != 1 or plan.get('settings') != basis:
            from .qt.i18n import tr

            raise ValueError(tr('{path} uses different mask batch settings. '
                                'Use the saved settings or move the mask checkpoint folder '
                                'away before starting a new segmentation run.', path=path))
    return path, plan


def _iter_array_batches(records, load, settings, gpu, plan_path, max_fields):
    """Yield input records from one owned resident batch at a time.

    The loader retains the pipeline-specific preparation and normalisation
    contract. Inputs are packed without changing values; each field is
    restored to its original shape and dtype before public model eval.
    Every role of every field finishes before the next batch is loaded.

    :param records: input identifiers, in processing order.
    :param load: callable returning ``(image or None, metadata)``.
    :param settings: shared field, tile, precision and memory settings.
    :param gpu: whether device staging is enabled.
    :param plan_path: optional persisted grouping beside mask checkpoints.
    :param max_fields: optional upstream staging capacity.
    :yields: ``(record, image or None, metadata, batch field count)``.
    """
    plan_path, plan = _batch_plan(plan_path, settings)
    known = {name: group for group in plan['groups'] for name in group['records']}
    position = 0
    even = None
    auto_vram = batches._batch_size_request(settings.get('batch_size', 0)) == batches._AUTO_VRAM
    while position < len(records):
        batches._wait_for_ram(0)
        record = records[position]
        image, metadata = load(record)
        position += 1
        if image is None:
            yield record, None, metadata, 1
            continue
        saved_group = known.get(str(record))
        if saved_group is None and auto_vram:
            if even is None:
                remaining = len(records) - position + 1
                even = batches._even_sizes(remaining, _batch_limit(settings, image, remaining, gpu))
            limit = min(even.pop(0) if even else 1, len(records) - position + 1)
            for step in range(limit - 1):
                if str(records[position + step]) in known:
                    limit = step + 1
                    break
        elif saved_group is None:
            limit = _batch_limit(settings, image, len(records) - position + 1, gpu)
            for step in range(limit - 1):
                if str(records[position + step]) in known:
                    limit = step + 1
                    break
        else:
            limit = 1
            while (position + limit - 1 < len(records)
                   and str(records[position + limit - 1]) in saved_group['records']):
                limit += 1
        if max_fields is not None:
            limit = min(limit, max(1, int(max_fields)))
        entries = [(record, image.shape, image.dtype, metadata, image.nbytes)]
        images = [image]
        for _ in range(limit - 1):
            batches._wait_for_ram(image.nbytes * 6)
            record = records[position]
            other, metadata = load(record)
            position += 1
            entries.append((record, None if other is None else other.shape,
                            None if other is None else other.dtype, metadata,
                            0 if other is None else other.nbytes))
            if other is not None:
                images.append(other)
            del other
        field_count = len(images) if saved_group is None else saved_group['fields']
        if saved_group is None and plan_path is not None:
            group = {'records': [str(entry[0]) for entry in entries], 'fields': field_count}
            plan['groups'].append(group)
            known.update((name, group) for name in group['records'])
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = plan_path.with_name(plan_path.name + '.part')
            temporary.write_text(json.dumps(plan, indent=1), encoding='utf-8')
            os.replace(temporary, plan_path)
        packed = np.concatenate([np.ascontiguousarray(value).view(np.uint8).reshape(-1)
                                 for value in images])
        images.clear()
        del image
        on_gpu = (batches._store_on_gpu(
            packed, margin_gb=float(settings.get('gpu_memory_margin_gb') or 1.5),
            tile_batch=batches._resolved_tiles(settings, field_count)) if gpu else None)
        resident = batches._ResidentBatch({}, on_gpu if on_gpu is not None else packed, None)
        del packed
        try:
            offset = 0
            for record, shape, dtype, metadata, size in entries:
                current = (None if shape is None else resident['data'][
                    offset:offset + size].view(dtype).reshape(shape))
                yield record, current, metadata, field_count
                del current
                offset += size
        finally:
            resident.free()


@contextmanager
def _array_batches(records, load, settings, *, gpu=False, plan_path=None,
                    max_fields=None):
    """Own an array batch until every role finishes, including on early exit.

    :param records: input identifiers; metadata alone may span many batches.
    :param load: the existing pipeline reader and preparation callable.
    :param settings: the shared mask settings.
    :param gpu: enable the existing GPU-fit check and resident device copy.
    :param plan_path: persist original group sizes for partial-batch resume.
    :param max_fields: cap loading by the capacity of an upstream staging queue.
    :yields: a field iterator; closing it frees the final resident batch.
    """
    iterator = _iter_array_batches(list(records), load, settings, gpu, plan_path, max_fields)
    try:
        yield iterator
    finally:
        iterator.close()
