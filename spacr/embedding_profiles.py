"""Raw crop shards, pretrained-backbone embeddings, well embedding profiles and a well UMAP.

The computer-vision arm of the BBBC021 benchmark, from settings, in four
steps that :func:`run_embedding_profiles` runs in order:

1. :func:`write_crop_shard` cuts a fixed-size window around each chosen
   cell of a plate from its merged fields (:func:`spacr.crops.open_merged_field`),
   raw: never stretched per crop (item 772), optionally divided by the
   plate's saved flat field (``crop_illumination='gain_offset'``), with a
   bit-flag mask (1 this cell, 2 any nucleus, 4 another cell, 8 outside
   the field). At most ``crop_cap_per_well`` cells per well are drawn by
   ``numpy.random.default_rng(crop_seed)``, well by well in sorted order.
   One HDF5 shard per plate holds ``images`` and ``masks``, beside a
   ``.cells.parquet`` with one row per crop;
2. :func:`embed_crop_shard` encodes the masked crops with a pretrained
   backbone (:mod:`spacr.embeddings`), each channel alone, under one fixed
   scale per plate and channel (the 99th percentile of a 2,048-crop
   sample, item 410);
3. :func:`aggregate_embedding_wells` writes ``wells_median.parquet``,
   ``wells_mean.parquet`` and the control cells' percentiles in the layout
   of the feature aggregate (:func:`spacr.screen_staging.aggregate_wells`),
   so :func:`spacr.sp_stats.moa_evaluation` scores them unchanged;
4. :func:`well_umap` writes a UMAP (McInnes et al. 2018) of the wells.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Key the settings of this module are registered under.
APP_KEY = 'embedding_profiles'

#: The crop stores :func:`write_crop_shard` writes.
CROP_STORES = ('h5_shards',)

#: How a crop window is corrected for illumination.
CROP_ILLUMINATIONS = ('none', 'gain_offset')

#: The bits of a crop's mask.
MASK_BITS = {1: 'this cell', 2: 'any nucleus', 4: 'another cell', 8: 'outside the field'}

#: The embedding input normalisations: 'none' feeds the [0, 1] crops as
#: they are; 'imagenet' resizes and standardises them as the backbone was
#: trained (:class:`spacr.embeddings.EmbeddingSpec`, ``input_transform``).
INPUT_NORMALISATIONS = ('none', 'imagenet')

_DEFAULTS: Dict[str, Any] = {
    'embedding_src': '',
    'embedding_cells': '',
    'embedding_wells': '',
    'crop_store': 'h5_shards',
    'crop_window': 128,
    'crop_channels': '0,1,2',
    'crop_cap_per_well': 300,
    'crop_mask_flags': True,
    'crop_illumination': 'none',
    'crop_illumination_model': '',
    'crop_seed': 0,
    'embedding_backbone': 'resnet18',
    'embedding_channel_mode': 'per_channel',
    'embedding_channel_scale': 'estimate',
    'embedding_input_normalisation': 'none',
    'embedding_weights': '',
    'embedding_masked': True,
    'embedding_batch_size': 64,
    'embedding_aggregate': 'well_median,well_mean',
    'embedding_control': 'DMSO',
    'embedding_control_column': 'moa',
    'embedding_umap': False,
    'umap_metric': 'cosine',
    'umap_neighbors': 15,
    'umap_min_dist': 0.3,
    'umap_seed': 0,
    'umap_percentiles': True,
    'embedding_out': '',
}

_EXPECTED_TYPES: Dict[str, Any] = {key: type(value) for key, value in _DEFAULTS.items()}

_TOOLTIPS: Dict[str, str] = {
    'embedding_src': "(str) - The plate folder: merged/ holds its fields and illumination/illumination_model.npz its saved flat field. Default blank.",
    'embedding_cells': "(str) - Per-cell table of the cells to crop, such as the screen run's <plate>.parquet (plate, well, site, Metadata_cell_id). Default blank.",
    'embedding_wells': "(str) - Optional table of the wells to crop, with plate, well and any labels (such as moa) to carry; blank crops every well of the cell table. Default blank.",
    'crop_store': "(str) - How the crops are stored: 'h5_shards', one HDF5 shard per plate with images and masks and a .cells.parquet index. Default 'h5_shards'.",
    'crop_window': "(int) - Side of the square window around each cell's centroid, in pixels; past the field edge it is zero padded. Default 128.",
    'crop_channels': "(str) - Comma-separated intensity planes of the merged field to crop, in order, such as 0,1,2. Each plane becomes one channel of the crop and, in per_channel mode, its own block of the embedding, so more planes give a wider embedding and a longer run. Default '0,1,2'.",
    'crop_cap_per_well': "(int) - Most cells cropped per well, drawn at random with crop_seed; 0 crops every cell. Default 300.",
    'crop_mask_flags': "(bool) - Store a bit-flag mask with each crop: 1 this cell, 2 any nucleus, 4 another cell, 8 outside the field. Default True.",
    'crop_illumination': "(str) - 'gain_offset' divides each window by the plate's saved flat field (offset subtracted) before it is stored as uint16; 'none' stores it raw. Never a per-crop stretch. Default 'none'.",
    'crop_illumination_model': "(str) - Path to the saved illumination model (.npz) used when crop_illumination is 'gain_offset'. Blank takes <src>/illumination/illumination_model.npz. With crop_illumination 'none' this path is not read at all. Default blank.",
    'crop_seed': "(int) - Seed of the random draw of cells in each well when crop_cap_per_well limits them. The same seed picks the same cells again; another seed picks a different sample. It has no effect when every cell is cropped. Default 0.",
    'embedding_backbone': "(str) - Pretrained encoder that turns each crop into a vector, such as resnet18 (timm, ImageNet weights) or openphenom. A different backbone gives different features and a different vector width, so profiles from two backbones cannot be mixed in one table. Default 'resnet18'.",
    'embedding_channel_mode': "(str) - 'per_channel' encodes each channel alone and concatenates the vectors; 'project' encodes them together. Default 'per_channel'.",
    'embedding_channel_scale': "(str) - 'estimate' divides each channel by its 99th percentile over a 2,048-crop sample of the plate, clipped to [0, 1]; 'none' feeds the values as they are. Default 'estimate'.",
    'embedding_input_normalisation': "(str) - 'none' feeds the [0, 1] crops at their own size; 'imagenet' resizes them to the backbone's training size and standardises by its mean and standard deviation. Default 'none'.",
    'embedding_weights': "(str) - A local weights file for the backbone; blank uses the backbone's own cached weights. Default blank.",
    'embedding_masked': "(bool) - Multiply each crop by its cell's own mask, so background and neighbours are 0. Default True.",
    'embedding_batch_size': "(int) - Crops sent through the backbone in one forward pass. Higher is faster on a GPU but needs more memory; lower fits a small GPU or the CPU and runs slower. Default 64.",
    'embedding_aggregate': "(str) - Comma-separated well tables of the embeddings: well_median (wells_median.parquet) and well_mean (wells_mean.parquet). The median resists outlier cells, the mean follows every cell. The UMAP reads the median table, so keep well_median when embedding_umap is on. Default 'well_median,well_mean'.",
    'embedding_control': "(str) - Value of embedding_control_column that marks control cells; their percentiles are written per plate for Ljosa scaling. Default 'DMSO'.",
    'embedding_control_column': "(str) - Column of the cell table that marks the control cells, compared with embedding_control. When the column is missing no control percentiles are written, and the UMAP then uses the well medians without scaling. Default 'moa'.",
    'embedding_umap': "(bool) - Also writes a two-dimensional UMAP of the median well profiles to well_umap.csv, for plotting. On needs the umap-learn package and adds run time; off skips the map and leaves the well tables unchanged. Default False.",
    'umap_metric': "(str) - Distance UMAP uses to find each well's neighbours; any metric umap-learn accepts works. Profiles are scaled to unit length first, so 'euclidean' ranks neighbours the same way as 'cosine', while 'correlation' or 'manhattan' give a different map. Default 'cosine'.",
    'umap_neighbors': "(int) - Number of nearest wells UMAP looks at around each well. Higher keeps more of the global layout between groups of wells but blurs small clusters; lower keeps fine local structure and can split the map into islands. Default 15.",
    'umap_min_dist': "(float) - Smallest distance UMAP allows between points in the map. Lower packs similar wells into tight clumps; higher spreads them out and shows the overall layout more evenly. Default 0.3.",
    'umap_seed': "(int) - Random state of the UMAP, run on one thread so the same seed and wells give the same map. Another seed can rotate or rearrange the map while keeping similar neighbourhoods. Default 0.",
    'umap_percentiles': "(bool) - Scale each plate's wells by the control cells' percentiles (Ljosa) before the UMAP; else the medians as they are. Default True.",
    'embedding_out': "(str) - Folder of the shards, embeddings, well tables and UMAP. Blank writes to <src>/embeddings. Default blank.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Embedding crops': ['embedding_src', 'embedding_cells', 'embedding_wells', 'crop_store',
                        'crop_window', 'crop_channels', 'crop_cap_per_well', 'crop_mask_flags',
                        'crop_illumination', 'crop_illumination_model', 'crop_seed',
                        'embedding_out'],
    'Embedding profiles': ['embedding_backbone', 'embedding_channel_mode',
                           'embedding_channel_scale', 'embedding_input_normalisation',
                           'embedding_weights', 'embedding_masked', 'embedding_batch_size',
                           'embedding_aggregate', 'embedding_control',
                           'embedding_control_column', 'embedding_umap', 'umap_metric',
                           'umap_neighbors', 'umap_min_dist', 'umap_seed', 'umap_percentiles'],
}

_DESCRIPTION = (
    "Cut raw single-cell crops into shards, embed them with a pretrained "
    "backbone, aggregate the embeddings per well and draw a UMAP of the wells."
)


def embedding_profiles_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_embedding_profiles`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every key of this module.
    """
    merged = dict(settings or {})
    for key, value in _DEFAULTS.items():
        merged.setdefault(key, value)
    return merged


def _register() -> None:
    """Register this module's settings through the defaults seam, once per process."""
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, embedding_profiles_defaults, expected_types=_EXPECTED_TYPES,
                      tooltips=_TOOLTIPS, categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def _ints(value: Any) -> List[int]:
    """A list of integers from '0,1,2' or a sequence."""
    if isinstance(value, str):
        return [int(part) for part in value.split(',') if part.strip()]
    return [int(part) for part in value]


def select_crop_cells(cells: pd.DataFrame, *, cap_per_well: int = 300,
                      seed: int = 0) -> pd.DataFrame:
    """The cells to crop: at most ``cap_per_well`` per well, drawn without replacement.

    Wells are visited in sorted order and one generator,
    ``numpy.random.default_rng(seed)``, draws for every well holding more
    than the cap, so the same table and seed always give the same cells.
    The result is sorted by well, site and cell and carries
    ``n_cells_in_well``, the cells the well had before the draw.

    :param cells: one plate's cells with ``well``, ``site`` and
        ``Metadata_cell_id``.
    :param cap_per_well: the cap; ``0`` keeps every cell.
    :param seed: the generator's seed.
    :returns: the chosen cells, index reset.
    :raises ValueError: when a needed column is missing.
    """
    missing = [c for c in ('well', 'site', 'Metadata_cell_id') if c not in cells.columns]
    if missing:
        raise ValueError(f'the cell table lacks {missing}')
    available = cells.groupby('well').size()
    rng = np.random.default_rng(seed)
    parts = []
    for _, part in cells.groupby('well', sort=True):
        if cap_per_well and len(part) > cap_per_well:
            part = part.iloc[np.sort(rng.choice(len(part), cap_per_well, replace=False))]
        parts.append(part)
    chosen = pd.concat(parts).sort_values(['well', 'site', 'Metadata_cell_id']).reset_index(drop=True)
    chosen['n_cells_in_well'] = chosen['well'].map(available).astype(np.int32)
    return chosen


def _field_key(plate: str) -> str:
    """The plate name as spaCR writes it into merged field names."""
    from .schema import escape_filename_component

    return escape_filename_component(plate)


def write_crop_shard(merged: Any, cells: pd.DataFrame, out: Any, *, plate: str,
                     size: int = 128, channels: Sequence[int] = (0, 1, 2),
                     illumination: Optional[str] = None, mask_flags: bool = True,
                     only_wells: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Cut one plate's chosen cells into an HDF5 shard of raw crops.

    For each cell the window of side ``size`` centred on its rounded
    centroid in the ``cell`` label plane is read from
    ``merged/<plate>_<well>_<site>_1.npy`` (zero padded past the edge). With
    ``illumination`` the plate's flat field is applied as Measure applies
    it: ``rint((window - offset) * gain)``, clipped to uint16 and the
    clipped pixels counted. ``mask_flags`` adds the bit mask of
    :data:`MASK_BITS`. Writes ``<out>/<plate>.h5`` (``images`` uint16
    ``(n, size, size, channels)`` and ``masks`` uint8, gzip 4, chunks of
    32), ``<plate>.cells.parquet`` (the cells in shard order with
    ``row``, centroid, bounding box, mask area, clipped pixels and
    ``truncated``) and ``<plate>.json``.

    :param merged: the plate's ``merged/`` folder.
    :param cells: the chosen cells (:func:`select_crop_cells`).
    :param out: the output folder.
    :param plate: the plate name.
    :param size: the window side, in pixels.
    :param channels: the intensity planes to crop.
    :param illumination: path of the saved illumination model, or ``None``
        for raw windows.
    :param mask_flags: write the mask dataset.
    :param only_wells: crop these wells alone (the draw is unchanged).
    :returns: the shard record: ``plate``, ``n_crops``, ``n_wells``,
        ``truncated_frac``, ``clipped_px``, ``seconds`` and the paths.
    :raises ImportError: when h5py is missing.
    """
    try:
        import h5py
    except ImportError as missing:
        raise ImportError('writing crop shards needs h5py: pip install "spacr[embeddings]"') from missing

    from .crops import open_merged_field

    start = time.time()
    if only_wells is not None:
        cells = cells[cells['well'].isin(list(only_wells))].reset_index(drop=True)
    channels = [int(c) for c in channels]
    key = _field_key(plate)
    gain = dark = None
    if illumination:
        from .illumination import load_illumination_model

        field = load_illumination_model(os.fspath(illumination)).field_for(key)
        gain = field.gain_stack(channels)
        dark = np.broadcast_to(field._offset_stack(channels), gain.shape)
    n, half, k = len(cells), size // 2, len(channels)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / f'{plate}.h5.tmp'
    meta = np.zeros((n, 7), dtype=np.float64)
    with h5py.File(pending, 'w') as handle:
        images = handle.create_dataset('images', (n, size, size, k), dtype='uint16',
                                       chunks=(min(32, max(n, 1)), size, size, k),
                                       compression='gzip', compression_opts=4, shuffle=True)
        masks = handle.create_dataset('masks', (n, size, size), dtype='uint8',
                                      chunks=(min(32, max(n, 1)), size, size),
                                      compression='gzip', compression_opts=4,
                                      shuffle=True) if mask_flags else None
        row = 0
        for (well, site), part in cells.groupby(['well', 'site'], sort=False):
            fld = open_merged_field(os.path.join(os.fspath(merged), f'{key}_{well}_{site}_1.npy'),
                                    use_cache=False)
            height, width, _ = fld.shape
            index = fld.label_index('cell')
            for label in part['Metadata_cell_id'].to_numpy():
                cy, cx = index.centroid(int(label))
                by0, by1, bx0, bx1 = index.bbox(int(label))
                y0, x0 = int(round(cy)) - half, int(round(cx)) - half
                y1, x1 = y0 + size, x0 + size
                window = fld.read_window(y0, y1, x0, x1, tuple(channels), np.float32)
                sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, height), max(x0, 0), min(x1, width)
                if gain is not None:
                    g = np.ones((size, size, k), np.float32)
                    d = np.zeros((size, size, k), np.float32)
                    g[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gain[sy0:sy1, sx0:sx1]
                    d[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = dark[sy0:sy1, sx0:sx1]
                    window = np.rint((window - d) * g)
                clipped = int((window > 65535).sum())
                cell_mask = fld.read_mask_window('cell', y0, y1, x0, x1)
                bits = (cell_mask == label).astype(np.uint8)
                if mask_flags:
                    nucleus = fld.read_mask_window('nucleus', y0, y1, x0, x1)
                    bits |= (nucleus > 0).astype(np.uint8) << 1
                    bits |= ((cell_mask > 0) & (cell_mask != label)).astype(np.uint8) << 2
                    inside = np.zeros((size, size), bool)
                    inside[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = True
                    bits |= (~inside).astype(np.uint8) << 3
                    masks[row] = bits
                images[row] = np.clip(window, 0, 65535).astype(np.uint16)
                area = index.area(int(label))
                meta[row] = (cy, cx, by1 - by0, bx1 - bx0, area, clipped,
                             float(int((bits & 1).sum()) < area))
                row += 1
        handle.attrs['plate'] = plate
        handle.attrs['size'] = size
        handle.attrs['channels'] = json.dumps(channels)
        handle.attrs['mask_bits'] = json.dumps({str(b): t for b, t in MASK_BITS.items()})
        handle.attrs['illumination'] = 'gain_offset' if gain is not None else 'none'
    shard = folder / f'{plate}.h5'
    os.replace(pending, shard)
    table = cells.assign(row=np.arange(n), centroid_y=meta[:, 0], centroid_x=meta[:, 1],
                         bbox_h=meta[:, 2].astype(np.int32), bbox_w=meta[:, 3].astype(np.int32),
                         mask_area=meta[:, 4].astype(np.int32),
                         clipped_px=meta[:, 5].astype(np.int32), truncated=meta[:, 6].astype(bool))
    from .tabular import write_table

    index_path = write_table(table, folder / f'{plate}.cells.parquet', canonicalise=False)
    info = {'plate': plate, 'n_crops': int(n), 'n_wells': int(table['well'].nunique()),
            'truncated_frac': float(table['truncated'].mean()) if n else 0.0,
            'clipped_px': int(table['clipped_px'].sum()), 'seconds': round(time.time() - start, 1),
            'shard': str(shard), 'cells': index_path}
    (folder / f'{plate}.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    return info


class ShardCrops:
    """The crops of one shard as ``float32`` ``(n, h, w, c)``, optionally masked to the cell.

    Indexable by a slice or a sorted integer array, which is what
    :func:`spacr.embeddings._estimate_channel_scale` and
    :func:`spacr.embeddings._embed_plate` read.

    :param path: the ``.h5`` shard.
    :param masked: multiply by bit 1 of the mask (this cell).
    """

    def __init__(self, path: Any, masked: bool = True) -> None:
        """Open the shard read-only."""
        import h5py

        self.handle = h5py.File(os.fspath(path), 'r')
        self.images = self.handle['images']
        self.masks = self.handle['masks'] if 'masks' in self.handle else None
        if masked and self.masks is None:
            raise ValueError(f'{path} has no masks to mask the crops with')
        self.masked = masked
        self.shape = tuple(self.images.shape)

    def __len__(self) -> int:
        """The number of crops."""
        return self.shape[0]

    def __getitem__(self, index: Any) -> np.ndarray:
        """The crops at ``index`` (a slice or sorted integers), as float32."""
        if not isinstance(index, slice):
            index = np.asarray(index)
        values = self.images[index].astype(np.float32)
        if self.masked:
            values *= (self.masks[index] & 1)[..., None]
        return values

    def close(self) -> None:
        """Close the shard."""
        self.handle.close()


def embedding_spec(settings: Mapping[str, Any]) -> Any:
    """The :class:`spacr.embeddings.EmbeddingSpec` the settings describe.

    :param settings: the ``embedding_*`` keys of
        :func:`embedding_profiles_defaults`.
    :returns: the spec, on the CPU when torch sees no GPU.
    :raises ValueError: for an unknown input normalisation or channel scale.
    """
    from .embeddings import EmbeddingSpec

    settings = embedding_profiles_defaults(settings)
    norm = str(settings['embedding_input_normalisation']).strip().lower()
    if norm not in INPUT_NORMALISATIONS:
        raise ValueError(f'embedding_input_normalisation must be one of {INPUT_NORMALISATIONS}')
    scale = str(settings['embedding_channel_scale']).strip().lower()
    if scale not in ('estimate', 'none'):
        raise ValueError("embedding_channel_scale must be 'estimate' or 'none'")
    return EmbeddingSpec(backbone=str(settings['embedding_backbone']),
                         channel_policy=str(settings['embedding_channel_mode']),
                         channels=tuple(_ints(settings['crop_channels'])) and tuple(
                             range(len(_ints(settings['crop_channels'])))),
                         batch_size=int(settings['embedding_batch_size']),
                         normalize=scale == 'estimate',
                         checkpoint_path=settings['embedding_weights'] or None,
                         input_transform='pretrained' if norm == 'imagenet' else 'none')


def embed_crop_shard(shard: Any, spec: Any, out: Any, *, masked: bool = True,
                     encoder: Any = None, chunk: int = 1024) -> Dict[str, Any]:
    """Embed one shard's crops and write ``<out>/<plate>.npy`` and ``<plate>.json``.

    The plate's channel scale is estimated once from the shard
    (:func:`spacr.embeddings._estimate_channel_scale`) and recorded; the
    crops are then embedded ``chunk`` at a time under it
    (:func:`spacr.embeddings._embed_plate`).

    :param shard: the ``.h5`` shard of :func:`write_crop_shard`.
    :param spec: an :class:`spacr.embeddings.EmbeddingSpec`.
    :param out: the output folder.
    :param masked: mask each crop to its cell.
    :param encoder: a ready encoder; ``None`` builds the spec's backbone.
    :param chunk: crops embedded per call.
    :returns: the record: ``plate``, ``n``, ``dims``, ``fingerprint``,
        ``channel_scale``, ``columns_first``/``columns_last``, ``seconds``
        and ``path``.
    """
    from .embeddings import _backbone_encoder, _embed_plate, _estimate_channel_scale

    shard = Path(shard)
    plate = shard.name[:-len('.h5')]
    crops = ShardCrops(shard, masked=masked)
    try:
        start = time.time()
        encoder = encoder or _backbone_encoder(spec)
        record: Dict[str, Any] = {}
        n = len(crops)
        if spec.normalize and spec.channel_scale is None:
            scale = _estimate_channel_scale(crops)
            record['channel_scale'] = {str(c): v for c, v in enumerate(scale)}
            record['channel_scale_sample'] = dict(percentile=99.0, size=int(min(n, 2048)),
                                                  seed=0, n_crops=int(n))
        parts, columns, fingerprint = [], [], ''
        for first in range(0, n, chunk):
            result = _embed_plate(crops[first:min(n, first + chunk)], spec, encoder=encoder,
                                  record=record)
            parts.append(result.values.astype(np.float32))
            columns, fingerprint = list(result.columns), result.spec.fingerprint()
    finally:
        crops.close()
    values = np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{plate}.npy'
    np.save(path, values)
    info = {'plate': plate, 'backbone': spec.backbone, 'n': int(n), 'dims': int(values.shape[1]),
            'masked': bool(masked), 'fingerprint': fingerprint,
            'channel_scale': record.get('channel_scale'),
            'channel_scale_sample': record.get('channel_scale_sample'),
            'columns_first': columns[0] if columns else '', 'columns_last': columns[-1] if columns else '',
            'seconds': round(time.time() - start, 1), 'path': str(path)}
    (folder / f'{plate}.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    return info


def embedding_cells(values: np.ndarray, cells: pd.DataFrame,
                    keep: Sequence[str] = ('plate', 'well')) -> pd.DataFrame:
    """One row per crop: the ``keep`` columns of the shard index beside ``emb_0000``...

    :param values: the crops' embeddings, rows in shard order.
    :param cells: the shard's ``.cells.parquet``.
    :param keep: the index columns carried.
    :returns: the per-cell embedding table.
    :raises ValueError: when the two disagree on the number of crops.
    """
    if len(values) != len(cells):
        raise ValueError(f'{len(values)} embeddings for {len(cells)} crops')
    columns = [f'emb_{i:04d}' for i in range(values.shape[1])]
    return pd.concat([cells[[c for c in keep if c in cells.columns]].reset_index(drop=True),
                      pd.DataFrame(values, columns=columns)], axis=1)


def aggregate_embedding_wells(tables: Sequence[pd.DataFrame], dst: Any, *,
                              levels: Any = ('well_median', 'well_mean'),
                              control_column: str = 'moa', control: str = 'DMSO') -> Dict[str, Any]:
    """Aggregate per-cell embeddings into well tables, as the feature aggregate does.

    Each plate's table (:func:`embedding_cells`) is written to a temporary
    per-plate Parquet file and aggregated by
    :func:`spacr.screen_staging.aggregate_wells` over its ``emb_`` columns:
    ``wells_median.parquet`` / ``wells_mean.parquet`` (with ``n_cells``)
    and, from the cells whose ``control_column`` is ``control``,
    ``<control>_percentiles.parquet`` at the 1st and 99th percentile.

    :param tables: one per-cell embedding table per plate.
    :param dst: the output folder.
    :param levels: ``'well_median'`` and/or ``'well_mean'``.
    :param control_column: the column that marks control cells.
    :param control: the control's value.
    :returns: the :func:`spacr.screen_staging.aggregate_wells` record.
    """
    import tempfile

    from .screen_staging import aggregate_wells

    folder = Path(dst)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        files, controls = [], set()
        for number, table in enumerate(tables):
            if control_column in table.columns:
                chosen = table.loc[table[control_column].astype(str) == str(control),
                                   ['plate', 'well']]
                controls |= set(map(tuple, chosen.astype(str).values))
            path = Path(scratch) / f'{number:05d}.parquet'
            table.drop(columns=[c for c in (control_column,) if c in table.columns]).to_parquet(
                path, index=False)
            files.append(path)
        return aggregate_wells(files, folder, levels=levels, controls=controls or None,
                               control_name=control, prefixes=('emb_',))


def well_umap(wells: pd.DataFrame, *, features: Optional[Sequence[str]] = None,
              metric: str = 'cosine', n_neighbors: int = 15, min_dist: float = 0.3,
              seed: int = 0) -> pd.DataFrame:
    """A two-dimensional UMAP of well profiles (McInnes et al. 2018), L2-normalised first.

    :param wells: one row per well; its non-feature columns are carried.
    :param features: the profile columns; ``None`` takes the ``emb_`` ones.
    :param metric: the UMAP distance.
    :param n_neighbors: the UMAP neighbours.
    :param min_dist: the UMAP minimum distance.
    :param seed: the UMAP random state (one thread, so it is reproducible).
    :returns: the carried columns with ``x`` and ``y``; ``attrs['umap_version']``
        names the umap-learn release.
    :raises ImportError: when umap-learn is missing.
    """
    import umap

    features = list(features) if features is not None else [
        c for c in wells.columns if str(c).startswith('emb_')]
    values = wells[features].to_numpy(float)
    values = values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-9)
    points = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist), metric=metric,
                       random_state=int(seed), n_jobs=1).fit_transform(values)
    carried = wells[[c for c in wells.columns if c not in set(features)]].reset_index(drop=True)
    result = carried.assign(x=points[:, 0], y=points[:, 1])
    result.attrs['umap_version'] = getattr(umap, '__version__', '')
    return result


def run_embedding_profiles(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Crop, embed, aggregate and (optionally) map one plate from settings.

    Reads the cells of ``embedding_cells`` (joined to ``embedding_wells``
    when given), draws them with :func:`select_crop_cells`, writes the
    shard (:func:`write_crop_shard`), embeds it (:func:`embed_crop_shard`),
    aggregates the wells (:func:`aggregate_embedding_wells`) and, with
    ``embedding_umap``, writes ``well_umap.csv`` (:func:`well_umap`) of
    the median wells, scaled to the control cells' percentiles when
    ``umap_percentiles`` is on. Everything goes to ``embedding_out``
    (default ``<src>/embeddings``): ``crops/``, ``emb/``, ``tables/``.

    :param settings: the keys of :func:`embedding_profiles_defaults`.
    :returns: ``{"shard", "embedding", "aggregate", "umap"}``.
    :raises ValueError: without ``embedding_src`` or ``embedding_cells``, or
        for an unknown store or illumination.
    """
    from .tabular import read_table, write_table

    s = embedding_profiles_defaults(settings)
    if not s['embedding_src'] or (isinstance(s['embedding_cells'], str) and not s['embedding_cells']):
        raise ValueError('embedding_src and embedding_cells are needed')
    if s['crop_store'] not in CROP_STORES:
        raise ValueError(f'crop_store must be one of {CROP_STORES}')
    if s['crop_illumination'] not in CROP_ILLUMINATIONS:
        raise ValueError(f'crop_illumination must be one of {CROP_ILLUMINATIONS}')
    src = Path(os.fspath(s['embedding_src']))
    out = Path(s['embedding_out'] or src / 'embeddings')
    cells = s['embedding_cells']
    cells = cells if isinstance(cells, pd.DataFrame) else read_table(
        cells, canonicalise=False, report=None)
    if s['embedding_wells'] is not None and not (isinstance(s['embedding_wells'], str)
                                                 and not s['embedding_wells']):
        wells = s['embedding_wells']
        wells = wells if isinstance(wells, pd.DataFrame) else read_table(
            wells, canonicalise=False, report=None)
        cells = cells.merge(wells, on=['plate', 'well'], how='inner')
    plate = str(cells['plate'].iat[0])
    chosen = select_crop_cells(cells, cap_per_well=int(s['crop_cap_per_well']),
                               seed=int(s['crop_seed']))
    illumination = None
    if s['crop_illumination'] == 'gain_offset':
        illumination = s['crop_illumination_model'] or src / 'illumination' / 'illumination_model.npz'
    shard = write_crop_shard(src / 'merged', chosen, out / 'crops', plate=plate,
                             size=int(s['crop_window']), channels=_ints(s['crop_channels']),
                             illumination=illumination, mask_flags=bool(s['crop_mask_flags']))
    embedding = embed_crop_shard(shard['shard'], embedding_spec(s), out / 'emb',
                                 masked=bool(s['embedding_masked']))
    index = read_table(shard['cells'], canonicalise=False, report=None)
    keep = ['plate', 'well'] + ([s['embedding_control_column']]
                                if s['embedding_control_column'] in index.columns else [])
    table = embedding_cells(np.load(embedding['path']), index, keep=keep)
    aggregate = aggregate_embedding_wells([table], out / 'tables', levels=s['embedding_aggregate'],
                                          control_column=s['embedding_control_column'],
                                          control=s['embedding_control'])
    result = {'shard': shard, 'embedding': embedding, 'aggregate': aggregate, 'umap': None}
    if s['embedding_umap']:
        median = read_table(out / 'tables' / 'wells_median.parquet', canonicalise=False, report=None)
        features = [c for c in median.columns if c.startswith('emb_')]
        limits = out / 'tables' / f"{str(s['embedding_control']).lower()}_percentiles.parquet"
        if s['umap_percentiles'] and limits.exists():
            from .sp_stats import _normalize_profiles

            q = read_table(limits, canonicalise=False, report=None)
            low = q[np.isclose(q['q'], 0.01)].set_index('plate')
            high = q[np.isclose(q['q'], 0.99)].set_index('plate')
            median = _normalize_profiles(median, features, method='ljosa_percentile', by='plate',
                                         percentiles=(low, high))
            flat = set(median.attrs.get('flat_features', ()))
            median = median.drop(columns=list(flat))
            features = [c for c in features if c not in flat]
            median[features] = median[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        points = well_umap(median, features=features, metric=s['umap_metric'],
                           n_neighbors=int(s['umap_neighbors']), min_dist=float(s['umap_min_dist']),
                           seed=int(s['umap_seed']))
        path = write_table(points, out / 'well_umap.csv', canonicalise=False)
        result['umap'] = {'path': path, 'umap_version': points.attrs.get('umap_version', '')}
    return result
