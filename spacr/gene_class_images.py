"""Image-model arms of the gene-class and MoA benchmark: crops, samplers, the saved transform and the train loop.

This is the image half of item 889, ported from the BMC reanalysis
``gc_train.py`` so that a benchmark or pilot fold is a spaCR run. One fold:

1. the crops: :class:`MemmapCrops` (BBBC021, a ``(cells, H, W, C)`` uint16
   ``.npy`` with an index table holding ``scale0..`` per cell) or
   :class:`OpsCrops` (OPS, cell-masked TIFF crops in tar shards with
   ``.index.csv`` byte offsets and a per-plate scale file). Both divide by a
   fixed per-plate, per-channel scale and clip, so crops stay raw on disk;
2. the labels and folds: :func:`bbbc_fold_setup` (MoA classes and the
   NSCB / NSC folds of :func:`spacr.well_set_training.moa_folds`) or
   :func:`ops_fold_setup` (multi-label gene classes from the
   :mod:`spacr.gene_classes` tables, gene folds, decoy heads, the shuffled
   control and the per-gene loss weight);
3. the sampler (``train_balanced_sampling``): :class:`BalancedSampler`,
   class-uniform for BBBC021 and, for OPS, :func:`class_gene_sampler` (half
   the batch from positive genes, class then gene then cell; half from
   negative genes);
4. the model (``train_arch``, :data:`IMAGE_ARCHITECTURES`):
   :func:`build_image_model` with the first convolution adapted to the
   channel count (:func:`adapt_first_conv`);
5. the transform: :func:`transform_spec` records the whole preprocessing,
   which is saved with the fold and checked again before scoring
   (:func:`check_transform`, item 772). :class:`DevicePrep` applies it on the
   device, with the train-time augmentation (dihedral, shift of up to 4 px,
   per-channel gain 0.9 to 1.1);
6. :func:`train_image_fold` trains with AdamW and a cosine schedule (AMP on a
   GPU, micro-batch halving on out-of-memory) and scores the held-out cells;
   :func:`score_bbbc_run` scores a BBBC021 run per well and per treatment.

The config keys and seeds are the reference's (``seed + fold`` for torch,
the sampler and the augmentation), so a CPU fold with the same config draws
the same batches. :func:`run_gene_class_images` runs every fold from the
settings of :mod:`spacr.gene_classes` (``ml_model='image'``).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The image architectures, by ``train_arch`` name: (reference arch, mode).
IMAGE_ARCHITECTURES: Dict[str, Tuple[str, str]] = {
    'resnet18': ('resnet18', 'finetune'),
    'resnet50': ('resnet50', 'finetune'),
    'efficientnet_b0': ('efficientnet_b0', 'finetune'),
    'convnext_tiny': ('convnext_tiny', 'finetune'),
    'dinov2': ('vit_s_dinov2', 'finetune'),
    'dinov2_frozen': ('vit_s_dinov2', 'frozen'),
    'maxvit_t': ('spacr_maxvit_t', 'finetune'),
    'spacr_cnn': ('spacr_cnn', 'finetune'),
}

#: ImageNet input statistics.
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

#: Interpolation of each pretrained arm, from its weights' transforms.
PRETRAINED_PREP = {
    'resnet18': 'bilinear', 'resnet50': 'bilinear', 'convnext_tiny': 'bilinear', 'efficientnet_b0': 'bicubic',
    'spacr_maxvit_t': 'bicubic', 'vit_s_dinov2': 'bicubic'}

#: The balanced sampling rules (``train_balanced_sampling``).
SAMPLING_RULES = ('class_gene', 'class', 'none')

#: Tooltip of ``train_arch``, shared by :mod:`spacr.architecture_training` and :mod:`spacr.gene_classes`.
TRAIN_ARCH_TOOLTIP = (
    "(str) - The model trained. In spacr_architectures, 'spacr_channel_cnn' and "
    "'spacr_channel_vit' train on cell crops (train_level 'cell') and 'spacr_well_transformer' on"
    " a well's cell embeddings (train_level 'well_set'); a model that does not fit the level "
    "stops the run. In gene_classes image arms: 'resnet18', 'resnet50', 'efficientnet_b0', "
    "'convnext_tiny', 'dinov2', 'dinov2_frozen' (linear probe), 'maxvit_t' or 'spacr_cnn' (from "
    "scratch). Default 'spacr_well_transformer' in spacr_architectures, 'resnet18' in "
    "gene_classes.")

#: Tooltip of ``train_protocol``, shared by :mod:`spacr.architecture_training` and :mod:`spacr.gene_classes`.
TRAIN_PROTOCOL_TOOLTIP = (
    "(str) - BBBC021 folds: 'NSCB' (not same compound or batch: one fold per imaging week, and taxol at 1 "
    "and 3 uM) or 'NSC' (not same compound: four folds, every compound of a MoA in another fold). "
    "Default 'NSCB'.")

#: The heading ``train_arch`` and ``train_protocol`` share in both modules (a key has one category).
ARCHITECTURE_CATEGORY = 'Model architecture'

#: The defaults of a fold config, as the reference benchmark's ``make_configs.py``.
FOLD_DEFAULTS: Dict[str, Any] = {'steps': 1500, 'batch': 64, 'lr': 3e-4, 'probe_lr': 1e-3, 'weight_decay': 1e-5,
                                 'size': 224, 'dropout': 0.1, 'seed': 0, 'threads': 2, 'loader_threads': 4}


def _read(path: Any) -> pd.DataFrame:
    """Read a table exactly as written."""
    from .tabular import read_table

    return read_table(path, canonicalise=False, report=None)


class OpsCrops:
    """The OPS crop table: shard, byte offset, size, gene and plate of every crop, and the plate scales.

    :param cfg: fold config with ``shard_dirs``, ``genotype``, ``scales``
        and ``tables``; optional ``shard_glob`` (default ``GC_*.tar.index.csv``),
        ``restrict_to_glob``, ``genes`` (a gene list file),
        ``drop_window_edge`` and ``cells_per_gene_cap`` (drawn with seed 0).
    """

    def __init__(self, cfg: Mapping[str, Any]):
        """Read the shard indexes, join the genotype and map the control guides to pseudo-genes."""
        idx = []
        for d in cfg['shard_dirs']:
            for p in sorted(Path(d).rglob(cfg.get('shard_glob', 'GC_*.tar.index.csv'))):
                x = _read(p)
                x['shard'] = str(p)[: -len('.index.csv')]
                idx.append(x)
        idx = pd.concat(idx, ignore_index=True)
        if cfg.get('restrict_to_glob'):
            keys = pd.concat([_read(p)[['well', 'site', 'label']] for d in cfg['shard_dirs']
                              for p in Path(d).rglob(cfg['restrict_to_glob'])])
            idx = idx.merge(keys.drop_duplicates(), on=['well', 'site', 'label'])
        gt = _read(cfg['genotype'])
        if 'plate' not in gt:
            gt['plate'] = idx['plate'].iloc[0]
        idx = idx.merge(gt[['plate', 'well', 'site', 'label', 'sgRNA', 'gene']], on=['plate', 'well', 'site', 'label'])
        nt = _read(Path(cfg['tables']) / 'nt_pseudogenes.csv')
        ntmap = dict(zip(nt['sgRNA'], nt['pseudogene']))
        is_nt = idx['gene'] == 'nontargeting'
        idx.loc[is_nt, 'gene'] = idx.loc[is_nt, 'sgRNA'].map(ntmap)
        idx = idx.dropna(subset=['gene'])
        if cfg.get('genes'):
            keep = set(Path(cfg['genes']).read_text().split())
            idx = idx[idx['gene'].isin(keep)]
        idx = idx[idx['window_edge'] == 0] if cfg.get('drop_window_edge', False) else idx
        cap = cfg.get('cells_per_gene_cap', 0)
        if cap:
            idx = pd.concat([_cap_sample(g, cap) for _, g in idx.groupby('gene', sort=True)])
        self.idx = idx.reset_index(drop=True)
        self.scales = json.loads(Path(cfg['scales']).read_text())


def _cap_sample(g: pd.DataFrame, cap: int) -> pd.DataFrame:
    """At most ``cap`` rows of one gene, drawn with seed 0."""
    return g.sample(min(cap, len(g)), random_state=0)


def read_shard_crop(path: str, offset: int, size: int, handles: Dict[str, int]) -> np.ndarray:
    """Read one TIFF crop from a tar shard by byte offset.

    :param path: the shard.
    :param offset: byte offset of the crop's data.
    :param size: its length in bytes.
    :param handles: open file descriptors by path, filled on first use.
    :returns: the crop array, channels first.
    """
    import tifffile

    fd = handles.get(path)
    if fd is None:
        fd = handles[path] = os.open(path, os.O_RDONLY)
    with tifffile.TiffFile(io.BytesIO(os.pread(fd, size, offset))) as t:
        return t.series[0].asarray()


class OpsCropDataset:
    """OPS crops divided by their plate's channel scale and clipped.

    :param cells: an :class:`OpsCrops`.
    :param rows: the crop rows served.
    :param clip: upper clip after the scale.
    """

    def __init__(self, cells: OpsCrops, rows: Sequence[int], clip: float = 1.0):
        """Store the table, the rows and the clip."""
        self.c, self.rows, self.fh, self.clip = cells, np.asarray(rows), {}, clip

    def __len__(self) -> int:
        """Number of crops served."""
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, int]:
        """The ``i``-th crop and its row.

        :param i: position in ``rows``.
        :returns: the scaled crop and its row number.
        """
        r = self.c.idx.iloc[self.rows[i]]
        x = read_shard_crop(r['shard'], int(r['offset_data']), int(r['size']), self.fh).astype(np.float32)
        s = np.asarray(self.c.scales[r['plate']], np.float32)[:, None, None]
        return np.clip(x / s, 0, self.clip), int(self.rows[i])


class MemmapCrops:
    """Crops of one ``(cells, H, W, C)`` array, divided by the per-cell scale columns of the index.

    :param mm_path: the ``.npy`` array (opened memory-mapped on first use).
    :param idx: the index table, one row per crop, with ``scale0``, ``scale1``...
    :param rows: the crop rows served.
    :param clip: upper clip after the scale.
    """

    def __init__(self, mm_path: Any, idx: pd.DataFrame, rows: Sequence[int], clip: float = 1.0):
        """Store the path, the scales, the rows and the clip."""
        self.mm_path, self.idx, self.rows, self.mm, self.clip = mm_path, idx, np.asarray(rows), None, clip
        cols = sorted((c for c in idx.columns if c.startswith('scale') and c[5:].isdigit()), key=lambda c: int(c[5:]))
        self.sc = idx[cols].to_numpy(np.float32)

    def __len__(self) -> int:
        """Number of crops served."""
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, int]:
        """The ``i``-th crop, channels first, and its row.

        :param i: position in ``rows``.
        :returns: the scaled crop and its row number.
        """
        if self.mm is None:
            self.mm = np.load(self.mm_path, mmap_mode='r')
        j = int(self.rows[i])
        x = np.moveaxis(np.asarray(self.mm[j], np.float32), -1, 0) / self.sc[j][:, None, None]
        return np.clip(x, 0, self.clip), j


def collate_crops(batch: Sequence[Tuple[np.ndarray, int]]):
    """Stack ``(crop, row)`` pairs into a float tensor and a row tensor.

    :param batch: the items of a dataset.
    :returns: the ``(B, C, H, W)`` crops and the ``(B,)`` rows.
    """
    import torch

    return torch.from_numpy(np.stack([b[0] for b in batch])), torch.tensor([b[1] for b in batch])


def adapt_first_conv(conv, c_in: int):
    """A copy of a 3-channel convolution for ``c_in`` channels.

    Every input filter is the mean of the RGB filters, scaled by ``3 / c_in``,
    so the response to a uniform input is unchanged.

    :param conv: the pretrained convolution.
    :param c_in: the new input channel count.
    :returns: the new ``nn.Conv2d``.
    """
    import torch.nn as nn

    w = conv.weight.data
    new = nn.Conv2d(c_in, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    new.weight.data = w.mean(1, keepdim=True).repeat(1, c_in, 1, 1) * (3.0 / c_in)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    return new


def build_image_model(arch: str, c_in: int, n_out: int, pretrained: bool = True):
    """The backbone of a reference arch and its feature dimension.

    :param arch: a reference arch name (the first item of a :data:`IMAGE_ARCHITECTURES` value).
    :param c_in: input channels.
    :param n_out: outputs, used by the arms that bring their own head.
    :param pretrained: load the ImageNet (or DINOv2) weights.
    :returns: ``(backbone, dim)``; ``dim`` is None for the arms with their own
        head (``spacr_maxvit_t``: spaCR's TorchModel; ``spacr_cnn``: spaCR's
        :class:`spacr.utils.CustomCellClassifier`).
    :raises ValueError: an unknown arch.
    """
    import torch.nn as nn
    import torchvision.models as tvm

    if arch in ('resnet18', 'resnet50'):
        m = getattr(tvm, arch)(weights='DEFAULT' if pretrained else None)
        m.conv1 = adapt_first_conv(m.conv1, c_in)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif arch == 'efficientnet_b0':
        m = tvm.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        m.features[0][0] = adapt_first_conv(m.features[0][0], c_in)
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    elif arch == 'convnext_tiny':
        m = tvm.convnext_tiny(weights='DEFAULT' if pretrained else None)
        m.features[0][0] = adapt_first_conv(m.features[0][0], c_in)
        dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()
    elif arch == 'vit_s_dinov2':
        import timm

        m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=pretrained, num_classes=0,
                              img_size=224, dynamic_img_size=True)
        m.patch_embed.proj = adapt_first_conv(m.patch_embed.proj, c_in)
        dim = m.num_features
    elif arch == 'spacr_maxvit_t':
        from .utils import TorchModel

        m = TorchModel(model_name='maxvit_t', pretrained=pretrained, dropout_rate=0.1, num_classes=n_out,
                       multilabel=n_out > 12, image_size=224)
        if c_in != 3:
            m.base_model.stem[0][0] = adapt_first_conv(m.base_model.stem[0][0], c_in)
        return m, None
    elif arch == 'spacr_cnn':
        from .utils import CustomCellClassifier

        return CustomCellClassifier(num_classes=n_out, in_channels=c_in), None
    else:
        raise ValueError(f"no image architecture {arch!r}")
    return m, dim


def make_image_net(cfg: Mapping[str, Any], c_in: int, n_out: int):
    """The model of a fold config: backbone plus a dropout and linear head, frozen backbone in ``'frozen'`` mode.

    :param cfg: fold config with ``arch``, ``mode``, ``pretrained`` and ``dropout``.
    :param c_in: input channels.
    :param n_out: outputs.
    :returns: ``(model, backbone)``; ``backbone`` is None for the arms with their own head.
    """
    import torch.nn as nn

    backbone, dim = build_image_model(cfg['arch'], c_in, n_out, cfg.get('pretrained', True))
    if dim is None:
        return backbone, None
    head = nn.Sequential(nn.Dropout(cfg.get('dropout', 0.1)), nn.Linear(dim, n_out))
    model = nn.Sequential(backbone, head)
    if cfg.get('mode') == 'frozen':
        for p in backbone.parameters():
            p.requires_grad = False
    return model, backbone


def transform_spec(cfg: Mapping[str, Any], c_in: int) -> Dict[str, Any]:
    """The whole preprocessing of a fold, saved with it and checked before scoring.

    Pretrained arms: the fixed plate x channel scale, clip to [0, 1], the
    model's ImageNet mean and standard deviation per channel (channels past
    the third take the mean of the RGB values, as the stem adaptation) and a
    resize to ``size`` with the model's interpolation and antialiasing.
    From-scratch arms: the scale, clip to [0, 1], ``(x - 0.5) / 0.25`` and a
    bilinear resize. ``legacy_transform`` gives the first queue's clip to
    [0, 1.5] and ``(x - 0.5) / 0.25`` for every arm.

    :param cfg: fold config with ``arch``, ``dataset``, ``size``,
        ``pretrained`` and ``scales`` (OPS) or ``index`` (BBBC021).
    :param c_in: input channels.
    :returns: the spec dict (the md5 of the scales file or index included).
    """
    legacy = bool(cfg.get('legacy_transform'))
    arch = cfg['arch']
    pre = arch in PRETRAINED_PREP and cfg.get('pretrained', True) and not legacy
    if pre:
        mean = [IMAGENET_MEAN[c] if c < 3 else float(np.mean(IMAGENET_MEAN)) for c in range(c_in)]
        std = [IMAGENET_STD[c] if c < 3 else float(np.mean(IMAGENET_STD)) for c in range(c_in)]
        interp = PRETRAINED_PREP[arch]
    else:
        mean, std, interp = [0.5] * c_in, [0.25] * c_in, 'bilinear'
    sc = cfg.get('scales') if cfg['dataset'] == 'ops' else cfg.get('index')
    flat = cfg.get('flat_md5', 'OPS: shard index column flat_md5' if cfg['dataset'] == 'ops'
                   else 'BBBC021: spaCR per-plate illumination model applied at crop export (crops README)')
    return {'version': 2 if not legacy else 1, 'arch': arch, 'kind': 'pretrained' if pre else 'dataset_standardised',
            'clip': [0.0, 1.5 if legacy else 1.0], 'mean': mean, 'std': std, 'size': int(cfg.get('size', 224)),
            'interpolation': interp, 'antialias': True if not legacy else False,
            'crop_px': 96 if cfg['dataset'] == 'ops' else 128,
            'channels': ['DAPI', 'GFP', 'A594', 'AF750'] if cfg['dataset'] == 'ops' else ['DAPI(w1)', 'w2', 'w4'],
            'scales_file': sc,
            'scales_md5': hashlib.md5(Path(sc).read_bytes()).hexdigest() if sc and Path(sc).is_file() else None,
            'flat_field': flat, 'masked': cfg.get('masked', True),
            'augment': 'dihedral(8) + shift +-4 px + per-channel gain U(0.9, 1.1), train only, batch-wise on device'}


def check_transform(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Refuse to score with a preprocessing that differs from the saved one.

    :param saved: the spec saved with the fold or checkpoint.
    :param current: the spec of this run.
    :raises RuntimeError: naming every key that differs.
    """
    diff = {k: (saved.get(k), current.get(k)) for k in set(saved) | set(current) if saved.get(k) != current.get(k)}
    if diff:
        raise RuntimeError(f"preprocessing differs from the checkpoint's transform: {diff}")


class DevicePrep:
    """Train-time augmentation and the fixed transform of a spec, batch-wise on the device.

    Augmentation (``train``): one random rotation by 90-degree steps and one
    random horizontal flip per batch, one random shift of up to 4 px (rolled),
    and a per-crop, per-channel gain in [0.9, 1.1] followed by the spec's clip.
    No batch statistics are used.

    :param spec: a :func:`transform_spec` dict.
    :param train: apply the augmentation.
    :param seed: seed of the augmentation's CPU generator.
    """

    def __init__(self, spec: Mapping[str, Any], train: bool, seed: int = 0):
        """Store the spec and seed the generator."""
        import torch

        self.spec, self.train = spec, train
        self.size = spec['size']
        self.g = torch.Generator(device='cpu').manual_seed(seed)
        self.mean = torch.tensor(spec['mean']).view(1, -1, 1, 1)
        self.std = torch.tensor(spec['std']).view(1, -1, 1, 1)

    def __call__(self, x):
        """Augment (in training), normalise and resize a batch.

        :param x: scaled crops shaped ``(B, C, H, W)``.
        :returns: the model input shaped ``(B, C, size, size)``.
        """
        import torch
        import torch.nn.functional as F

        if self.train:
            k = int(torch.randint(0, 4, (1,), generator=self.g))
            x = torch.rot90(x, k, (2, 3))
            if torch.rand(1, generator=self.g) < 0.5:
                x = torch.flip(x, (3,))
            dy, dx = (int(v) for v in torch.randint(-4, 5, (2,), generator=self.g))
            x = torch.roll(x, (dy, dx), (2, 3))
            gain = 0.9 + 0.2 * torch.rand(x.shape[0], x.shape[1], 1, 1, generator=self.g).to(x.device)
            x = (x * gain).clamp(*self.spec['clip'])
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode=self.spec['interpolation'],
                              align_corners=False, antialias=self.spec['antialias'])
        return x


class BalancedSampler:
    """Draw crop rows: a weighted choice of group, then a uniform crop of it.

    :param groups: arrays of crop rows; empty groups are dropped.
    :param weights: one weight per group (normalised).
    :param seed: numpy seed.
    """

    def __init__(self, groups: Sequence[np.ndarray], weights: Sequence[float], seed: int):
        """Keep the non-empty groups and their normalised weights."""
        self.groups = [g for g in groups if len(g)]
        self.w = np.asarray([w for g, w in zip(groups, weights) if len(g)], float)
        self.w /= self.w.sum()
        self.rng = np.random.default_rng(seed)

    def draw(self, n: int) -> np.ndarray:
        """Draw ``n`` crop rows.

        :param n: batch size.
        :returns: the rows.
        """
        gi = self.rng.choice(len(self.groups), n, p=self.w)
        return np.array([self.groups[k][self.rng.integers(len(self.groups[k]))] for k in gi])


def class_gene_sampler(setup: Mapping[str, Any], rows: np.ndarray, seed: int) -> BalancedSampler:
    """The OPS ``'class_gene'`` sampler.

    Half the weight goes to positive genes, class-uniform and then
    gene-uniform within the class (a gene in several classes gets each share);
    the other half to the negative genes (control pseudo-genes and genes in no
    class), uniformly; cells are uniform within a gene.

    :param setup: the :func:`ops_fold_setup` dict.
    :param rows: the training crop rows.
    :param seed: numpy seed.
    :returns: the sampler.
    """
    goc = setup['gene_of_cell'][rows]
    Y = setup['Y']
    by_gene = pd.Series(rows).groupby(goc).apply(np.asarray).to_dict()
    groups, weights = [], []
    gene_groups: Dict[int, float] = {}
    for j in range(Y.shape[1]):
        gs = [g for g in np.flatnonzero(Y[:, j]) if g in by_gene]
        for g in gs:
            gene_groups.setdefault(g, 0.0)
            gene_groups[g] += 1.0 / max(len(gs), 1)
    tot = sum(gene_groups.values()) or 1.0
    for g, w in gene_groups.items():
        groups.append(by_gene[g])
        weights.append(0.5 * w / tot)
    neg = [g for g in by_gene if Y[g].sum() == 0]
    for g in neg:
        groups.append(by_gene[g])
        weights.append(0.5 / max(len(neg), 1))
    return BalancedSampler(groups, weights, seed)


def ops_fold_setup(cfg: Mapping[str, Any], fold: int) -> Dict[str, Any]:
    """Labels, masks, weights and train / test crops of one OPS gene fold.

    ``Y`` is 1 for a trainable class member; ``M`` is 0 where a gene is an
    annotated member without a phenotype (an unknown label). With
    ``decoy_heads``, random gene sets matched in size to the real classes
    are appended (seed ``seed + 99``). ``shuffle_labels`` permutes the class
    membership across the real genes (seed 1). ``gene_weighting='inv_sqrt'``
    weights a gene by one over the square root of its real class count.

    :param cfg: fold config with the :class:`OpsCrops` keys, ``classes``,
        ``decoy_heads``, ``shuffle_labels``, ``gene_weighting``, ``seed`` and ``out``.
    :param fold: the held-out gene fold.
    :returns: dict with cells, classes, genes, Y, M, W, gene_of_cell,
        train_rows, test_rows and c_in.
    """
    cells = OpsCrops(cfg)
    tb = Path(cfg['tables'])
    cm = _read(tb / 'class_members.csv')
    classes = list(cfg.get('classes') or sorted(cm['class_id'].unique()))
    cm = cm[cm['class_id'].isin(classes)]
    folds = _read(tb / 'gene_folds.csv')
    gf = dict(zip(folds['gene'], folds['fold']))
    nt = _read(tb / 'nt_pseudogenes.csv')
    gf.update(dict(zip(nt['pseudogene'], nt['fold'])))
    genes = sorted(cells.idx['gene'].unique())
    gi = {g: i for i, g in enumerate(genes)}
    ci = {c: j for j, c in enumerate(classes)}
    Y = np.zeros((len(genes), len(classes)), np.float32)
    M = np.ones((len(genes), len(classes)), np.float32)
    for r in cm.itertuples():
        if r.gene in gi:
            if r.trainable:
                Y[gi[r.gene], ci[r.class_id]] = 1
            else:
                M[gi[r.gene], ci[r.class_id]] = 0
    n_decoy = cfg.get('decoy_heads', 0)
    if n_decoy:
        rng = np.random.default_rng(cfg.get('seed', 0) + 99)
        pool = [g for g in genes if not g.startswith('NT_') and Y[gi[g]].sum() > 0]
        sizes = rng.choice(Y.sum(0).astype(int), n_decoy)
        D = np.zeros((len(genes), n_decoy), np.float32)
        for j, s in enumerate(sizes):
            for g in rng.choice(pool, int(s), replace=False):
                D[gi[g], j] = 1
        Y = np.hstack([Y, D])
        M = np.hstack([M, np.ones_like(D)])
        classes = classes + [f'DECOY{j:03d}' for j in range(n_decoy)]
        dec_rows = [(f'DECOY{j:03d}', g) for j in range(n_decoy) for g in np.array(genes)[D[:, j] > 0]]
    if cfg.get('shuffle_labels'):
        rng = np.random.default_rng(1)
        real = np.array([not g.startswith('NT_') for g in genes])
        perm = np.flatnonzero(real)
        src = rng.permutation(perm)
        Y[perm], M[perm] = Y[src], M[src]
    n_real = Y[:, : len(Y[0]) - n_decoy] if n_decoy else Y
    W = (1.0 / np.sqrt(np.maximum(1.0, n_real.sum(1))) if cfg.get('gene_weighting', 'none') == 'inv_sqrt'
         else np.ones(len(genes)))
    gene_of_cell = cells.idx['gene'].map(gi).to_numpy()
    fold_of_cell = cells.idx['gene'].map(gf).to_numpy()
    test = fold_of_cell == fold
    train = ~test & np.isfinite(fold_of_cell)
    if n_decoy and cfg.get('out'):
        from .tabular import write_table

        dp = Path(cfg['out']) / 'decoys.csv'
        if not dp.is_file():
            write_table(pd.DataFrame(dec_rows, columns=['class_id', 'gene']), dp, canonicalise=False)
    return {'cells': cells, 'classes': classes, 'genes': genes, 'Y': Y, 'M': M, 'W': W.astype(np.float32),
            'gene_of_cell': gene_of_cell, 'train_rows': np.flatnonzero(train), 'test_rows': np.flatnonzero(test),
            'c_in': 4}


def bbbc_fold_setup(cfg: Mapping[str, Any], fold: int) -> Dict[str, Any]:
    """Labels and train / test crops of one BBBC021 MoA fold.

    :param cfg: fold config with ``index`` (the crop index, with week,
        compound, concentration and moa), ``design`` (``'NSCB'`` or ``'NSC'``)
        and ``shuffle_labels`` (compound-to-MoA map permuted with seed 1).
    :param fold: the fold number.
    :returns: dict with idx, classes, lab, fold_name, train_rows, test_rows and c_in.
    """
    from .well_set_training import moa_folds

    idx = _read(cfg['index'])
    name, train, test = moa_folds(idx, cfg['design'])[fold]
    classes = sorted(idx['moa'].unique())
    lab = idx['moa'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    if cfg.get('shuffle_labels'):
        rng = np.random.default_rng(1)
        cmoa = idx.drop_duplicates('compound').set_index('compound')['moa']
        perm = dict(zip(cmoa.index, rng.permutation(cmoa.values)))
        lab = idx['compound'].map(perm).map({c: i for i, c in enumerate(classes)}).to_numpy()
    n_ch = sum(1 for c in idx.columns if c.startswith('scale') and c[5:].isdigit()) or 3
    return {'idx': idx, 'classes': classes, 'lab': lab, 'fold_name': name, 'train_rows': np.flatnonzero(train),
            'test_rows': np.flatnonzero(test), 'c_in': n_ch}


def forward_with_feats(model, x):
    """Logits and penultimate features for the arms with a separable head.

    :param model: the fold model.
    :param x: the model input.
    :returns: ``(logits, features)``; features are None for arms without a separable head.
    """
    import torch.nn as nn

    if isinstance(model, nn.Sequential) and len(model) == 2:
        f = model[0](x)
        return model[1](f), f
    if hasattr(model, '_run_backbone') and hasattr(model, 'spacr_classifier'):
        return model(x), model._run_backbone(x)
    return model(x), None


def _cosine_warmup(step: int, *, warm: int, steps: int) -> float:
    """Learning-rate factor: linear warm-up, then a cosine decay to zero."""
    return min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))


def _load_rows(pool, ds, rows):
    """Load crop rows through a thread pool, in order, and collate them."""
    return collate_crops(list(pool.map(ds.__getitem__, rows)))


def _jsonl_log(path: Path, t_start: float, kw: Dict[str, Any]) -> None:
    """Append one event to the fold's train_log.jsonl."""
    kw = dict(kw, t=round(time.time() - t_start, 1))
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(kw) + '\n')


def train_image_fold(cfg: Mapping[str, Any], fold: int, out: Any) -> Dict[str, Any]:
    """Train and score one fold of an image arm.

    Seeds: ``seed + fold`` for torch, the sampler and the augmentation, as the
    reference. ``smoke`` runs ``smoke_steps`` (3) steps of batch 8 and scores
    ``smoke_test_cells`` (256) test cells drawn with seed 0. The loss is the
    label-smoothed (0.1) cross-entropy for BBBC021 and, for OPS, the masked
    binary cross-entropy over the class heads weighted per gene. Writes
    ``transform.json``, ``train_log.jsonl``, ``test_rows.parquet``,
    ``test_probs.npy``, ``cell_probs.parquet`` (BBBC021) or
    ``gene_scores.parquet`` and ``test_cells.parquet`` (OPS),
    ``timing.json`` and, with ``save_weights``, ``model.pt`` holding the state
    dict, the transform and the config.

    :param cfg: the fold config (the reference ``gc_train.py`` keys: dataset,
        arch, mode, pretrained, steps, batch, lr, probe_lr, weight_decay,
        size, dropout, seed, threads, loader_threads, design / tables and the
        crop keys; ``sampling`` picks a rule of :data:`SAMPLING_RULES`).
    :param fold: the fold number.
    :param out: the fold folder.
    :returns: the timing dict.
    :raises RuntimeError: out of memory at micro-batch 1, or a transform that
        changed between training and scoring.
    """
    import torch
    import torch.nn as nn
    from concurrent.futures import ThreadPoolExecutor

    from .tabular import write_table

    cfg = dict(FOLD_DEFAULTS, **cfg)
    out = Path(os.fspath(out))
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'train_log.jsonl'
    smoke = cfg.get('smoke', False)
    seed = cfg.get('seed', 0) + fold
    torch.manual_seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() and not cfg.get('cpu') else 'cpu')
    threads = torch.get_num_threads()
    torch.set_num_threads(cfg.get('threads', 2))
    ds_kind = cfg['dataset']
    S = ops_fold_setup(cfg, fold) if ds_kind == 'ops' else bbbc_fold_setup(cfg, fold)
    n_out = len(S['classes'])
    train_rows, test_rows = S['train_rows'], S['test_rows']
    if smoke:
        rng = np.random.default_rng(0)
        test_rows = np.sort(rng.choice(test_rows, min(len(test_rows), cfg.get('smoke_test_cells', 256)),
                                       replace=False))
    clip = 1.5 if cfg.get('legacy_transform') else 1.0
    rule = cfg.get('sampling', 'class_gene' if ds_kind == 'ops' else 'class')
    if ds_kind == 'ops':
        ds = OpsCropDataset(S['cells'], np.arange(len(S['cells'].idx)), clip)
        sampler = (class_gene_sampler(S, train_rows, seed) if rule == 'class_gene'
                   else BalancedSampler([train_rows], [1.0], seed))
    else:
        ds = MemmapCrops(cfg['memmap'], S['idx'], np.arange(len(S['idx'])), clip)
        lab = S['lab'][train_rows]
        groups = [train_rows[lab == c] for c in range(n_out)] if rule != 'none' else [train_rows]
        sampler = BalancedSampler(groups, [1.0] * len(groups), seed)
    _jsonl_log(log_path, t_start, {'event': 'setup', 'fold': fold, 'train_cells': int(len(train_rows)),
                                   'test_cells': int(len(test_rows)), 'n_out': n_out, 'device': str(dev),
                                   'arch': cfg['arch'], 'mode': cfg.get('mode', 'finetune')})
    model, backbone = make_image_net(cfg, S['c_in'], n_out)
    model.to(dev)
    spec = transform_spec(cfg, S['c_in'])
    (out / 'transform.json').write_text(json.dumps(spec, indent=1), encoding='utf-8')
    prep_tr = DevicePrep(spec, True, seed)
    steps = cfg['steps'] if not smoke else cfg.get('smoke_steps', 3)
    batch = cfg['batch'] if not smoke else 8
    params = [p for p in model.parameters() if p.requires_grad]
    lr = cfg['lr'] if cfg.get('mode') != 'frozen' else cfg['probe_lr']
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg['weight_decay'])
    import functools

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, functools.partial(_cosine_warmup, warm=max(1, int(0.03 * steps)), steps=steps))
    amp = dev.type == 'cuda'
    from .deep_spacr import _gradient_scaler

    scaler = _gradient_scaler(dev, amp)
    Yt = torch.from_numpy(S['Y']) if ds_kind == 'ops' else None
    Mt = torch.from_numpy(S['M']) if ds_kind == 'ops' else None
    Wt = torch.from_numpy(S['W']) if ds_kind == 'ops' else None
    labt = torch.from_numpy(S['lab']) if ds_kind != 'ops' else None
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    pool = ThreadPoolExecutor(cfg['loader_threads'])
    micro, step, seen = batch, 0, 0
    try:
        t0 = time.time()
        nxt = pool.submit(_load_rows, pool, ds, sampler.draw(batch))
        model.train()
        if cfg.get('mode') == 'frozen' and backbone is not None:
            backbone.eval()
        while step < steps:
            x, j = nxt.result()
            nxt = pool.submit(_load_rows, pool, ds, sampler.draw(batch))
            try:
                opt.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for s0 in range(0, len(j), micro):
                    xb = prep_tr(x[s0:s0 + micro].to(dev, non_blocking=True))
                    jb = j[s0:s0 + micro]
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                        logits = model(xb).float()
                    if ds_kind == 'ops':
                        g = torch.from_numpy(S['gene_of_cell'][jb.numpy()])
                        y, m = Yt[g].to(dev), Mt[g].to(dev)
                        m = m * Wt[g].to(dev)[:, None]
                        loss = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
                                * m).sum() / m.sum()
                    else:
                        loss = ce(logits, labt[jb].to(dev))
                    loss = loss * len(jb) / len(j)
                    scaler.scale(loss).backward()
                    loss_sum += float(loss.detach())
                scaler.step(opt)
                scaler.update()
                sched.step()
                oom = False
            except torch.OutOfMemoryError:
                oom = True
            if oom:
                xb = logits = loss = None
                opt.zero_grad(set_to_none=True)
                from . import _gc

                _gc.collect()
                torch.cuda.empty_cache()
                if micro == 1:
                    raise RuntimeError('CUDA out of memory at micro-batch 1')
                micro = max(1, micro // 2)
                _jsonl_log(log_path, t_start, {'event': 'oom', 'micro_batch': micro, 'effective_batch': batch})
                continue
            step += 1
            seen += len(j)
            if step % 50 == 0 or step == steps:
                _jsonl_log(log_path, t_start, {'event': 'step', 'step': step, 'loss': round(loss_sum, 4),
                                               'lr': sched.get_last_lr()[0], 'micro_batch': micro})
        train_s = time.time() - t0
        saved_spec = json.loads((out / 'transform.json').read_text())
        check_transform(saved_spec, transform_spec(cfg, S['c_in']))
        prep_te = DevicePrep(saved_spec, False)
        model.eval()
        t1 = time.time()
        probs = np.zeros((len(test_rows), n_out), np.float32)
        feats = []
        with torch.no_grad():
            for s0 in range(0, len(test_rows), 256):
                rows = test_rows[s0:s0 + 256]
                x, _ = _load_rows(pool, ds, rows)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    if cfg.get('export_embeddings'):
                        lo, fe = forward_with_feats(model, prep_te(x.to(dev)))
                    else:
                        lo, fe = model(prep_te(x.to(dev))), None
                    lo = lo.float()
                probs[s0:s0 + len(rows)] = (torch.sigmoid(lo) if ds_kind == 'ops'
                                            else torch.softmax(lo, 1)).cpu().numpy()
                if fe is not None:
                    feats.append(fe.float().cpu().numpy().astype(np.float16))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        torch.set_num_threads(threads)
    if feats:
        np.save(out / 'test_embeddings_f16.npy', np.concatenate(feats))
    score_s = time.time() - t1
    write_table(pd.DataFrame({'row': test_rows}), out / 'test_rows.parquet', canonicalise=False)
    np.save(out / 'test_probs.npy', probs)
    if ds_kind == 'ops':
        g = S['gene_of_cell'][test_rows]
        gp = pd.DataFrame(probs, columns=S['classes']).groupby(np.array(S['genes'])[g]).mean()
        gp['n_cells'] = pd.Series(np.array(S['genes'])[g]).value_counts()
        gp.to_parquet(out / 'gene_scores.parquet')
        cols = [c for c in ('plate', 'well', 'site', 'label', 'prcfo', 'gene') if c in S['cells'].idx.columns]
        write_table(S['cells'].idx.iloc[test_rows][cols].reset_index(drop=True), out / 'test_cells.parquet',
                    canonicalise=False)
    else:
        idx = S['idx'].iloc[test_rows]
        w = pd.DataFrame(probs, columns=S['classes'])
        meta = ['plate', 'well', 'compound', 'concentration', 'moa']
        w[meta] = idx[meta].to_numpy()
        write_table(w, out / 'cell_probs.parquet', canonicalise=False)
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0.0
    timing = {'fold': fold, 'train_s': round(train_s, 1), 'score_s': round(score_s, 1), 'steps': steps,
              'batch': batch, 'final_micro_batch': micro, 'peak_vram_gb': round(peak, 2),
              'train_img_s': round(seen / max(train_s, 1e-9), 1), 'test_cells': int(len(test_rows)),
              'device': str(dev), 'params_m': round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    (out / 'timing.json').write_text(json.dumps(timing, indent=1), encoding='utf-8')
    if cfg.get('save_weights'):
        torch.save({'state_dict': model.state_dict(), 'transform': spec, 'config': cfg}, out / 'model.pt')
    _jsonl_log(log_path, t_start, dict(timing, event='done'))
    return timing


def score_bbbc_run(run: Any, cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Score a BBBC021 image run as the paper CNN: well mean, untrained classes zeroed, treatment mean, argmax.

    Writes ``eval/treatment_probs.csv``, ``eval/well_probs.csv`` and
    ``eval/summary.json`` with accuracy, accuracy over solvable treatments
    (MoA trained in that fold), well accuracy, macro AUROC and AP, and the
    majority-class control.

    :param run: the run folder with ``fold_XX/cell_probs.parquet``.
    :param cfg: the run config; None reads ``run/config.json``.
    :returns: the summary dict.
    :raises FileNotFoundError: no fold has cell probabilities.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from .tabular import write_table
    from .well_set_training import moa_folds

    run = Path(os.fspath(run))
    cfg = dict(cfg or json.loads((run / 'config.json').read_text()))
    idx = _read(cfg['index'])
    classes = sorted(idx['moa'].unique())
    fl = moa_folds(idx, cfg['design'])
    wells, tim = [], []
    for fd in sorted(run.glob('fold_*')):
        k = int(fd.name.split('_')[1])
        if not (fd / 'cell_probs.parquet').is_file():
            continue
        c = _read(fd / 'cell_probs.parquet')
        trained = set(idx['moa'].values[fl[k][1]])
        w = c.groupby(['plate', 'well', 'compound', 'concentration', 'moa'], as_index=False)[classes].mean()
        P = w[classes].to_numpy(float, copy=True)
        P[:, [cl not in trained for cl in classes]] = 0
        P /= P.sum(1, keepdims=True)
        w[classes] = P
        w['fold'] = fl[k][0]
        w['solvable'] = w['moa'].isin(trained)
        wells.append(w)
        if (fd / 'timing.json').is_file():
            tim.append(json.loads((fd / 'timing.json').read_text()))
    if not wells:
        raise FileNotFoundError(f"no fold_*/cell_probs.parquet in {run}")
    w = pd.concat(wells, ignore_index=True)
    if cfg.get('shuffle_labels'):
        rng = np.random.default_rng(1)
        cmoa = idx.drop_duplicates('compound').set_index('compound')['moa']
        perm = dict(zip(cmoa.index, rng.permutation(cmoa.values)))
        w['moa'] = w['compound'].map(perm)
    t = w.groupby(['compound', 'concentration', 'moa', 'fold', 'solvable'], as_index=False)[classes].mean()
    T = t[classes].to_numpy()
    t['pred'] = np.array(classes)[T.argmax(1)]
    hit = t['pred'] == t['moa']
    au, ap = [], []
    for c in classes:
        y = (t['moa'] == c).to_numpy()
        if 0 < y.sum() < len(y):
            au.append(roc_auc_score(y, t[c]))
            ap.append(average_precision_score(y, t[c]))
    res = {'run': run.name, 'arch': cfg['arch'], 'mode': cfg.get('mode', 'finetune'), 'design': cfg['design'],
           'shuffled': bool(cfg.get('shuffle_labels')), 'folds_done': len(wells), 'n_treatments': len(t),
           'accuracy': float(hit.mean()), 'accuracy_solvable': float(hit[t.solvable].mean()),
           'well_accuracy': float((w[classes].to_numpy().argmax(1)
                                   == w['moa'].map({c: i for i, c in enumerate(classes)})).mean()),
           'macro_auroc': float(np.mean(au)), 'macro_ap': float(np.mean(ap)),
           'majority_class_accuracy': float(t['moa'].value_counts(normalize=True).iloc[0]),
           'train_h': sum(x['train_s'] + x['score_s'] for x in tim) / 3600,
           'peak_vram_gb': max((x['peak_vram_gb'] for x in tim), default=0)}
    (run / 'eval').mkdir(exist_ok=True)
    write_table(t, run / 'eval' / 'treatment_probs.csv', canonicalise=False)
    write_table(w, run / 'eval' / 'well_probs.csv', canonicalise=False)
    (run / 'eval' / 'summary.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    return res


def image_fold_config(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The fold config of an image run from the settings of :mod:`spacr.gene_classes`.

    :param settings: the gene-class settings with ``ml_model='image'``.
    :returns: the config dict of :func:`train_image_fold`.
    :raises ValueError: an unknown ``train_arch`` or ``train_balanced_sampling``,
        or a dataset whose crop inputs are missing.
    """
    arch = settings['train_arch']
    if arch not in IMAGE_ARCHITECTURES:
        raise ValueError(f"train_arch={arch!r} is not one of {list(IMAGE_ARCHITECTURES)}")
    rule = settings['train_balanced_sampling']
    if rule not in SAMPLING_RULES:
        raise ValueError(f"train_balanced_sampling={rule!r} is not one of {list(SAMPLING_RULES)}")
    ref_arch, mode = IMAGE_ARCHITECTURES[arch]
    dataset = settings['train_image_dataset']
    cfg: Dict[str, Any] = {'dataset': dataset, 'arch': ref_arch, 'mode': mode, 'size': int(settings['train_input_size']),
                           'steps': int(settings['train_steps']), 'batch': int(settings['train_batch']),
                           'seed': int(settings['train_seed']), 'sampling': rule,
                           'shuffle_labels': bool(settings['train_shuffle_control']),
                           'pretrained': arch != 'spacr_cnn', 'save_weights': True}
    if dataset == 'bbbc021':
        if not settings['train_crops'] or not settings['train_crop_index']:
            raise ValueError('train_crops and train_crop_index name the BBBC021 crop array and its index')
        cfg.update(memmap=settings['train_crops'], index=settings['train_crop_index'],
                   design=settings['train_protocol'])
    elif dataset == 'ops':
        if not settings['train_crops'] or not settings['train_genotype'] or not settings['train_crop_scales']:
            raise ValueError('train_crops (shard folders), train_genotype and train_crop_scales name the OPS crops')
        cfg.update(shard_dirs=[p.strip() for p in str(settings['train_crops']).split(',') if p.strip()],
                   genotype=settings['train_genotype'], scales=settings['train_crop_scales'],
                   tables=settings['train_gene_classes'], decoy_heads=int(settings['train_decoy_heads']),
                   classes=[c for c in str(settings.get('train_classes') or '').split(',') if c.strip()] or None)
    else:
        raise ValueError(f"train_image_dataset={dataset!r} is not 'bbbc021' or 'ops'")
    return cfg


def run_gene_class_images(settings: Mapping[str, Any], folds: Optional[Sequence[int]] = None,
                          log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Train every fold of an image arm from settings and score it.

    :param settings: the settings of :mod:`spacr.gene_classes` with ``ml_model='image'``.
    :param folds: fold numbers; None runs ``train_folds`` folds (OPS) or every fold of ``train_protocol`` (BBBC021).
    :param log: called with one line per finished fold.
    :returns: dict with ``config``, ``timings`` and, for BBBC021, ``summary``.
    """
    from .well_set_training import moa_folds

    cfg = image_fold_config(settings)
    dst = Path(os.fspath(settings['gene_class_dst'] or 'gene_class_images'))
    cfg['out'] = str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    if folds is None:
        n = (len(moa_folds(_read(cfg['index']), cfg['design'])) if cfg['dataset'] == 'bbbc021'
             else int(settings['train_folds']))
        folds = list(range(n))
    timings = []
    for k in folds:
        timings.append(train_image_fold(cfg, k, dst / f'fold_{k:02d}'))
        log(f"fold {k}: {timings[-1]['test_cells']} test cells, {timings[-1]['train_s']} s")
    out: Dict[str, Any] = {'config': cfg, 'timings': timings}
    if cfg['dataset'] == 'bbbc021':
        out['summary'] = score_bbbc_run(dst, cfg)
    return out
