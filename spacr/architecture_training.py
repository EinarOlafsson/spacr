"""Train the spaCR channel-adaptive architectures from settings: well sets, cells, pre-training, missing stains.

One settings file runs one of two levels (``train_level``):

* ``'well_set'``: the :class:`spacr.channel_models.SpacrWellTransformer`
  on per-cell embeddings, scored by mechanism of action with its ablations
  and controls (:func:`spacr.well_set_training.run_well_set_training`).
* ``'cell'``: the channel-adaptive CNN or ViT (``train_arch``) on single-cell
  crops (:func:`run_cell_architecture`). Optionally the ViT is first
  pre-trained without labels (``pretrain_method``: ``'mae'``, He et al.
  2022, or ``'dino'``, Caron et al. 2021) for ``pretrain_epochs`` epochs;
  ``train_labels_per_class`` then keeps only that many labelled cells per
  class (few-shot). Every model is scored on the full stain panel and on
  each subset of ``eval_channel_subsets`` (the missing-stain evaluation),
  beside a majority-class control and, with ``train_set_shuffle_control``, a
  model trained on shuffled labels.

Cell crops come as one ``(cells, channels, height, width)`` array
(``train_cell_crops``, ``.npy``) with a row-aligned table
(``train_cell_table``) holding the label and the group (well) column; cells
are split by group, so no well is in both training and test. The crops are
raw: each channel is divided by the 99th percentile of up to 2,048 training
crops, clipped at 1.5 and mapped by ``(x - 0.5) / 0.25``, the transform of
the gene-class benchmark. Training augments by flips, 90-degree rotations and
a per-channel gain within 10%.

``spacr-run spacr_architectures`` and ``python -m
spacr.architecture_training --settings file.csv`` run it headless.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .channel_models import (CHANNEL_ARCHITECTURES, CHANNEL_IMAGE_ARCHITECTURES, ChannelClassifier,
                             ChannelDINO, ChannelMAE, SpacrChannelViT, build_channel_model, default_stains)
from .well_set_training import cosine_schedule

logger = logging.getLogger(__name__)

#: Key the settings of this module are registered under.
APP_KEY = 'spacr_architectures'

#: The training levels.
TRAIN_LEVELS = ('well_set', 'cell')

#: The self-supervised pre-training methods (``'none'`` trains from scratch).
PRETRAIN_METHODS = ('none', 'mae', 'dino')

#: The columns of the cell-level ``summary.csv``, in order.
CELL_SUMMARY_COLUMNS = ('arm', 'arch', 'pretrain', 'labels_per_class', 'shuffled', 'channels', 'n_train',
                        'n_test', 'accuracy', 'macro_f1', 'seed')

#: What each column of the cell-level ``summary.csv`` means.
CELL_GLOSSARY: Dict[str, str] = {
    'arm': 'the model scored, or the control row majority_class (always the commonest training class)',
    'arch': 'spacr_channel_cnn (channel-adaptive CNN) or spacr_channel_vit (channel-aware ViT)',
    'pretrain': 'self-supervised pre-training before the labels: none, mae (masked autoencoder) or '
                'dino (self-distillation)',
    'labels_per_class': 'labelled training cells kept per class (few-shot); 0 keeps all',
    'shuffled': 'True for the shuffled-label control: the training labels permuted',
    'channels': 'the stains the model saw at test time; the full panel or a subset (missing stains)',
    'n_train': 'labelled training cells',
    'n_test': 'test cells, from wells never seen in training',
    'accuracy': 'share of test cells predicted right',
    'macro_f1': 'F1 score per class, averaged over the classes',
    'seed': 'seed of the split, the few-shot draw, the initialisation and the batches',
}

_DEFAULTS: Dict[str, Any] = {
    'train_arch': 'spacr_well_transformer',
    'train_level': 'well_set',
    'train_set_input': '',
    'train_set_cells': '',
    'train_set_tag': 'resnet18',
    'train_set_normalisation': 'plate_control_pca',
    'train_set_pca_dims': 256,
    'train_set_control': 'DMSO',
    'train_set_max_cells': 300,
    'train_protocol': 'NSCB',
    'train_set_arms': 'settf,n1,mean_lr,mean_knn',
    'train_set_steps': 800,
    'train_set_bag_batch': 16,
    'train_set_bag_cells': '32,128',
    'train_set_dim': 128,
    'train_set_depth': 2,
    'train_set_lr': 0.0003,
    'train_set_weight_decay': 0.05,
    'train_set_instance_weight': 0.1,
    'train_set_shuffle_control': False,
    'train_set_seed': 0,
    'train_set_threads': 2,
    'train_set_folds': '',
    'train_set_dst': '',
    'train_cell_crops': '',
    'train_cell_table': '',
    'train_cell_label_column': 'label',
    'train_cell_group_column': 'well',
    'train_cell_stains': '',
    'train_cell_test_fraction': 0.2,
    'train_cell_size': 96,
    'train_cell_vit_size': 'small',
    'train_cell_patch': 12,
    'train_cell_epochs': 10,
    'train_cell_batch': 64,
    'train_cell_lr': 0.0003,
    'pretrain_method': 'none',
    'pretrain_epochs': 0,
    'pretrain_mask_ratio': 0.75,
    'train_labels_per_class': 0,
    'eval_channel_subsets': '',
}

_EXPECTED_TYPES: Dict[str, Any] = {key: type(value) for key, value in _DEFAULTS.items()}

_TOOLTIPS: Dict[str, str] = {
    'train_arch': "(str) - The model trained: 'spacr_channel_cnn' (channel-adaptive CNN), 'spacr_channel_vit' (channel-aware ViT) or 'spacr_well_transformer' (set model over the cell embeddings of a well). The two image models train on crops (train_level 'cell'); the well transformer trains on cell embeddings (train_level 'well_set'). Default 'spacr_well_transformer'.",
    'train_level': "(str) - What one training example is: 'well_set', all cell embeddings of one well, or 'cell', one cell crop. Default 'well_set'.",
    'train_set_input': "(str) - Folder of per-plate cell embeddings, <plate>.npy with one row per cell, as the embedding profiles step writes them. Default blank.",
    'train_set_cells': "(str) - Folder of the matching <plate>.cells.parquet tables (plate, well, compound, concentration, moa), in the same row order as the embeddings. Blank looks in train_set_input. Default blank.",
    'train_set_tag': "(str) - Name of the embedding, written to the summary (such as resnet18 or openphenom). Default 'resnet18'.",
    'train_set_normalisation': "(str) - How the embeddings are normalised before training, without labels: 'plate_control_pca' centres and scales each dimension by the plate's control cells, then whitens with a PCA fitted on control cells of all plates; 'plate_control' stops after the plate step; 'none' keeps them as they are. Default 'plate_control_pca'.",
    'train_set_pca_dims': "(int) - Dimensions kept by the PCA whitening of 'plate_control_pca'. Default 256.",
    'train_set_control': "(str) - MoA value of the negative-control cells, used for the normalisation and for the penetrance thresholds; control wells are not scored. Default 'DMSO'.",
    'train_set_max_cells': "(int) - Most cells of a well used at prediction, the first ones in table order. Default 300.",
    'train_protocol': "(str) - Folds: 'NSCB' (not same compound or batch: one fold per imaging week, and taxol at 1 and 3 uM) or 'NSC' (not same compound: four folds, every compound of a MoA in another fold). Default 'NSCB'.",
    'train_set_arms': "(str) - Comma-separated arms: settf (set transformer), abmil (attention-MIL only), n1 (one cell at a time, the no-context ablation), mixbag (bags from random wells of the same class), mean_lr (mean profile, logistic regression), mean_knn (mean profile, cosine 1-NN). Default 'settf,n1,mean_lr,mean_knn'.",
    'train_set_steps': "(int) - Training steps per fold of the transformer arms. Default 800.",
    'train_set_bag_batch': "(int) - Wells per training step, drawn class-uniformly. Default 16.",
    'train_set_bag_cells': "(str) - Smallest and largest number of cells in a training bag, comma-separated. Default '32,128'.",
    'train_set_dim': "(int) - Model dimension of the well transformer. Default 128.",
    'train_set_depth': "(int) - ISAB blocks of the set encoder (abmil uses 0). Default 2.",
    'train_set_lr': "(float) - AdamW learning rate of the well transformer, with a 5% warm-up and a cosine decay. Default 0.0003.",
    'train_set_weight_decay': "(float) - AdamW weight decay. Default 0.05.",
    'train_set_instance_weight': "(float) - Weight of the per-cell loss beside the well loss; it trains the instance head behind the penetrant fraction. Default 0.1.",
    'train_set_shuffle_control': "(bool) - Also train every arm on shuffled labels (well sets: the compound-to-MoA map permuted with seed 1; cells: the training labels permuted): the control that shows what chance alone gives. Default False.",
    'train_set_seed': "(int) - Seed of the model initialisation, the bag or batch draws, the cell split and the few-shot draw. Default 0.",
    'train_set_threads': "(int) - CPU threads of the training; results are the same only at the same thread count. Default 2.",
    'train_set_folds': "(str) - Comma-separated fold numbers to run, for a quick check; blank runs every fold. Default blank.",
    'train_set_dst': "(str) - Folder the results and summary.csv are written to. Blank writes beside the input. Default blank.",
    'train_cell_crops': "(str) - Single-cell crops for train_level 'cell': one .npy array of cells x channels x height x width, raw intensities. Default blank.",
    'train_cell_table': "(str) - Table with one row per crop, in the same order, holding the label and group columns. CSV or Parquet. Default blank.",
    'train_cell_label_column': "(str) - Column with each cell's class. Rows without one are not used. Default 'label'.",
    'train_cell_group_column': "(str) - Column that groups cells for the split, such as the well, so no group is in both training and test. Default 'well'.",
    'train_cell_stains': "(str) - Comma-separated stain of each channel, such as dna,tubulin,actin. The models look stains up by name, so the same weights work on another panel. Blank names them other_0, other_1 and so on. Default blank.",
    'train_cell_test_fraction': "(float) - Share of the groups held out for testing. Default 0.2.",
    'train_cell_size': "(int) - Size the crops are resized to; a multiple of 32 for the CNN and of train_cell_patch for the ViT. Default 96.",
    'train_cell_vit_size': "(str) - ViT size: 'mini', 'tiny' or 'small'. Default 'small'.",
    'train_cell_patch': "(int) - ViT patch size in pixels. Default 12.",
    'train_cell_epochs': "(int) - Supervised epochs over the labelled training cells. Default 10.",
    'train_cell_batch': "(int) - Cells per training batch. Default 64.",
    'train_cell_lr': "(float) - AdamW learning rate of the cell models, with a 5% warm-up and a cosine decay. Default 0.0003.",
    'pretrain_method': "(str) - Self-supervised pre-training of the ViT before the labels are used: 'none', 'mae' (masked autoencoder, with whole stains masked) or 'dino' (self-distillation, the student seeing random stain subsets). Only the training cells are used, without labels. Default 'none'.",
    'pretrain_epochs': "(int) - Pre-training epochs over the training cells. Default 0.",
    'pretrain_mask_ratio': "(float) - Share of the patch tokens the MAE masks. Default 0.75.",
    'train_labels_per_class': "(int) - Labelled training cells kept per class, for few-shot training after pre-training; 0 keeps all. Default 0.",
    'eval_channel_subsets': "(str) - Stain subsets each model is also scored on, as if the other stains were missing: subsets separated by semicolons, stains by commas, such as dna,tubulin;dna. Blank scores the full panel only. Default blank.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Architecture data': ['train_arch', 'train_level', 'train_set_input', 'train_set_cells', 'train_set_tag',
                          'train_set_normalisation', 'train_set_pca_dims', 'train_set_control',
                          'train_set_max_cells', 'train_protocol', 'train_cell_crops', 'train_cell_table',
                          'train_cell_label_column', 'train_cell_group_column', 'train_cell_stains',
                          'train_cell_test_fraction', 'train_set_dst'],
    'Architecture training': ['train_set_arms', 'train_set_steps', 'train_set_bag_batch', 'train_set_bag_cells',
                              'train_set_dim', 'train_set_depth', 'train_set_lr', 'train_set_weight_decay',
                              'train_set_instance_weight', 'train_cell_size', 'train_cell_vit_size',
                              'train_cell_patch', 'train_cell_epochs', 'train_cell_batch', 'train_cell_lr',
                              'pretrain_method', 'pretrain_epochs', 'pretrain_mask_ratio',
                              'train_labels_per_class', 'eval_channel_subsets', 'train_set_shuffle_control',
                              'train_set_seed', 'train_set_threads', 'train_set_folds'],
}

_DESCRIPTION = (
    "Train spaCR's channel-adaptive models for any stain panel: the well "
    "transformer on cell embeddings with its ablations and controls, or the "
    "channel CNN and ViT on crops with MAE or DINO pre-training, few-shot "
    "labels and the missing-stain evaluation."
)


def architecture_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_spacr_architectures`, filled in with their defaults.

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
    register_defaults(APP_KEY, architecture_defaults, expected_types=_EXPECTED_TYPES, tooltips=_TOOLTIPS,
                      categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def channel_subsets(value: Any) -> List[List[str]]:
    """Parse ``eval_channel_subsets``: subsets separated by ``;``, stains by ``,``.

    :param value: the setting, or a list of lists.
    :returns: the non-empty subsets.
    """
    if not value:
        return []
    if isinstance(value, str):
        parts = [p for p in value.split(';')]
        return [[s.strip() for s in p.split(',') if s.strip()] for p in parts if p.strip()]
    return [[str(s) for s in sub] for sub in value if sub]


def fixed_channel_scale(crops: np.ndarray, rows: np.ndarray, sample: int = 2048, seed: int = 0) -> np.ndarray:
    """Per-channel scale: the 99th percentile of up to ``sample`` crops of ``rows``.

    :param crops: crops shaped ``(cells, channels, height, width)``.
    :param rows: the crops the scale may look at (the training cells).
    :param sample: most crops read.
    :param seed: seed of the draw.
    :returns: one positive scale per channel.
    """
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(rows, min(sample, len(rows)), replace=False))
    x = np.asarray(crops[pick], np.float32)
    scale = np.percentile(x.transpose(1, 0, 2, 3).reshape(x.shape[1], -1), 99, axis=1)
    return np.where(scale > 0, scale, 1.0).astype(np.float32)


def prepare_batch(crops: np.ndarray, rows: np.ndarray, scale: np.ndarray, size: int, *,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Turn raw crops into model input, with augmentation when a generator is given.

    ``((x / scale).clip(0, 1.5) - 0.5) / 0.25``, bilinear resize to ``size``;
    with ``generator``, a random flip, a random 90-degree rotation (one per
    batch) and a per-channel gain within 10% (per crop).

    :param crops: crops shaped ``(cells, channels, height, width)``.
    :param rows: the crops of the batch.
    :param scale: the per-channel scale (:func:`fixed_channel_scale`).
    :param size: output size.
    :param generator: torch generator of the augmentation; None for none.
    :returns: the ``(len(rows), channels, size, size)`` float tensor.
    """
    x = torch.from_numpy(np.asarray(crops[np.asarray(rows)], np.float32))
    x = x / torch.from_numpy(scale)[None, :, None, None]
    if generator is not None:
        if bool(torch.rand(1, generator=generator) < 0.5):
            x = x.flip(-1)
        x = torch.rot90(x, int(torch.randint(0, 4, (1,), generator=generator)), (-2, -1))
        x = x * (0.9 + 0.2 * torch.rand(x.shape[0], x.shape[1], 1, 1, generator=generator))
    x = (x.clamp(0, 1.5) - 0.5) / 0.25
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    return x


def split_by_group(groups: Sequence[Any], test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split cells into training and test rows so that no group is in both.

    :param groups: the group of each cell (such as its well).
    :param test_fraction: share of the groups held out, at least one.
    :param seed: seed of the permutation of the sorted groups.
    :returns: the training rows and the test rows.
    :raises ValueError: fewer than two groups.
    """
    g = np.asarray(groups).astype(str)
    uniq = np.unique(g)
    if len(uniq) < 2:
        raise ValueError('the cells need at least two groups to split into training and test')
    rng = np.random.default_rng(seed)
    n_test = min(len(uniq) - 1, max(1, int(round(test_fraction * len(uniq)))))
    test_groups = set(rng.permutation(uniq)[:n_test])
    is_test = np.array([v in test_groups for v in g])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def few_shot_rows(rows: np.ndarray, labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Keep at most ``per_class`` training cells of each class.

    :param rows: the training rows.
    :param labels: the class index of every cell.
    :param per_class: cells kept per class; 0 or less keeps all.
    :param seed: seed of the draw.
    :returns: the kept rows, sorted.
    """
    if per_class <= 0:
        return np.asarray(rows)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels[rows]):
        r = rows[labels[rows] == c]
        keep.append(rng.choice(r, min(per_class, len(r)), replace=False))
    return np.sort(np.concatenate(keep))


def _random_resized_crop(x: torch.Tensor, g: torch.Generator, lo: float = 0.6) -> torch.Tensor:
    """One random resized crop for the whole batch (DINO views)."""
    H = x.shape[-1]
    s = math.sqrt(lo + (1 - lo) * float(torch.rand(1, generator=g)))
    h = max(8, int(H * s))
    y0, x0 = (int(v) for v in torch.randint(0, H - h + 1, (2,), generator=g))
    return F.interpolate(x[..., y0:y0 + h, x0:x0 + h], size=(H, H), mode='bilinear', align_corners=False)


def pretrain_encoder(enc: SpacrChannelViT, crops: np.ndarray, rows: np.ndarray, scale: np.ndarray, *,
                     method: str, epochs: int, batch: int = 64, mask_ratio: float = 0.75, lr: float = 1.5e-4,
                     seed: int = 0, size: Optional[int] = None) -> List[float]:
    """Pre-train a channel ViT without labels, in place.

    MAE: the encoder's own channel dropout is off (the MAE masks whole
    stains itself). DINO: an EMA teacher (momentum 0.996 rising to 1 on a
    cosine) sees two random resized views; the student also drops stains.
    AdamW with the reference learning rate scaled by ``batch / 256`` and
    gradient clipping at 3.

    :param enc: the encoder; its weights are updated (DINO keeps the student's).
    :param crops: raw crops shaped ``(cells, channels, height, width)``.
    :param rows: the cells used (labels are never read).
    :param scale: the per-channel scale.
    :param method: ``'mae'`` or ``'dino'``.
    :param epochs: passes over ``rows``.
    :param batch: cells per step.
    :param mask_ratio: MAE masking ratio.
    :param lr: base learning rate for a batch of 256.
    :param seed: seed of the batches and the augmentation.
    :param size: input size; None uses the encoder's.
    :returns: the loss of every step.
    :raises ValueError: an unknown method.
    """
    if method not in ('mae', 'dino'):
        raise ValueError(f"pretrain_method={method!r} is not 'mae' or 'dino'")
    size = int(size or enc.img)
    if method == 'mae':
        enc.channel_dropout = False
        model = ChannelMAE(enc, mask_ratio=mask_ratio)
    else:
        teacher = SpacrChannelViT(stains=enc.stains, img=enc.img, patch=enc.patch, dim=enc.dim,
                                  depth=len(enc.blocks), heads=enc.blocks[0].heads, bag_of_channels=enc.boc)
        teacher.channel_dropout = False
        model = ChannelDINO(enc, teacher, out=1024)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr * batch / 256, betas=(0.9, 0.95 if method == 'mae' else 0.999),
                            weight_decay=0.05)
    steps = max(1, int(math.ceil(epochs * len(rows) / batch)))
    sc = cosine_schedule(opt, steps)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for step in range(steps):
        b = rng.choice(rows, min(batch, len(rows)), replace=False)
        x = prepare_batch(crops, b, scale, size, generator=g)
        if method == 'mae':
            loss = model(x, enc.stains)
        else:
            loss = model(_random_resized_crop(x, g), _random_resized_crop(x, g), enc.stains)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 3.0)
        opt.step()
        sc.step()
        if method == 'dino':
            model.ema(1 - (1 - 0.996) * (math.cos(math.pi * step / steps) + 1) / 2)
        losses.append(float(loss.detach()))
    enc.channel_dropout = True
    return losses


def train_cell_classifier(model: ChannelClassifier, crops: np.ndarray, rows: np.ndarray, labels: np.ndarray,
                          scale: np.ndarray, *, size: int, epochs: int, batch: int, lr: float,
                          seed: int) -> List[float]:
    """Train a :class:`spacr.channel_models.ChannelClassifier` on labelled cells, in place.

    Batches are drawn with replacement, class-balanced; cross-entropy with
    label smoothing 0.1, AdamW (weight decay 0.05), :func:`spacr.well_set_training.cosine_schedule`
    and gradient clipping at 1.

    :param model: the classifier.
    :param crops: raw crops.
    :param rows: the labelled training cells.
    :param labels: the class index of every cell.
    :param scale: the per-channel scale.
    :param size: input size.
    :param epochs: passes over ``rows``.
    :param batch: cells per step.
    :param lr: learning rate.
    :param seed: seed of the batches and the augmentation.
    :returns: the loss of every step.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = max(1, int(math.ceil(epochs * len(rows) / batch)))
    sc = cosine_schedule(opt, steps)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    classes = np.unique(labels[rows])
    by_cls = [rows[labels[rows] == c] for c in classes]
    losses = []
    model.train()
    for _ in range(steps):
        ks = rng.integers(0, len(classes), batch)
        b = np.array([rng.choice(by_cls[k]) for k in ks])
        x = prepare_batch(crops, b, scale, size, generator=g)
        loss = F.cross_entropy(model(x), torch.from_numpy(labels[b]).long(), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sc.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def predict_cells(model: ChannelClassifier, crops: np.ndarray, rows: np.ndarray, scale: np.ndarray, *,
                  size: int, subset: Optional[Sequence[str]] = None, batch: int = 256) -> np.ndarray:
    """Predicted class of each cell, optionally with only a subset of the stains.

    :param model: the trained classifier.
    :param crops: raw crops.
    :param rows: the cells to predict.
    :param scale: the per-channel scale.
    :param size: input size.
    :param subset: stains kept (the others treated as missing); None keeps the training panel.
    :param batch: cells per forward pass.
    :returns: the class index of each cell.
    """
    saved = model.eval_subset
    model.set_subsets([model.stains[i] for i in model.train_subset] if model.train_subset else None, subset)
    model.eval()
    try:
        out = [model(prepare_batch(crops, rows[s:s + batch], scale, size)).argmax(1).numpy()
               for s in range(0, len(rows), batch)]
    finally:
        model.eval_subset = saved
    return np.concatenate(out) if out else np.array([], int)


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Macro-averaged F1 over the classes present in ``y``."""
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, labels=np.unique(y), average='macro', zero_division=0))


def _cell_model(s: Mapping[str, Any], stains: List[str], n_cls: int) -> ChannelClassifier:
    """The classifier ``train_arch`` names, at the settings' size."""
    arch = s['train_arch']
    if arch == 'spacr_channel_vit':
        opts = {'size': s['train_cell_vit_size'], 'patch': int(s['train_cell_patch'])}
    else:
        opts = {}
    return build_channel_model(arch, num_classes=n_cls, channels=len(stains), stains=stains,
                               image_size=int(s['train_cell_size']), options=opts)


def run_cell_architecture(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Train and score a channel-adaptive CNN or ViT on cell crops from settings.

    Writes ``<dst>/summary.csv`` (one row per model and stain subset, the
    shuffled-label control when asked, and the majority-class control),
    ``glossary.json``, ``config.json``, ``losses.json`` and ``model.pt`` (the
    trained classifier's state dict).

    :param settings: the keys of :func:`architecture_defaults`.
    :returns: dict with ``summary`` (the frame) and ``paths``.
    :raises ValueError: a blank input, an architecture that is not an image
        model, pre-training on the CNN, or a crops/table length mismatch.
    """
    from .tabular import write_table

    s = architecture_defaults(settings)
    arch = s['train_arch']
    if arch not in CHANNEL_IMAGE_ARCHITECTURES:
        raise ValueError(f"train_level 'cell' trains {list(CHANNEL_IMAGE_ARCHITECTURES)}, not {arch!r}")
    method = str(s['pretrain_method'] or 'none').lower()
    if method not in PRETRAIN_METHODS:
        raise ValueError(f"pretrain_method={method!r} is not one of {list(PRETRAIN_METHODS)}")
    if method != 'none' and arch != 'spacr_channel_vit':
        raise ValueError("pre-training (mae or dino) is built for spacr_channel_vit")
    if not s['train_cell_crops'] or not s['train_cell_table']:
        raise ValueError('train_cell_crops and train_cell_table name the crops and their table')
    crops = np.load(os.fspath(s['train_cell_crops']), mmap_mode='r')
    from .tabular import read_table

    table = read_table(s['train_cell_table'], canonicalise=False, report=None)
    if len(table) != len(crops):
        raise ValueError(f"{len(table)} table rows for {len(crops)} crops")
    keep = table[s['train_cell_label_column']].notna().to_numpy()
    classes = sorted(table.loc[keep, s['train_cell_label_column']].astype(str).unique())
    labels = np.full(len(table), -1)
    labels[keep] = table.loc[keep, s['train_cell_label_column']].astype(str).map(
        {c: i for i, c in enumerate(classes)}).to_numpy()
    seed = int(s['train_set_seed'])
    tr_all, te = split_by_group(table[s['train_cell_group_column']].to_numpy(), float(s['train_cell_test_fraction']),
                                seed)
    tr_all, te = tr_all[labels[tr_all] >= 0], te[labels[te] >= 0]
    tr = few_shot_rows(tr_all, labels, int(s['train_labels_per_class']), seed)
    stains = default_stains(crops.shape[1], [x.strip() for x in str(s['train_cell_stains'] or '').split(',') if x.strip()] or None)
    subsets = [None] + channel_subsets(s['eval_channel_subsets'])
    for sub in subsets[1:]:
        missing = [x for x in sub if x not in stains]
        if missing:
            raise ValueError(f"eval_channel_subsets names {missing}, not among the stains {stains}")
    size = int(s['train_cell_size'])
    scale = fixed_channel_scale(crops, tr_all, seed=seed)
    dst = Path(os.fspath(s['train_set_dst'] or Path(os.fspath(s['train_cell_crops'])).parent / 'cell_architecture'))
    dst.mkdir(parents=True, exist_ok=True)
    threads = torch.get_num_threads()
    torch.set_num_threads(int(s['train_set_threads']))
    rows, losses = [], {}
    arm = f"{arch}_{method}" + (f"_{int(s['train_labels_per_class'])}pc" if int(s['train_labels_per_class']) > 0
                                else '')
    try:
        for shuffled in ([False, True] if s['train_set_shuffle_control'] else [False]):
            torch.manual_seed(seed)
            y = labels.copy()
            if shuffled:
                y[tr] = np.random.default_rng(seed + 1).permutation(y[tr])
            model = _cell_model(s, stains, len(classes))
            if method != 'none':
                losses[f'pretrain{"_shuffled" if shuffled else ""}'] = pretrain_encoder(
                    model.enc, crops, tr_all, scale, method=method, epochs=int(s['pretrain_epochs']),
                    batch=int(s['train_cell_batch']), mask_ratio=float(s['pretrain_mask_ratio']), seed=seed,
                    size=size)
            losses[f'train{"_shuffled" if shuffled else ""}'] = train_cell_classifier(
                model, crops, tr, y, scale, size=size, epochs=int(s['train_cell_epochs']),
                batch=int(s['train_cell_batch']), lr=float(s['train_cell_lr']), seed=seed)
            for sub in subsets:
                p = predict_cells(model, crops, te, scale, size=size, subset=sub)
                rows.append({'arm': arm + ('_shuffled' if shuffled else ''), 'arch': arch, 'pretrain': method,
                             'labels_per_class': int(s['train_labels_per_class']), 'shuffled': shuffled,
                             'channels': ','.join(sub or stains), 'n_train': int(len(tr)), 'n_test': int(len(te)),
                             'accuracy': float((p == labels[te]).mean()), 'macro_f1': _macro_f1(labels[te], p),
                             'seed': seed})
            if not shuffled:
                torch.save(model.state_dict(), dst / 'model.pt')
    finally:
        torch.set_num_threads(threads)
    major = np.bincount(labels[tr], minlength=len(classes)).argmax()
    rows.append({'arm': 'majority_class', 'arch': '', 'pretrain': '', 'labels_per_class': int(s['train_labels_per_class']),
                 'shuffled': False, 'channels': ','.join(stains), 'n_train': int(len(tr)), 'n_test': int(len(te)),
                 'accuracy': float((labels[te] == major).mean()),
                 'macro_f1': _macro_f1(labels[te], np.full(len(te), major)), 'seed': seed})
    table_out = pd.DataFrame(rows, columns=list(CELL_SUMMARY_COLUMNS))
    paths = [write_table(table_out, dst / 'summary.csv', canonicalise=False)]
    for name, obj in (('glossary.json', CELL_GLOSSARY), ('losses.json', losses),
                      ('config.json', {k: s[k] for k in _DEFAULTS} | {'classes': classes, 'stains': stains,
                                                                      'scale': scale.tolist()})):
        (dst / name).write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8')
        paths.append(str(dst / name))
    paths.append(str(dst / 'model.pt'))
    return {'summary': table_out, 'paths': paths}


def run_spacr_architectures(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the level ``train_level`` names from settings.

    :param settings: the keys of :func:`architecture_defaults`.
    :returns: the result of :func:`spacr.well_set_training.run_well_set_training`
        (``'well_set'``) or of :func:`run_cell_architecture` (``'cell'``).
    :raises ValueError: an unknown level or architecture.
    """
    s = architecture_defaults(settings)
    if s['train_arch'] not in CHANNEL_ARCHITECTURES:
        raise ValueError(f"train_arch={s['train_arch']!r} is not one of {list(CHANNEL_ARCHITECTURES)}")
    if s['train_level'] == 'well_set':
        from .well_set_training import run_well_set_training

        return run_well_set_training(s)
    if s['train_level'] == 'cell':
        return run_cell_architecture(s)
    raise ValueError(f"train_level={s['train_level']!r} is not one of {list(TRAIN_LEVELS)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run :func:`run_spacr_architectures` from a settings file.

    ``python -m spacr.architecture_training --settings file.csv`` reads the
    settings file with :func:`spacr.cli.load_settings_file`.

    :param argv: the arguments; None reads :data:`sys.argv`.
    :returns: 0 when a summary was written, 1 otherwise.
    """
    from .cli import load_settings_file

    parser = argparse.ArgumentParser(prog='python -m spacr.architecture_training', description=_DESCRIPTION)
    parser.add_argument('--settings', required=True)
    args = parser.parse_args(argv)
    t0 = time.time()
    result = run_spacr_architectures(load_settings_file(args.settings))
    logger.info('wrote %s in %.1f s', result['paths'][0], time.time() - t0)
    return 0 if len(result['summary']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
