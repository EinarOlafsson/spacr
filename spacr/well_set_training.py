"""Train and score the spaCR well transformer on per-cell embeddings, with its ablations and controls.

A well is a set of single-cell embeddings (such as the per-channel ResNet-18
embeddings :mod:`spacr.embedding_profiles` writes, ``<plate>.npy`` beside
the crop index ``<plate>.cells.parquet``). This module trains the
:class:`spacr.channel_models.SpacrWellTransformer` on those sets, the way the
BMC reanalysis ``train_well.py`` did, and scores it on BBBC021-style
mechanism-of-action (MoA) folds:

1. :func:`load_cell_embeddings` reads the embeddings and normalises them
   without labels (``train_set_normalisation``): ``'plate_control_pca'``
   centres and scales every dimension per plate by that plate's control
   (DMSO) cells, then whitens with a PCA fitted on control cells of all
   plates (typical-variation-normalisation-like, Ando et al. 2017); the
   control is the negative control on every plate, so using it on test
   plates is not label leakage. ``'plate_control'`` stops after the plate
   step and ``'none'`` keeps the embeddings as they are.
2. :func:`group_wells` groups the cells into wells, and :func:`moa_folds`
   gives the folds of ``train_protocol``: ``'NSCB'`` (not same compound or
   batch: one fold per batch, plus the taxol 1 and 3 uM fold) or ``'NSC'``
   (not same compound: four folds, every compound of a MoA in another fold).
3. :func:`train_arrayed` trains one arm per fold, and :func:`predict_arrayed`
   predicts the test wells. Arms (:data:`WELL_SET_ARMS`): ``settf`` (ISAB set
   encoder and gated attention-MIL), ``abmil`` (attention-MIL only), ``n1``
   (set size 1 in training and test, the well being the mean of its cells'
   softmax: the no-context ablation), ``mixbag`` (bags drawn from cells of
   random training wells of the same class, which keeps the label but breaks
   within-well co-occurrence), ``mean_lr`` (mean profile and logistic
   regression) and ``mean_knn`` (mean profile and cosine 1-NN on treatment
   profiles, the classic BBBC021 baseline).
4. :func:`score_treatments` scores the wells as the reference
   ``eval_bbbc.py`` does: well softmax with untrained classes zeroed,
   treatment mean, argmax; accuracy with a bootstrap confidence interval
   over treatments, macro AUROC and macro AP.

:func:`run_well_set_arm` runs one arm from a config dict;
:func:`run_well_set_training` runs every arm of ``train_set_arms`` from the
settings of :mod:`spacr.architecture_training`, with the shuffled-label control (``train_set_shuffle_control``: the
compound to MoA map is permuted) and the majority-class control, and writes
``summary.csv`` with one row per arm and control. Synthetic data with known
penetrance (:func:`synthetic_arrayed`) and with a well batch effect
(:func:`synthetic_pooled`) check the arrayed and the pooled mode on the CPU.

Everything is seeded: ``train_set_seed`` seeds torch once per arm before the
first fold, and numpy generators seed the bag draws, so a rerun with the same
thread count gives the same table.
"""
from __future__ import annotations

import functools
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .channel_models import SpacrWellTransformer, count_params

logger = logging.getLogger(__name__)

#: The arrayed arms :func:`run_well_set_arm` runs.
WELL_SET_ARMS = ('settf', 'abmil', 'n1', 'mixbag', 'mean_lr', 'mean_knn')

#: The arms that train a :class:`spacr.channel_models.SpacrWellTransformer`.
TRANSFORMER_ARMS = ('settf', 'abmil', 'n1', 'mixbag')

#: The pooled arms of :func:`run_synthetic_pooled`: context from the same well, none, or another well.
POOLED_ARMS = {'ctx': 'same', 'noctx': 'none', 'ctxshuf': 'other'}

#: How the embeddings are normalised before training.
NORMALISATIONS = ('plate_control_pca', 'plate_control', 'none')

#: The fold protocols of :func:`moa_folds`.
PROTOCOLS = ('NSCB', 'NSC')

#: The columns of ``summary.csv``, in order.
SUMMARY_COLUMNS = ('run', 'design', 'emb_tag', 'arm', 'shuffled', 'seed', 'accuracy', 'acc_lo', 'acc_hi',
                   'macro_auroc', 'macro_ap', 'majority_class_accuracy', 'median_penetrant_fraction',
                   'n_treatments', 'params', 'wall_s')

#: What each abbreviation and column of ``summary.csv`` means.
GLOSSARY: Dict[str, str] = {
    'MoA': 'mechanism of action',
    'NSC': 'not same compound: the test compound is never in training',
    'NSCB': 'not same compound or batch: as NSC, and the test batch (imaging week) is not in training',
    'settf': 'set transformer: ISAB set encoder and gated attention-MIL pooling over all cells of a well',
    'abmil': 'attention-based multiple-instance learning: the set transformer without the ISAB encoder',
    'n1': 'one cell at a time, the well being the mean of its cells\' softmax (no-context ablation)',
    'mixbag': 'bags drawn from cells of random training wells of the same class (within-well '
              'co-occurrence broken)',
    'mean_lr': 'mean cell profile per well and logistic regression',
    'mean_knn': 'mean cell profile per well, cosine 1-NN between treatment profiles',
    'shuffled': 'True for the shuffled-label control: compound to MoA map permuted with seed 1',
    'majority_class': 'control row: always predicting the commonest MoA',
    'accuracy': 'treatments whose predicted MoA (argmax of the mean well probability) is right',
    'acc_lo': 'lower end of the 95% bootstrap interval of the accuracy over treatments (1,000 draws)',
    'acc_hi': 'upper end of the 95% bootstrap interval',
    'macro_auroc': 'area under the ROC curve per MoA, averaged over MoAs',
    'macro_ap': 'average precision per MoA, averaged over MoAs',
    'majority_class_accuracy': 'accuracy of always predicting the commonest MoA',
    'median_penetrant_fraction': 'median over test wells of the share of cells whose instance '
                                 'probability for the called MoA exceeds the 95th percentile of '
                                 'control cells',
    'params': 'parameters of the trained model',
    'wall_s': 'wall time in seconds',
}

def _names(value: Any) -> List[str]:
    """A list of names from a comma-separated string or a sequence."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(',') if part.strip()]
    if isinstance(value, (int, float)):
        return [str(value)]
    return [str(part) for part in value]


def _read(path: Any) -> pd.DataFrame:
    """Read a table exactly as written."""
    from .tabular import read_table

    return read_table(path, canonicalise=False, report=None)


def normalise_embeddings(blocks: Sequence[np.ndarray], tables: Sequence[pd.DataFrame], *,
                         method: str = 'plate_control_pca', pca_dims: int = 256, control: str = 'DMSO',
                         moa_column: str = 'moa', sample: int = 50000, seed: int = 0) -> np.ndarray:
    """Normalise per-plate cell embeddings without labels.

    :param blocks: one ``(cells, dims)`` array per plate.
    :param tables: the matching cell tables, with ``moa_column``.
    :param method: one of :data:`NORMALISATIONS`.
    :param pca_dims: dimensions kept by the PCA whitening.
    :param control: value of ``moa_column`` of the control cells.
    :param moa_column: column naming each cell's MoA.
    :param sample: at most this many control cells fit the PCA (drawn with ``seed``).
    :param seed: seed of that draw.
    :returns: the concatenated embeddings, float16 after the PCA and float32 otherwise.
    :raises ValueError: an unknown method, a table and block of different lengths, or a plate without control cells.
    """
    if method not in NORMALISATIONS:
        raise ValueError(f"train_set_normalisation={method!r} is not one of {list(NORMALISATIONS)}")
    xs = []
    for X, c in zip(blocks, tables):
        X = np.asarray(X).astype(np.float32)
        if len(c) != len(X):
            raise ValueError(f"{len(c)} table rows for {len(X)} embeddings")
        if method == 'none':
            xs.append(X)
            continue
        d = X[(c[moa_column] == control).to_numpy()]
        if not len(d):
            raise ValueError(f"a plate has no {control!r} cells to normalise by")
        mu, sd = d.mean(0), d.std(0)
        sd[sd < 1e-6] = 1.0
        xs.append(((X - mu) / sd).astype(np.float32))
    X = np.concatenate(xs)
    if method != 'plate_control_pca':
        return X
    dm = np.concatenate([(c[moa_column] == control).to_numpy() for c in tables])
    rng = np.random.default_rng(seed)
    sub = rng.choice(np.flatnonzero(dm), min(sample, dm.sum()), replace=False)
    mu = X[sub].mean(0)
    _, S, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
    W = Vt[:pca_dims].T / (S[:pca_dims] / math.sqrt(len(sub) - 1) + 1e-3)
    return ((X - mu) @ W).astype(np.float16)


def load_cell_embeddings(embeddings: Any, cells: Any = None, *, method: str = 'plate_control_pca',
                         pca_dims: int = 256, control: str = 'DMSO') -> Tuple[pd.DataFrame, np.ndarray]:
    """Read per-plate embeddings and their cell tables, and normalise them.

    :param embeddings: folder of ``<plate>.npy``, read in sorted order.
    :param cells: folder of ``<plate>.cells.parquet``; None uses ``embeddings``.
    :param method: one of :data:`NORMALISATIONS`.
    :param pca_dims: dimensions kept by the PCA.
    :param control: MoA value of the control cells.
    :returns: the cell table (plate, well, compound, concentration, moa and
        ``week``, the plate name up to its first underscore) and the
        normalised embeddings, row-aligned.
    :raises FileNotFoundError: no ``.npy`` file in ``embeddings``.
    """
    emb = Path(os.fspath(embeddings))
    cdir = Path(os.fspath(cells)) if cells else emb
    files = sorted(emb.glob('*.npy'))
    if not files:
        raise FileNotFoundError(f"no <plate>.npy embeddings in {emb}")
    blocks, tables = [], []
    for f in files:
        c = _read(cdir / f"{f.stem}.cells.parquet")[['plate', 'well', 'compound', 'concentration', 'moa']]
        blocks.append(np.load(f))
        tables.append(c.reset_index(drop=True))
    Z = normalise_embeddings(blocks, tables, method=method, pca_dims=pca_dims, control=control)
    t = pd.concat(tables, ignore_index=True)
    t['week'] = t['plate'].str.split('_').str[0]
    return t, Z


def group_wells(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per well, in order of first appearance, with the row numbers of its cells.

    :param cells: cell table with plate, well, compound, concentration, moa and week.
    :returns: a well table with those columns, ``n`` (cells) and ``rows`` (an array of cell rows).
    """
    w = cells.groupby(['plate', 'well'], sort=False).agg(
        compound=('compound', 'first'), concentration=('concentration', 'first'), moa=('moa', 'first'),
        week=('week', 'first'), n=('moa', 'size')).reset_index()
    key = cells['plate'] + '|' + cells['well']
    order = {k: i for i, k in enumerate(w['plate'] + '|' + w['well'])}
    wid = key.map(order).to_numpy()
    rows = pd.Series(np.arange(len(cells))).groupby(wid).apply(np.asarray).to_dict()
    w['rows'] = [rows[i] for i in range(len(w))]
    return w


def moa_folds(wells: pd.DataFrame, protocol: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """The folds of a fold protocol over labelled wells, as the BBBC021 CNN reference defines them.

    ``'NSCB'``: one fold per batch (``week``, ordered by its number), testing
    that week's non-taxol wells and training on every other week; then a
    taxol fold, testing taxol at 1 and 3 uM and training on the non-taxol
    wells of the other weeks. ``'NSC'``: four folds; within each MoA the
    compounds are dealt to folds 0, 1, 2, 3 in a permutation drawn with seed 0.

    :param wells: labelled wells with compound, week, concentration and moa.
    :param protocol: one of :data:`PROTOCOLS` (``'NSC_A'`` is accepted for ``'NSC'``).
    :returns: ``(name, train mask, test mask)`` per fold.
    :raises ValueError: an unknown protocol.
    """
    comp, week, conc = wells['compound'].values, wells['week'].values, wells['concentration'].values
    taxol = comp == 'taxol'
    out = []
    if protocol == 'NSCB':
        for w in sorted(np.unique(week), key=lambda s: int(''.join(ch for ch in s if ch.isdigit()) or 0)):
            out.append((f'{w}', week != w, (week == w) & ~taxol))
        test = taxol & np.isin(conc, [1.0, 3.0])
        out.append(('taxol_1_3uM', ~taxol & ~np.isin(week, np.unique(week[test])), test))
    elif protocol in ('NSC', 'NSC_A'):
        cm = wells.groupby('compound').moa.first()
        rng = np.random.default_rng(0)
        assign = {}
        for _, g in cm.groupby(cm):
            for i, c in enumerate(rng.permutation(sorted(g.index))):
                assign[c] = i
        fold_of = np.array([assign[c] for c in comp])
        for k in range(4):
            out.append((f'A{k}', fold_of != k, fold_of == k))
    else:
        raise ValueError(f"train_protocol={protocol!r} is not one of {list(PROTOCOLS)}")
    return out


def _cosine_warmup(step: int, *, warm: int, steps: int) -> float:
    """Learning-rate factor: linear warm-up, then a cosine decay to zero."""
    return min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))


def cosine_schedule(opt: torch.optim.Optimizer, steps: int, warm_frac: float = 0.05):
    """A LambdaLR with a linear warm-up over ``warm_frac`` of the steps and a cosine decay.

    :param opt: the optimiser.
    :param steps: total steps.
    :param warm_frac: share of the steps spent warming up (at least one step).
    :returns: the scheduler.
    """
    warm = max(1, int(warm_frac * steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, functools.partial(_cosine_warmup, warm=warm, steps=steps))


def pad_bags(X: np.ndarray, bags: Sequence[np.ndarray], dev: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack bags of cell rows into one zero-padded batch.

    :param X: cell embeddings, one row per cell.
    :param bags: the cell rows of each bag.
    :param dev: torch device.
    :returns: the ``(B, N, D)`` batch and its ``(B, N)`` validity mask.
    """
    n = max(len(b) for b in bags)
    out = torch.zeros(len(bags), n, X.shape[1])
    m = torch.zeros(len(bags), n, dtype=torch.bool)
    for i, b in enumerate(bags):
        out[i, : len(b)] = torch.from_numpy(np.asarray(X[b], np.float32))
        m[i, : len(b)] = True
    return out.to(dev), m.to(dev)


def zero_untrained(P: np.ndarray, classes: Sequence[str], trained: set) -> np.ndarray:
    """Zero the classes a fold did not train on and renormalise each row.

    :param P: well probabilities, one column per class.
    :param classes: the class names of the columns.
    :param trained: the classes the fold trained on.
    :returns: the renormalised probabilities.
    """
    P = P.copy()
    P[:, [c not in trained for c in classes]] = 0
    return P / np.clip(P.sum(1, keepdims=True), 1e-12, None)


def score_treatments(wells: pd.DataFrame, P: np.ndarray, classes: Sequence[str], boot: int = 1000,
                     seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Score well probabilities per treatment.

    Wells are averaged per treatment (compound, concentration, MoA and fold),
    the treatment takes the MoA of its largest mean probability, and the
    accuracy gets a percentile bootstrap interval over treatments.

    :param wells: the scored wells with plate, well, compound, concentration, moa and fold.
    :param P: their probabilities, untrained classes already zeroed (:func:`zero_untrained`).
    :param classes: the class names of the columns of ``P``.
    :param boot: bootstrap draws.
    :param seed: seed of the bootstrap.
    :returns: the treatment table and a dict with accuracy, acc_ci, macro_auroc,
        macro_ap, n_treatments and majority_class_accuracy.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    classes = list(classes)
    df = wells[['plate', 'well', 'compound', 'concentration', 'moa', 'fold']].copy()
    df[classes] = P
    t = df.groupby(['compound', 'concentration', 'moa', 'fold'], as_index=False)[classes].mean()
    T = t[classes].to_numpy()
    t['pred'] = np.array(classes)[T.argmax(1)]
    hit = (t['pred'] == t['moa']).to_numpy()
    au, ap = [], []
    for c in classes:
        y = (t['moa'] == c).to_numpy()
        if 0 < y.sum() < len(y):
            au.append(roc_auc_score(y, t[c]))
            ap.append(average_precision_score(y, t[c]))
    rng = np.random.default_rng(seed)
    bs = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(boot)]
    return t, {'accuracy': float(hit.mean()),
               'acc_ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               'macro_auroc': float(np.mean(au)), 'macro_ap': float(np.mean(ap)), 'n_treatments': int(len(t)),
               'majority_class_accuracy': float(t['moa'].value_counts(normalize=True).iloc[0])}


def _quiet(**_kw: Any) -> None:
    """A log callback that drops its events."""
    return None


def _well_model(cfg: Mapping[str, Any], in_dim: int, n_cls: int) -> SpacrWellTransformer:
    """The arrayed well transformer of an arm's config."""
    return SpacrWellTransformer(in_dim, n_cls, 'arrayed', dim=cfg.get('dim', 128),
                                depth=0 if cfg['arm'] == 'abmil' else cfg.get('depth', 2), heads=cfg.get('heads', 4),
                                inducing=cfg.get('inducing', 16), pool_heads=cfg.get('pool_heads', 4),
                                dropout=cfg.get('dropout', 0.1), feat_dropout=cfg.get('feat_dropout', 0.1))


def train_arrayed(cfg: Mapping[str, Any], X: np.ndarray, w_rows: Sequence[np.ndarray], y_well: np.ndarray,
                  n_cls: int, train_w: np.ndarray, dev: Any, log: Callable[..., None] = _quiet,
                  mixbag_pool: Optional[Sequence[np.ndarray]] = None) -> SpacrWellTransformer:
    """Train one arrayed arm on the training wells of a fold.

    Every step draws ``bag_batch`` classes uniformly, a training well of each
    and a bag of ``bag_cells`` cells of it (``n1``: one cell; ``mixbag``:
    cells from ``mixbag_pool``). The loss is the label-smoothed well
    cross-entropy plus ``instance_weight`` times the per-cell cross-entropy,
    with AdamW, gradient clipping at 1 and :func:`cosine_schedule`.

    :param cfg: the arm config: arm, steps, bag_batch, bag_cells, dim, depth,
        lr, weight_decay, instance_weight and seed.
    :param X: cell embeddings.
    :param w_rows: the cell rows of each well.
    :param y_well: the class of each well.
    :param n_cls: classes.
    :param train_w: the training wells.
    :param dev: torch device.
    :param log: called with keyword events every 250 steps.
    :param mixbag_pool: per class, the cell rows of its training wells (``mixbag`` only).
    :returns: the trained model.
    """
    arm = cfg['arm']
    model = _well_model(cfg, X.shape[1], n_cls).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 3e-4), weight_decay=cfg.get('weight_decay', 0.05))
    steps, B = cfg.get('steps', 1500), cfg.get('bag_batch', 16)
    sc = cosine_schedule(opt, steps)
    rng = np.random.default_rng(cfg.get('seed', 0))
    by_cls = [train_w[y_well[train_w] == k] for k in range(n_cls)]
    by_cls_ok = [k for k in range(n_cls) if len(by_cls[k])]
    lo, hi = cfg.get('bag_cells', [32, 128])
    lam = cfg.get('instance_weight', 0.1)
    model.train()
    for step in range(steps):
        ks = rng.choice(by_cls_ok, B)
        bags, ys = [], []
        for k in ks:
            if arm == 'n1':
                wi = rng.choice(by_cls[k])
                bags.append(rng.choice(w_rows[wi], 1))
            elif arm == 'mixbag':
                n = int(rng.integers(lo, hi + 1))
                bags.append(rng.choice(mixbag_pool[k], n))
            else:
                wi = rng.choice(by_cls[k])
                r = w_rows[wi]
                n = min(len(r), int(rng.integers(lo, hi + 1)))
                bags.append(rng.choice(r, n, replace=False))
            ys.append(k)
        x, m = pad_bags(X, bags, dev)
        yt = torch.tensor(ys, device=dev)
        o = model(x, m)
        loss = F.cross_entropy(o['logits'], yt, label_smoothing=0.1)
        if lam:
            cl = o['cell_logits'][m]
            loss = loss + lam * F.cross_entropy(cl, yt[:, None].expand_as(m)[m], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sc.step()
        if (step + 1) % 250 == 0:
            log(event='step', step=step + 1, loss=round(float(loss.detach()), 4))
    return model


@torch.no_grad()
def predict_arrayed(model: SpacrWellTransformer, X: np.ndarray, w_rows: Sequence[np.ndarray],
                    wells: Sequence[int], dev: Any, arm: str, max_cells: int = 300,
                    chunk: int = 32) -> Tuple[np.ndarray, List[Optional[np.ndarray]], List[np.ndarray]]:
    """Predict wells with a trained arrayed arm.

    :param model: the trained model.
    :param X: cell embeddings.
    :param w_rows: the cell rows of each well.
    :param wells: the wells to predict.
    :param dev: torch device.
    :param arm: the arm; ``n1`` scores each cell alone and averages the softmax.
    :param max_cells: the first this many cells of a well are used.
    :param chunk: wells per forward pass.
    :returns: the well probabilities, the per-well cell attention (None for
        ``n1``) and the per-well cell probabilities.
    """
    model.eval()
    P, A, CP = [], [], []
    for s0 in range(0, len(wells), chunk):
        ws = wells[s0:s0 + chunk]
        if arm == 'n1':
            for wi in ws:
                r = w_rows[wi][:max_cells]
                x = torch.from_numpy(np.asarray(X[r], np.float32)).to(dev)[:, None]
                p = torch.softmax(model(x, torch.ones(len(r), 1, dtype=torch.bool, device=dev))['logits'], 1)
                P.append(p.mean(0).cpu().numpy())
                A.append(None)
                CP.append(p.cpu().numpy())
            continue
        bags = [w_rows[wi][:max_cells] for wi in ws]
        x, m = pad_bags(X, bags, dev)
        o = model(x, m)
        P.extend(torch.softmax(o['logits'], 1).cpu().numpy())
        cp = torch.softmax(o['cell_logits'], -1)
        for i, b in enumerate(bags):
            A.append(o['attn'][i, :, : len(b)].mean(0).cpu().numpy())
            CP.append(cp[i, : len(b)].cpu().numpy())
    return np.stack(P), A, CP


def _mean_profile_fold(arm: str, M: np.ndarray, y: np.ndarray, labelled: pd.DataFrame, tr: np.ndarray,
                       te: np.ndarray, n_cls: int, C: float) -> np.ndarray:
    """Test-well probabilities of the mean-profile baselines on one fold."""
    if arm == 'mean_lr':
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        ss = StandardScaler().fit(M[tr])
        clf = LogisticRegression(C=C, max_iter=3000).fit(ss.transform(M[tr]), y[tr])
        P = np.zeros((len(te), n_cls))
        P[:, clf.classes_] = clf.predict_proba(ss.transform(M[te]))
        return P
    tt = labelled.iloc[tr].assign(y=y[tr]).groupby(['compound', 'concentration']).indices
    prof = np.stack([M[tr][v].mean(0) for v in tt.values()])
    py = np.array([y[tr][v[0]] for v in tt.values()])
    q = M[te] / np.linalg.norm(M[te], axis=1, keepdims=True)
    prof = prof / np.linalg.norm(prof, axis=1, keepdims=True)
    sim = q @ prof.T
    P = np.zeros((len(te), n_cls))
    for c in range(n_cls):
        if (py == c).any():
            P[:, c] = np.exp(10 * sim[:, py == c].max(1))
    P /= P.sum(1, keepdims=True)
    return P


def _penetrance_rows(model, Z, labelled, dmso, tr, te, P, A, CP, classes, w_rows, fold, arm, dev,
                     max_cells) -> List[Dict[str, Any]]:
    """Per test well: penetrant fraction against this fold's training-week control cells, attention entropy and top cells."""
    dw = dmso[dmso['week'].isin(labelled['week'].iloc[tr].unique())]
    _, _, dcp = predict_arrayed(model, Z, list(dw['rows']), np.arange(len(dw)), dev, arm, max_cells)
    thr = np.percentile(np.concatenate(dcp), 95, axis=0)
    rows = []
    for i, wi in enumerate(te):
        c = int(P[i].argmax())
        top = np.argsort(-A[i])[:10]
        topi = np.argsort(-CP[i][:, c])[:10]
        rows.append({'plate': labelled.at[wi, 'plate'], 'well': labelled.at[wi, 'well'], 'fold': fold,
                     'moa': labelled.at[wi, 'moa'], 'pred': classes[c],
                     'penetrant_fraction': float((CP[i][:, c] > thr[c]).mean()),
                     'attn_entropy': float(-(A[i] * np.log(A[i] + 1e-12)).sum() / np.log(len(A[i]))),
                     'top_instance_rows': ','.join(map(str, np.asarray(w_rows[wi])[topi])),
                     'top_attn_rows': ','.join(map(str, np.asarray(w_rows[wi])[top]))})
    return rows


def run_well_set_arm(cfg: Mapping[str, Any], cells: pd.DataFrame, Z: np.ndarray, out: Any = None,
                     log: Callable[..., None] = _quiet) -> Dict[str, Any]:
    """Train and score one arm over every fold of a protocol.

    :param cfg: the arm config: ``arm`` (one of :data:`WELL_SET_ARMS`),
        ``design`` (one of :data:`PROTOCOLS`), ``seed``, ``shuffle_labels``,
        ``threads``, ``folds`` (fold numbers to run, or None), ``control``,
        ``max_cells``, ``emb_tag``, ``C`` (mean_lr) and the training keys of
        :func:`train_arrayed`.
    :param cells: the cell table of :func:`load_cell_embeddings`.
    :param Z: the normalised embeddings.
    :param out: folder for treatment_probs.csv and well_penetrance.csv; None writes nothing.
    :param log: called with keyword events (per fold and every 250 steps).
    :returns: the summary dict of the arm (as the reference ``summary.json``).
    :raises ValueError: an unknown arm.
    """
    from .tabular import write_table

    arm = cfg['arm']
    if arm not in WELL_SET_ARMS:
        raise ValueError(f"arm {arm!r} is not one of {list(WELL_SET_ARMS)}")
    seed = cfg.get('seed', 0)
    control = cfg.get('control', 'DMSO')
    max_cells = int(cfg.get('max_cells', 300))
    torch.manual_seed(seed)
    W = group_wells(cells)
    labelled = W[W['moa'] != control].reset_index(drop=True)
    dmso = W[W['moa'] == control].reset_index(drop=True)
    classes = sorted(labelled['moa'].unique())
    moa = labelled['moa'].copy()
    if cfg.get('shuffle_labels'):
        rng = np.random.default_rng(1)
        cm = labelled.drop_duplicates('compound').set_index('compound')['moa']
        perm = dict(zip(cm.index, rng.permutation(cm.values)))
        moa = labelled['compound'].map(perm)
    y = moa.map({c: i for i, c in enumerate(classes)}).to_numpy()
    folds = moa_folds(labelled, cfg['design'])
    dev = torch.device('cpu')
    threads = torch.get_num_threads()
    torch.set_num_threads(int(cfg.get('threads', 2)))
    w_rows = list(labelled['rows'])
    allP = np.zeros((len(labelled), len(classes)))
    folds_col = np.empty(len(labelled), object)
    scored = np.zeros(len(labelled), bool)
    pen_rows: List[Dict[str, Any]] = []
    wanted = cfg.get('folds')
    try:
        for k, (name, tr, te) in enumerate(folds):
            if wanted is not None and k not in wanted:
                continue
            tr, te = np.flatnonzero(tr), np.flatnonzero(te)
            trained = set(np.array(classes)[np.unique(y[tr])])
            t0 = time.time()
            if arm in ('mean_lr', 'mean_knn'):
                M = np.stack([Z[r].astype(np.float32).mean(0) for r in w_rows])
                P = _mean_profile_fold(arm, M, y, labelled, tr, te, len(classes), cfg.get('C', 0.1))
            else:
                pool = None
                if arm == 'mixbag':
                    pool = [np.concatenate([w_rows[i] for i in tr[y[tr] == c]]) if (y[tr] == c).any()
                            else np.array([], int) for c in range(len(classes))]
                model = train_arrayed(cfg, Z, w_rows, y, len(classes), tr, dev, log, pool)
                P, A, CP = predict_arrayed(model, Z, w_rows, te, dev, arm, max_cells)
                if arm != 'n1':
                    pen_rows += _penetrance_rows(model, Z, labelled, dmso, tr, te, P, A, CP, classes, w_rows,
                                                 name, arm, dev, max_cells)
            allP[te] = zero_untrained(P, classes, trained)
            folds_col[te] = name
            scored[te] = True
            log(event='fold', fold=name, n_train_wells=int(len(tr)), n_test_wells=int(len(te)),
                s=round(time.time() - t0, 1))
        ws = labelled[scored].copy()
        ws['fold'] = folds_col[scored]
        ws['moa'] = moa[scored].values
        tdf, res = score_treatments(ws, allP[scored], classes)
        res.update({'arm': arm, 'design': cfg['design'], 'emb_tag': cfg.get('emb_tag', 'resnet18'),
                    'shuffled': bool(cfg.get('shuffle_labels')), 'seed': seed})
        if arm in TRANSFORMER_ARMS:
            res['params'] = int(count_params(_well_model(cfg, Z.shape[1], len(classes))))
    finally:
        torch.set_num_threads(threads)
    if pen_rows:
        pr = pd.DataFrame(pen_rows)
        res['median_penetrant_fraction'] = float(pr['penetrant_fraction'].median())
    if out is not None:
        folder = Path(os.fspath(out))
        folder.mkdir(parents=True, exist_ok=True)
        write_table(tdf, folder / 'treatment_probs.csv', canonicalise=False)
        if pen_rows:
            write_table(pr, folder / 'well_penetrance.csv', canonicalise=False)
    return res


def arm_config(settings: Mapping[str, Any], arm: str, shuffled: bool = False) -> Dict[str, Any]:
    """The :func:`run_well_set_arm` config of one arm from settings.

    :param settings: the keys of :func:`well_set_defaults`.
    :param arm: one of :data:`WELL_SET_ARMS`.
    :param shuffled: the shuffled-label control of the arm.
    :returns: the config dict, in the reference ``config.json`` keys.
    """
    s = well_set_defaults(settings)
    lo, hi = (int(v) for v in _names(s['train_set_bag_cells'])[:2])
    folds = _names(s['train_set_folds'])
    return {'arm': arm, 'design': s['train_protocol'], 'emb_tag': s['train_set_tag'],
            'pca_dims': int(s['train_set_pca_dims']), 'steps': int(s['train_set_steps']),
            'bag_batch': int(s['train_set_bag_batch']), 'bag_cells': [lo, hi], 'dim': int(s['train_set_dim']),
            'depth': int(s['train_set_depth']), 'lr': float(s['train_set_lr']),
            'weight_decay': float(s['train_set_weight_decay']),
            'instance_weight': float(s['train_set_instance_weight']), 'seed': int(s['train_set_seed']),
            'threads': int(s['train_set_threads']), 'shuffle_labels': bool(shuffled),
            'control': s['train_set_control'], 'max_cells': int(s['train_set_max_cells']),
            'folds': [int(f) for f in folds] if folds else None}


def summary_row(run: str, res: Mapping[str, Any]) -> Dict[str, Any]:
    """One ``summary.csv`` row from an arm's summary dict.

    :param run: the run name.
    :param res: the dict :func:`run_well_set_arm` returned (with ``wall_s``).
    :returns: the row, in :data:`SUMMARY_COLUMNS`.
    """
    r = dict(res, run=run)
    if 'acc_ci' in r:
        r['acc_lo'], r['acc_hi'] = r.pop('acc_ci')
    return {c: r.get(c, np.nan) for c in SUMMARY_COLUMNS}


def well_set_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_well_set_training`, filled in with their defaults.

    They are the settings of :mod:`spacr.architecture_training` (app key
    ``spacr_architectures``).

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every key.
    """
    from .architecture_training import architecture_defaults

    return architecture_defaults(settings)


def run_well_set_training(settings: Mapping[str, Any], log: Callable[..., None] = _quiet) -> Dict[str, Any]:
    """Run every arm of ``train_set_arms`` from settings and write the results.

    Each arm gets its folder ``<dst>/<run>/`` with config.json,
    summary.json, treatment_probs.csv and (transformer arms)
    well_penetrance.csv; ``<dst>/summary.csv`` has one row per arm, per
    shuffled control (``train_set_shuffle_control``) and the majority-class
    control, and ``<dst>/glossary.json`` spells out its columns.

    :param settings: the keys of :func:`well_set_defaults`.
    :param log: called with keyword events.
    :returns: dict with ``summary`` (the frame) and ``paths``.
    :raises ValueError: a blank input, an unknown arm or train_level, or an
        image architecture (those train on crops: :mod:`spacr.channel_pretraining`).
    """
    from .tabular import write_table

    s = well_set_defaults(settings)
    if s['train_level'] != 'well_set' or s['train_arch'] != 'spacr_well_transformer':
        raise ValueError("run_well_set_training trains train_arch='spacr_well_transformer' at "
                         "train_level='well_set'; the image models train on crops")
    if not s['train_set_input']:
        raise ValueError('train_set_input is blank: name the folder of per-plate cell embeddings')
    arms = _names(s['train_set_arms'])
    bad = [a for a in arms if a not in WELL_SET_ARMS]
    if bad:
        raise ValueError(f"arms {bad} are not among {list(WELL_SET_ARMS)}")
    cells, Z = load_cell_embeddings(s['train_set_input'], s['train_set_cells'] or None,
                                    method=s['train_set_normalisation'], pca_dims=int(s['train_set_pca_dims']),
                                    control=s['train_set_control'])
    dst = Path(os.fspath(s['train_set_dst'] or Path(os.fspath(s['train_set_input'])).parent / 'well_set_training'))
    dst.mkdir(parents=True, exist_ok=True)
    rows, paths = [], []
    runs = [(a, False) for a in arms] + ([(a, True) for a in arms] if s['train_set_shuffle_control'] else [])
    tag = f"{s['train_protocol']}_{s['train_set_tag']}"
    for arm, shuffled in runs:
        run = f"{tag}_{arm}" + ('_shuffled' if shuffled else '')
        cfg = arm_config(s, arm, shuffled)
        folder = dst / run
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
        t0 = time.time()
        res = run_well_set_arm(cfg, cells, Z, folder, log)
        res['wall_s'] = round(time.time() - t0, 1)
        (folder / 'summary.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
        rows.append(summary_row(run, res))
        logger.info('%s: accuracy %.3f over %s treatments', run, res['accuracy'], res['n_treatments'])
    if rows:
        maj = dict(rows[0], run=f'{tag}_majority_class', arm='majority_class', shuffled=False,
                   accuracy=rows[0]['majority_class_accuracy'], acc_lo=np.nan, acc_hi=np.nan, macro_auroc=np.nan,
                   macro_ap=np.nan, median_penetrant_fraction=np.nan, params=np.nan, wall_s=np.nan)
        rows.append(maj)
    table = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    paths.append(write_table(table, dst / 'summary.csv', canonicalise=False))
    (dst / 'glossary.json').write_text(json.dumps(GLOSSARY, indent=1), encoding='utf-8')
    paths.append(str(dst / 'glossary.json'))
    return {'summary': table, 'paths': paths}


def synthetic_arrayed(seed: int = 0, n_wells: int = 240, K: int = 4, D: int = 64, cells=(80, 200),
                      pen_range=(0.1, 0.9), shift: float = 2.5):
    """Synthetic arrayed wells with a known penetrance.

    Each well has one class; a well-specific fraction of its cells, uniform
    on ``pen_range``, is shifted along the class direction and the rest look
    like controls, plus a per-well batch offset.

    :param seed: numpy seed.
    :param n_wells: wells (well ``w`` has class ``w % K``).
    :param K: classes.
    :param D: embedding dimension.
    :param cells: range of cells per well.
    :param pen_range: range of the penetrant fraction.
    :param shift: length of the class shift.
    :returns: the embeddings, the cell rows of each well, the well labels,
        the true penetrance, the per-cell penetrant flag and 4,000 control cells.
    """
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(K, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    X, wrows, lab, pen, is_pen = [], [], [], [], []
    r0 = 0
    for w in range(n_wells):
        y = w % K
        n = int(rng.integers(*cells))
        p = rng.uniform(*pen_range)
        hit = rng.random(n) < p
        x = rng.normal(size=(n, D)) + rng.normal(scale=0.5, size=D) + shift * hit[:, None] * dirs[y]
        X.append(x)
        wrows.append(np.arange(r0, r0 + n))
        r0 += n
        lab.append(y)
        pen.append(hit.mean())
        is_pen.append(hit)
    ctrl = rng.normal(size=(4000, D)) + rng.normal(scale=0.5, size=(4000 // 200, 1, D)).repeat(200, 0).reshape(4000, D)
    return (np.concatenate(X).astype(np.float32), wrows, np.array(lab), np.array(pen), np.concatenate(is_pen),
            ctrl.astype(np.float32))


def run_synthetic_arrayed(cfg: Mapping[str, Any], data=None) -> Dict[str, Any]:
    """Train and score an arrayed arm on :func:`synthetic_arrayed` (every fifth well tested).

    :param cfg: arm config as :func:`train_arrayed`, plus ``pen`` and ``shift`` for the data.
    :param data: the output of :func:`synthetic_arrayed`; None draws it with ``cfg['seed']``.
    :returns: dict with ``well_accuracy`` and, except ``n1``, the penetrance
        Pearson r and mean absolute error and the median ratio of attention
        on penetrant over other cells.
    """
    torch.manual_seed(cfg.get('seed', 0))
    X, wrows, lab, pen, is_pen, ctrl = data or synthetic_arrayed(
        cfg.get('seed', 0), pen_range=tuple(cfg.get('pen', (0.1, 0.9))), shift=cfg.get('shift', 2.5))
    K = lab.max() + 1
    idx = np.arange(len(lab))
    tr, te = idx[idx % 5 != 0], idx[idx % 5 == 0]
    dev = torch.device('cpu')
    model = train_arrayed(cfg, X, wrows, lab, K, tr, dev)
    P, A, CP = predict_arrayed(model, X, wrows, te, dev, cfg['arm'])
    res: Dict[str, Any] = {'arm': cfg['arm'], 'well_accuracy': float((P.argmax(1) == lab[te]).mean())}
    if cfg['arm'] != 'n1':
        Xc = np.concatenate([X, ctrl])
        crow = [np.arange(len(X), len(Xc))[i:i + 200] for i in range(0, len(ctrl), 200)]
        _, _, dcp = predict_arrayed(model, Xc, crow, np.arange(len(crow)), dev, cfg['arm'])
        thr = np.percentile(np.concatenate(dcp), 95, axis=0)
        est = np.array([(CP[i][:, P[i].argmax()] > thr[P[i].argmax()]).mean() for i in range(len(te))])
        ratio = [A[i][is_pen[wrows[w]]].mean() / max(A[i][~is_pen[wrows[w]]].mean(), 1e-12)
                 for i, w in enumerate(te) if is_pen[wrows[w]].any() and (~is_pen[wrows[w]]).any()]
        res.update({'penetrance_pearson_r': float(np.corrcoef(est, pen[te])[0, 1]),
                    'penetrance_mae': float(np.abs(est - pen[te]).mean()),
                    'attn_ratio_penetrant_vs_not_median': float(np.median(ratio))})
    return res


def synthetic_pooled(seed: int = 0, n_wells: int = 60, cells: int = 300, K: int = 3, D: int = 32,
                     batch_sd: float = 2.0, sep: float = 1.0):
    """Synthetic pooled wells: every well mixes all classes under a strong per-well batch offset.

    :param seed: numpy seed.
    :param n_wells: wells.
    :param cells: cells per well.
    :param K: classes.
    :param D: embedding dimension.
    :param batch_sd: size of the per-well offset a no-context model cannot remove.
    :param sep: class separation.
    :returns: the embeddings, the cell labels and the cell wells.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(K, D)) * sep
    X, y, well = [], [], []
    for w in range(n_wells):
        off = rng.normal(scale=batch_sd, size=D)
        yy = rng.integers(0, K, cells)
        X.append(mu[yy] + off + rng.normal(size=(cells, D)))
        y.append(yy)
        well.append(np.full(cells, w))
    return np.concatenate(X).astype(np.float32), np.concatenate(y), np.concatenate(well)


def make_pooled_batch(rng: np.random.Generator, cells_by_well: Mapping[Any, np.ndarray], wells: Sequence[Any],
                      T: int, Mctx: int, tgt_ok: np.ndarray, ctx_ok: np.ndarray, ctx_mode: str,
                      all_wells: Sequence[Any]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Disjoint target and context cells per well.

    :param rng: numpy generator.
    :param cells_by_well: the cell rows of each well.
    :param wells: the wells of the batch.
    :param T: targets per well.
    :param Mctx: context cells per well.
    :param tgt_ok: per cell, may it be a target.
    :param ctx_ok: per cell, may it be context.
    :param ctx_mode: ``'same'`` (other cells of the same well), ``'none'`` or ``'other'`` (a random other well).
    :param all_wells: the wells ``'other'`` draws from.
    :returns: the target rows and the context rows of each well; a target is never in its own context.
    """
    tg, cx = [], []
    for w in wells:
        r = cells_by_well[w]
        cand = r[tgt_ok[r]]
        t = rng.choice(cand, min(T, len(cand)), replace=False)
        if ctx_mode == 'none':
            c = np.array([], int)
        else:
            src = w if ctx_mode == 'same' else rng.choice([v for v in all_wells if v != w])
            rr = cells_by_well[src]
            rr = rr[ctx_ok[rr] & ~np.isin(rr, t)]
            c = rng.choice(rr, min(Mctx, len(rr)), replace=False)
        tg.append(t)
        cx.append(c)
    return tg, cx


def pooled_forward(model: SpacrWellTransformer, X: np.ndarray, tg: Sequence[np.ndarray],
                   cx: Sequence[np.ndarray], dev: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Logits of the target cells given their contexts.

    :param model: a pooled well transformer.
    :param X: cell embeddings.
    :param tg: target rows per well.
    :param cx: context rows per well (may be empty).
    :param dev: torch device.
    :returns: the ``(B, T, K)`` logits and the target validity mask.
    """
    xt, mt = pad_bags(X, tg, dev)
    if max(len(c) for c in cx) == 0:
        return model(xt, None, None)['logits'], mt
    xc, mc = pad_bags(X, [c if len(c) else np.array([0]) for c in cx], dev)
    for i, c in enumerate(cx):
        if not len(c):
            mc[i] = False
    return model(xt, xc, mc)['logits'], mt


def run_synthetic_pooled(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Train and score a pooled arm on :func:`synthetic_pooled`, half of every well's cells held out.

    :param cfg: ``arm`` (a key of :data:`POOLED_ARMS`), seed, steps,
        well_batch, targets, context, dim, depth, lr, weight_decay,
        batch_sd and sep.
    :returns: dict with ``arm``, ``cell_accuracy`` and ``chance``.
    """
    torch.manual_seed(cfg.get('seed', 0))
    X, y, well = synthetic_pooled(cfg.get('seed', 0), batch_sd=cfg.get('batch_sd', 2.0), sep=cfg.get('sep', 1.0))
    K = int(y.max() + 1)
    cbw = {w: np.flatnonzero(well == w) for w in np.unique(well)}
    wells = np.array(sorted(cbw))
    is_test = np.random.default_rng(0).random(len(y)) < 0.5
    dev = torch.device('cpu')
    mode = POOLED_ARMS[cfg['arm']]
    model = SpacrWellTransformer(X.shape[1], K, 'pooled', dim=cfg.get('dim', 128), depth=cfg.get('depth', 2),
                                 heads=cfg.get('heads', 4), dropout=cfg.get('dropout', 0.1),
                                 feat_dropout=cfg.get('feat_dropout', 0.1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 3e-4), weight_decay=cfg.get('weight_decay', 0.05))
    steps = cfg.get('steps', 1500)
    sc = cosine_schedule(opt, steps)
    rng = np.random.default_rng(cfg.get('seed', 0))
    yt = torch.from_numpy(y)
    model.train()
    for _ in range(steps):
        ws = rng.choice(wells, cfg.get('well_batch', 8))
        tg, cx = make_pooled_batch(rng, cbw, ws, cfg.get('targets', 16), cfg.get('context', 128), ~is_test,
                                   ~is_test, mode, wells)
        lo, mt = pooled_forward(model, X, tg, cx, dev)
        loss = F.cross_entropy(lo[mt], yt[np.concatenate(tg)].to(lo.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sc.step()
    model.eval()
    prng = np.random.default_rng(123)
    rows, outs = [], []
    T = cfg.get('targets', 16)
    with torch.no_grad():
        for w in wells:
            r = cbw[w]
            r = r[is_test[r]]
            for s0 in range(0, len(r), T):
                t = r[s0:s0 + T]
                if mode == 'none':
                    c = np.array([], int)
                else:
                    src = w if mode == 'same' else prng.choice([v for v in wells if v != w])
                    rr = cbw[src]
                    rr = rr[~np.isin(rr, t)]
                    c = prng.choice(rr, min(cfg.get('context', 128), len(rr)), replace=False)
                lo, _ = pooled_forward(model, X, [t], [c], dev)
                rows.append(t)
                outs.append(lo[0, : len(t)].cpu().numpy())
    rows_a, lo_a = np.concatenate(rows), np.concatenate(outs)
    return {'arm': cfg['arm'], 'cell_accuracy': float((lo_a.argmax(1) == y[rows_a]).mean()), 'chance': 1 / K}
