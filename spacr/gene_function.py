"""Multimodal gene-function prediction: screen x prior-knowledge fusion, screen-contribution controls, explained hypotheses.

What an imaging screen adds over prior protein knowledge is stated by
fusing screen-based and knowledge-based predictions and comparing them under
controls, on the gene graph's held-out-gene protocol (:mod:`spacr.gene_graph`)
with the knowledge blocks of :mod:`spacr.gene_knowledge`:

* :func:`run_knn_stack`: per-block cosine averaged over the blocks both genes
  have, and a 25-nearest-neighbour similarity-weighted category vote, over
  arms of blocks (``k_*``) with their no-screen arms and their
  screen-shuffled controls (``shufOPS``: the ``ops`` rows permuted among
  library genes; ``shufScreen``: ``ops`` and the image blocks permuted with
  one permutation);
* :func:`train_gene_function`: the gene graph's GPS / LR trainer on the
  knowledge arms, with task-specific blocks (``_f<k>``, ``_time``) and the
  Foldseek edges as an extra message-passing edge type (``struct``);
* :func:`run_late_fusion`: the mean of z-scored score matrices of two runs,
  and the same with the screen model replaced by its shuffled run;
* :func:`evaluate_gene_function`: the gene graph metrics plus paralog-excluded
  link and category AUPRC, the results table, the paired fold deltas and the
  winner rule, fixed before the runs: an arm qualifies when it beats the
  screen-only reference and the knowledge-only model in at least 4 of 5
  folds and its screen-shuffled control loses in at least 4 of 5; a simpler
  qualifying arm within 0.005 of the best wins when it is not worse in at
  least 3 of 5 folds;
* :func:`explain_gene_function`: cross-fitted calibration, the decoy FDR and
  the hypothesis table for poorly characterised genes, each call explained
  from the inputs only (screen similarity to the class's training members,
  nearest member by sequence embedding, shared domains, structural hits);
* :func:`run_gene_function` runs them in order.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import gene_graph as gg
from . import gene_knowledge as gk

logger = logging.getLogger(__name__)

APP_KEY = 'gene_function'
PREFIX = 'gene_function_'

K = ['esm', 'dom', 'txt']
IMG = ['img_dino', 'img_rn50']
ST = ['s3di', 'struct_spec']

#: The kNN stack arms of the reference analysis.
STACK_ARMS: Dict[str, List[str]] = {
    'k_ops': ['ops'], 'k_esm': ['esm'], 'k_dom': ['dom'], 'k_txt': ['txt'], 'k_img': IMG, 'k_s3di': ['s3di'],
    'k_s3di_kmer': ['s3di_kmer'], 'k_struct': ['struct_spec'],
    'k_a': ['esm', 'ops'], 'k_b': ['esm', 'dom', 'ops'], 'k_b_noOPS': ['esm', 'dom'], 'k_c': K + ['ops'], 'k_f': K,
    'k_d': ['ops'] + IMG, 'k_e': K + IMG + ['ops'], 'k_e_noOPS': K + IMG, 'k_g': ['s3di', 'ops'],
    'k_h': ['struct_spec', 'ops'], 'k_ep': K + IMG + ST + ['ops'], 'k_ep_noOPS': K + IMG + ST, 'k_fp': K + ST,
    'k_ep_kmer': K + IMG + ['s3di_kmer', 'struct_spec', 'ops']}

#: The GPS / LR arms of the reference analysis; ``struct`` adds the Foldseek edges.
GPS_ARMS: Dict[str, List[str]] = {
    'ops': ['ops'], 'graph': [], 'a': ['esm', 'ops'], 'a_noOPS': ['esm'], 'b': ['esm', 'dom', 'ops'],
    'b_noOPS': ['esm', 'dom'], 'c': K + ['ops'], 'f': K, 'd': ['ops'] + IMG, 'd_noOPS': IMG,
    'e': K + IMG + ['ops'], 'e_noOPS': K + IMG, 'dom': ['dom'], 'txt': ['txt'], 'g': ['s3di', 'ops'],
    'g_noOPS': ['s3di'], 'h': ['ops', 'struct'], 'h_noOPS': ['struct'], 'ep': K + IMG + ['s3di', 'ops', 'struct'],
    'ep_noOPS': K + IMG + ['s3di', 'struct'], 'ep_noScreen': K + ['s3di', 'struct']}

#: (model, arm) -> (no-OPS counterpart, no-screen counterpart, shuffled control).
RELATIONS: Dict[str, List[Optional[str]]] = {
    'gps:a': ['a_noOPS', None, 'shufOPS'], 'gps:b': ['b_noOPS', None, 'shufOPS'], 'gps:c': ['f', None, 'shufOPS'],
    'gps:d': ['d_noOPS', 'graph', 'shufScreen'], 'gps:e': ['e_noOPS', 'f', 'shufScreen'],
    'gps:ep': ['ep_noOPS', 'f', 'shufScreen'], 'gps:g': ['g_noOPS', None, 'shufOPS'], 'gps:h': ['h_noOPS', None, 'shufOPS'],
    'lr:c': ['f', None, 'shufOPS'], 'lr:e': ['e_noOPS', 'f', 'shufScreen'], 'lr:ep': [None, 'ep_noScreen', 'shufScreen'],
    'late:late_opsf': [None, None, 'shufScreen'], 'late:late_opsf_kNNe': [None, None, 'shufScreen'],
    'cosine:k_a': ['k_esm', None, 'shufOPS'], 'cosine:k_b': ['k_b_noOPS', None, 'shufOPS'],
    'cosine:k_c': ['k_f', None, 'shufOPS'], 'cosine:k_d': ['k_img', None, 'shufScreen'],
    'cosine:k_e': ['k_e_noOPS', 'k_f', 'shufScreen'], 'cosine:k_ep': ['k_ep_noOPS', 'k_fp', 'shufScreen'],
    'cosine:k_g': ['k_s3di', None, 'shufOPS'], 'cosine:k_h': ['k_struct', None, 'shufOPS']}

#: The knowledge-only counterpart of each model family.
KNOWLEDGE_ONLY: Dict[str, List[str]] = {'gps': ['gps', 'f'], 'cosine': ['cosine', 'k_f'], 'lr': ['lr', 'f'],
                                        'late': ['gps', 'f']}

ARM_LABEL = {'a': 'ESM + OPS', 'a_noOPS': 'ESM', 'b': 'ESM + domains + OPS', 'b_noOPS': 'ESM + domains',
             'c': 'ESM + domains + text + OPS', 'f': 'knowledge only (ESM + domains + text)', 'd': 'OPS + images',
             'd_noOPS': 'images only', 'e': 'all (ESM, domains, text, images, OPS)', 'e_noOPS': 'all without OPS',
             'ep': 'all + structure (3Di, Foldseek edges)', 'ep_noOPS': 'all + structure without OPS',
             'ep_noScreen': 'knowledge + structure (no screen)', 'g': '3Di embedding + OPS', 'g_noOPS': '3Di embedding',
             'h': 'OPS + Foldseek edges', 'h_noOPS': 'Foldseek edges only', 'dom': 'domains only', 'txt': 'text only',
             'noJUMP': 'OPS only (gene_graph)', 'graph': 'graph only (gene_graph)'}

_DEFAULTS: Dict[str, Any] = {
    'gene_function_src': '',
    'gene_function_gene_graph_src': '',
    'gene_function_knowledge_src': '',
    'gene_function_tasks': ['fold0', 'fold1', 'fold2', 'fold3', 'fold4'],
    'gene_function_stack_arms': STACK_ARMS,
    'gene_function_stack_controls': {'k_a': 'shufOPS', 'k_b': 'shufOPS', 'k_c': 'shufOPS', 'k_g': 'shufOPS',
                                     'k_h': 'shufOPS', 'k_d': 'shufScreen', 'k_e': 'shufScreen', 'k_ep': 'shufScreen'},
    'gene_function_knn_k': 25,
    'gene_function_gps_arms': GPS_ARMS,
    'gene_function_gps_runs': ['gps:f', 'gps:c', 'gps:e', 'gps:ep', 'gps:ep_noScreen'],
    'gene_function_gps_controls': ['c:shufOPS', 'ep:shufScreen'],
    'gene_function_screen_blocks': ['ops', 'img_dino', 'img_rn50'],
    'gene_function_late_pairs': {'late_opsf': ['gene_graph:gps__noJUMP__none', 'self:gps__f__none'],
                                 'late_opsf_kNNe': ['self:gps__f__none', 'self:cosine__k_e__none']},
    'gene_function_late_controls': {'late_opsf': ['gene_graph:gps__full__shufX', 'self:gps__f__none'],
                                    'late_opsf_kNNe': ['self:gps__f__none', 'self:cosine__k_e__shufScreen']},
    'gene_function_reference_runs': ['gps:noJUMP:none', 'gps:graph:none', 'cosine:noJUMP:none', 'lr:noJUMP:none',
                                     'prior:graph:none', 'adamicadar:graph:none', 'gps:noJUMP:shufY',
                                     'cosine:noJUMP:shufX'],
    'gene_function_relations': RELATIONS,
    'gene_function_knowledge_only': KNOWLEDGE_ONLY,
    'gene_function_paralog_eval': True,
    'gene_function_winner_min_folds': 4,
    'gene_function_winner_simpler_within': 0.005,
    'gene_function_winner_simpler_min_folds': 3,
    'gene_function_explain_model': 'cosine',
    'gene_function_explain_arm': 'k_ep',
    'gene_function_explain_src': 'self',
    'gene_function_knowledge_arm': 'k_fp',
    'gene_function_shuffled_control': 'shufScreen',
    'gene_function_hypothesis_top_per_gene': 4,
    'gene_function_screen_specific_delta': 0.05,
    'gene_function_link_partners_from': 'late:late_opsf',
    'gene_function_zscore_null': 1000,
    'gene_function_uniprot_function_cache': '',
    'gene_function_interpro_entry_list': '',
    'gene_function_specificity_table': '',
    'gene_function_figures': True,
    'gene_function_seed': 20261109,
}

_EXPECTED_TYPES: Dict[str, Any] = {
    key: ((int, float) if isinstance(value, float) else type(value)) for key, value in _DEFAULTS.items()
}

_TOOLTIPS: Dict[str, str] = {
    'gene_function_src': "(str) - Output folder; gene_function/ is created under it. Default blank.",
    'gene_function_gene_graph_src': "(str) - The gene graph's src folder (holding gene_graph/: graph, folds, classes, reference runs, settings). Default blank.",
    'gene_function_knowledge_src': "(str) - The protein-knowledge src folder (holding gene_knowledge/). Default blank.",
    'gene_function_tasks': "(list) - Protocols the fusion arms run on: fold0 ... fold4 and optionally time. Default the five folds.",
    'gene_function_stack_arms': "(dict) - kNN stack arms, name -> blocks. Default the 22 arms of the reference analysis.",
    'gene_function_stack_controls': "(dict) - kNN stack arm -> its screen-shuffled control (shufOPS or shufScreen). Default the reference set.",
    'gene_function_knn_k': "(int) - Neighbours of the category vote. Default 25.",
    'gene_function_gps_arms': "(dict) - GPS / LR arms, name -> blocks; 'struct' adds the Foldseek edges as an edge type. Default the reference arms.",
    'gene_function_gps_runs': "(list) - 'model:arm' trained runs (gps or lr). Default f, c, e, ep and ep_noScreen with gps.",
    'gene_function_gps_controls': "(list) - 'arm:control' screen-shuffled controls of the gps runs. Default c:shufOPS, ep:shufScreen.",
    'gene_function_screen_blocks': "(list) - Blocks permuted together by shufScreen. Default ops, img_dino, img_rn50.",
    'gene_function_late_pairs': "(dict) - Late fusions, name -> ['root:model__arm__control', ...] with root gene_graph or self. Default late_opsf and late_opsf_kNNe.",
    'gene_function_late_controls': "(dict) - The same fusions with the screen model replaced by its shuffled run. Default the reference set.",
    'gene_function_reference_runs': "(list) - 'model:arm:control' gene graph runs evaluated beside the fusion runs. Default the reference set.",
    'gene_function_relations': "(dict) - 'model:arm' -> [no-OPS arm, no-screen arm, shuffled control] of the paired deltas and the winner rule. Default the reference table.",
    'gene_function_knowledge_only': "(dict) - Model family -> [model, arm] of its knowledge-only counterpart. Default the reference table.",
    'gene_function_paralog_eval': "(bool) - Also report every held-out metric paralog-excluded. Default True.",
    'gene_function_winner_min_folds': "(int) - Folds an arm must win against the reference, the knowledge-only model and its shuffled control. Default 4.",
    'gene_function_winner_simpler_within': "(float) - A simpler qualifying arm within this of the best wins. Default 0.005.",
    'gene_function_winner_simpler_min_folds': "(int) - Folds the simpler arm must not be worse in. Default 3.",
    'gene_function_explain_model': "(str) - Model of the hypothesis table. Default 'cosine'.",
    'gene_function_explain_arm': "(str) - Arm of the hypothesis table (the category winner). Default 'k_ep'.",
    'gene_function_explain_src': "(str) - 'self' (fusion runs) or 'gene_graph' for the explained arm. Default 'self'.",
    'gene_function_knowledge_arm': "(str) - Knowledge-only arm of the explained model (k_fp: knowledge + structure, the counterpart of k_ep). Default 'k_fp'.",
    'gene_function_shuffled_control': "(str) - Control of the explained arm. Default 'shufScreen'.",
    'gene_function_hypothesis_top_per_gene': "(int) - Calls kept per gene. Default 4.",
    'gene_function_screen_specific_delta': "(float) - A call is screen-specific when the winner exceeds knowledge-only and the shuffled control by this. Default 0.05.",
    'gene_function_link_partners_from': "(str) - 'model:arm' whose out-of-fold pair scores give the new partners. Default 'late:late_opsf'.",
    'gene_function_zscore_null': "(int) - Random member sets of the OPS / image z-scores. Default 1000.",
    'gene_function_uniprot_function_cache': "(str) - Gene -> function text table, read only post hoc (does the text name the class?). Default blank.",
    'gene_function_interpro_entry_list': "(str) - InterPro entry.list for domain names in the explanations. Default blank.",
    'gene_function_specificity_table': "(str) - Guide-specificity flags per gene. Default blank.",
    'gene_function_figures': "(bool) - Draw the module figures. Default True.",
    'gene_function_seed': "(int) - Seed of the shuffled controls, the calibration and the z-score nulls. Default 20261109.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Gene function inputs': ['gene_function_src', 'gene_function_gene_graph_src', 'gene_function_knowledge_src',
                             'gene_function_tasks', 'gene_function_uniprot_function_cache',
                             'gene_function_interpro_entry_list', 'gene_function_specificity_table',
                             'gene_function_seed'],
    'Gene function arms': ['gene_function_stack_arms', 'gene_function_stack_controls', 'gene_function_knn_k',
                           'gene_function_gps_arms', 'gene_function_gps_runs', 'gene_function_gps_controls',
                           'gene_function_screen_blocks', 'gene_function_late_pairs', 'gene_function_late_controls',
                           'gene_function_reference_runs'],
    'Gene function evaluation': ['gene_function_relations', 'gene_function_knowledge_only', 'gene_function_paralog_eval',
                                 'gene_function_winner_min_folds', 'gene_function_winner_simpler_within',
                                 'gene_function_winner_simpler_min_folds', 'gene_function_figures'],
    'Gene function hypotheses': ['gene_function_explain_model', 'gene_function_explain_arm', 'gene_function_explain_src',
                                 'gene_function_knowledge_arm', 'gene_function_shuffled_control',
                                 'gene_function_hypothesis_top_per_gene', 'gene_function_screen_specific_delta',
                                 'gene_function_link_partners_from', 'gene_function_zscore_null'],
}

_DESCRIPTION = (
    "Fuse screen gene profiles with protein knowledge (kNN stack, GPS on "
    "knowledge blocks, late fusion), with screen-shuffled controls, "
    "paralog-excluded metrics, a pre-registered winner rule and an explained "
    "hypothesis table."
)


def get_gene_function_default_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_gene_function`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every ``gene_function_*`` key.
    """
    merged = dict(settings or {})
    for key, value in _DEFAULTS.items():
        merged.setdefault(key, json.loads(json.dumps(value)))
    return merged


def _register() -> None:
    """Register this module's settings through the defaults seam, once per process."""
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, get_gene_function_default_settings, expected_types=_EXPECTED_TYPES,
                      tooltips=_TOOLTIPS, categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def options(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings without their prefix, list and dict values parsed.

    :param settings: ``gene_function_*`` settings.
    :returns: ``{unprefixed key: value}``.
    """
    return gg.options(settings, PREFIX, get_gene_function_default_settings)


def folders(o: Mapping[str, Any]) -> Dict[str, Path]:
    """Output, gene graph and knowledge folders.

    :param o: unprefixed options.
    :returns: ``root``, ``runs``, ``tables``, ``figures``, ``gg``, ``gg_data``,
        ``gg_runs``, ``gg_tables`` and ``knowledge``.
    :raises ValueError: when a folder is not named.
    """
    for k in ('src', 'gene_graph_src', 'knowledge_src'):
        if not o.get(k):
            raise ValueError(f'gene_function_{k} is blank')
    root = Path(os.path.expanduser(o['src'])) / 'gene_function'
    g = Path(os.path.expanduser(o['gene_graph_src'])) / 'gene_graph'
    return {'root': root, 'runs': root / 'runs', 'tables': root / 'tables', 'figures': root / 'figures',
            'gg': g, 'gg_data': g / 'data', 'gg_runs': g / 'runs', 'gg_tables': g / 'tables',
            'knowledge': Path(os.path.expanduser(o['knowledge_src'])) / 'gene_knowledge'}


def trainer_options(o: Mapping[str, Any], f: Mapping[str, Path]) -> Dict[str, Any]:
    """The gene graph trainer's options: the gene graph run's own settings, with the fusion arms.

    :param o: unprefixed options.
    :param f: :func:`folders`.
    :returns: unprefixed gene graph options.
    """
    saved = f['gg'] / 'settings' / 'gene_graph_settings.csv'
    base: Dict[str, Any] = {}
    if saved.exists():
        from .cli import load_settings_file

        base = load_settings_file(str(saved))
    go = gg.options(base)
    go['arms'] = {**dict(go['arms'] or {}), **dict(o['gps_arms']), **dict(o['stack_arms'])}
    go['knn_vote_k'] = int(o['knn_k'])
    return go


def function_data(f: Mapping[str, Path]) -> gg.GeneGraphData:
    """The gene graph with the knowledge and structure blocks and the Foldseek edges.

    :param f: :func:`folders`.
    :returns: the graph.
    """
    k = f['knowledge']
    sources = [p for p in (k / 'blocks.npz', k / 'structure.npz') if p.exists()]
    edges = np.load(k / 'structure.npz')['E_struct'] if (k / 'structure.npz').exists() else None
    return gg.GeneGraphData(f['gg_data'], block_sources=sources, extra_edges=edges)


def prepare_run(D: gg.GeneGraphData, name: str, go: Mapping[str, Any], o: Mapping[str, Any]) -> Dict[str, Any]:
    """The run arrays of a fusion arm, with its task-specific blocks and its screen control.

    :param D: the graph.
    :param name: the run name.
    :param go: trainer options (arms).
    :param o: unprefixed options (screen blocks, seed).
    :returns: see :func:`spacr.gene_graph.prepare`.
    """
    task, _, arm, control, seed = gg.parse_name(name)
    return gg.prepare(D, task, list(go['arms'][arm]), control, seed, go, screen_blocks=tuple(o['screen_blocks']),
                      control_seed=int(o['seed']))


def stack_names(o: Mapping[str, Any], D: gg.GeneGraphData) -> List[str]:
    """The kNN stack runs whose blocks exist, with their controls.

    :param o: unprefixed options.
    :param D: the graph.
    :returns: run names.
    """
    have = set(D.z) | {k for s in D.extra for k in s}
    names = []
    for t in o['tasks']:
        for a, bl in o['stack_arms'].items():
            if all(f'X_{b}' in have or f'X_{b}_{"time" if t == "time" else "f" + t[4:]}' in have for b in bl):
                names.append(f'{t}__cosine__{a}__none__s0')
                if a in o['stack_controls']:
                    names.append(f"{t}__cosine__{a}__{o['stack_controls'][a]}__s0")
    return names


def run_knn_stack(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Run every kNN stack arm and control on the CPU.

    :param settings: the ``gene_function_*`` settings.
    :returns: see :func:`spacr.gene_graph.train_runs`.
    """
    o = options(settings)
    f = folders(o)
    D = function_data(f)
    go = trainer_options(o, f)
    return gg.train_runs(D, stack_names(o, D), go, f['runs'], prepare_fn=lambda n: prepare_run(D, n, go, o))


def gps_names(o: Mapping[str, Any]) -> List[str]:
    """The trained fusion runs with their controls.

    :param o: unprefixed options.
    :returns: run names.
    """
    names = []
    for t in o['tasks']:
        for entry in o['gps_runs']:
            model, arm = entry.split(':')
            names.append(f'{t}__{model}__{arm}__none__s0')
        for entry in o['gps_controls']:
            arm, control = entry.split(':')
            names.append(f'{t}__gps__{arm}__{control}__s0')
    return names


def train_gene_function(settings: Mapping[str, Any], over_limit: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Train the GPS / LR fusion arms with the gene graph's trainer and card rule.

    :param settings: the ``gene_function_*`` settings.
    :param over_limit: the yield test (see :func:`spacr.gene_graph.run_config`).
    :returns: see :func:`spacr.gene_graph.train_runs`.
    """
    o = options(settings)
    f = folders(o)
    D = function_data(f)
    go = trainer_options(o, f)
    return gg.train_runs(D, gps_names(o), go, f['runs'], label='gene-function', over_limit=over_limit,
                         prepare_fn=lambda n: prepare_run(D, n, go, o))


def zscore_matrix(a: np.ndarray) -> np.ndarray:
    """``(a - mean) / (sd + 1e-9)`` over the whole matrix.

    :param a: scores.
    :returns: the z-scores.
    """
    return (a - a.mean()) / (a.std() + 1e-9)


def late_fusion(parts: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """The mean of the z-scored pair and category matrices of several runs of one fold.

    :param parts: ``scores.npz`` contents (``rows``, ``S``, ``C``), same rows.
    :returns: ``rows``, ``S`` and ``C``.
    :raises ValueError: when the runs score different rows.
    """
    rows = parts[0]['rows']
    if not all(np.array_equal(p['rows'], rows) for p in parts):
        raise ValueError('late fusion of runs with different rows')
    S = np.mean([zscore_matrix(p['S'].astype(np.float32)) for p in parts], 0)
    C = np.mean([zscore_matrix(p['C'].astype(np.float32)) for p in parts], 0)
    return {'rows': rows, 'S': S.astype(np.float32), 'C': C.astype(np.float32)}


def _part_dir(f: Mapping[str, Path], spec: str, task: str) -> Path:
    """The run folder of ``root:model__arm__control`` for one task."""
    root, run = spec.split(':', 1)
    return (f['gg_runs'] if root == 'gene_graph' else f['runs']) / f'{task}__{run}__s0'


def run_late_fusion(settings: Mapping[str, Any]) -> List[str]:
    """Write the late fusions and their screen-shuffled controls per fold.

    :param settings: the ``gene_function_*`` settings.
    :returns: the runs written.
    """
    o = options(settings)
    f = folders(o)
    done = []
    jobs = [(n, p, 'none') for n, p in o['late_pairs'].items()] + [(n, p, 'shufScreen') for n, p in o['late_controls'].items()]
    for name, parts, ctl in jobs:
        for task in [t for t in o['tasks'] if t != 'time']:
            out = f['runs'] / f'{task}__late__{name}__{ctl}__s0'
            if (out / 'done.json').exists():
                continue
            ds = [_part_dir(f, p, task) for p in parts]
            if not all((d / 'done.json').exists() for d in ds):
                logger.warning('late fusion %s %s: a part is missing', name, task)
                continue
            z = late_fusion([np.load(d / 'scores.npz') for d in ds])
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / 'scores.npz', **z)
            gg._write_json({'name': out.name, 'task': task, 'model': 'late', 'arm': name, 'control': ctl, 'seed': 0,
                            'parts': [str(d) for d in ds], 'device': 'cpu', 'seconds': 0}, out / 'done.json')
            done.append(out.name)
    return done


def fold_metrics_paralog(R: gg.Reference, z: Mapping[str, np.ndarray], PARA: np.ndarray) -> Dict[str, Any]:
    """Held-out link and category AUPRC with paralog pairs removed.

    Link cells of a test gene and a paralog are dropped from positives and
    negatives; a (gene, class) cell is dropped when a training member of
    the class is a paralog of the gene.

    :param R: the labels.
    :param z: one fold's scores.
    :param PARA: the paralog mask.
    :returns: the ``*_noparalog`` metrics and the dropped fractions.
    """
    from sklearn.metrics import average_precision_score

    rows, S, C = z['rows'], z['S'].astype(np.float64), z['C'].astype(np.float64)
    test = np.zeros(R.N, bool)
    test[rows] = True
    cols = np.flatnonzero(R.real & ~test)
    rr = R.real[rows]
    out: Dict[str, Any] = {}
    rng = np.random.default_rng(0)
    Sr = S[rr][:, cols]
    Sr = Sr + rng.random(Sr.shape) * 1e-12 * (np.abs(Sr).max() + 1)
    keep = ~PARA[np.ix_(rows[rr], cols)]
    for ref, M in (('2022', R.train), ('current', R.cur)):
        L = M[np.ix_(rows[rr], cols)]
        out[f'link_AUPRC_{ref}_noparalog'] = float(average_precision_score(L[keep], Sr[keep])) if L[keep].any() else np.nan
        out[f'link_base_{ref}_noparalog'] = float(L[keep].mean())
    Y = R.Y[rows[rr]]
    Ytr = R.Y & ~test[:, None] & R.real[:, None]
    bad = (PARA[rows[rr]].astype(np.float32) @ Ytr.astype(np.float32)) > 0
    Cr = C[rr] + rng.random(C[rr].shape) * 1e-12 * (np.abs(C).max() + 1)
    has = Y.sum(1) > 0
    k = ~bad & has[:, None]
    out['cat_micro_AUPRC_noparalog'] = float(average_precision_score(Y[k], Cr[k])) if Y[k].any() else np.nan
    out['cat_base_noparalog'] = float(Y[k].mean()) if k.any() else np.nan
    out['cat_cells_dropped_paralog_frac'] = float(bad[has].mean()) if has.any() else np.nan
    out['link_pairs_dropped_paralog_frac'] = float((~keep).mean())
    return out


def run_label(r: Any) -> str:
    """Plain-language label of a fusion results row.

    :param r: a row with ``model``, ``arm`` and ``control``.
    :returns: the label.
    """
    arm = r.arm[2:] if r.model == 'cosine' and r.arm.startswith('k_') else r.arm
    fam = {'gps': 'GPS', 'cosine': 'kNN stack' if r.arm.startswith('k_') else 'cosine', 'lr': 'LR',
           'prior': 'degree prior'}.get(r.model, r.model)
    ctl = '' if r.control == 'none' else f' [control: {r.control}]'
    return f'{fam}: {ARM_LABEL.get(arm, arm)}{ctl}'


def _reference_names(o: Mapping[str, Any]) -> List[str]:
    """The gene graph runs evaluated beside the fusion runs."""
    out = []
    for t in ['time'] + list(o['tasks']):
        for e in o['reference_runs']:
            m, a, c = e.split(':')
            out.append(f'{t}__{m}__{a}__{c}__s0')
    return out


def evaluate_gene_function(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Metrics, results table, paired deltas and the winner rule.

    :param settings: the ``gene_function_*`` settings.
    :returns: the frames written and ``winners``.
    """
    o = options(settings)
    f = folders(o)
    D = function_data(f)
    R = gg.Reference(D)
    go = trainer_options(o, f)
    PARA = gk.load_paralog_mask(f['knowledge'], D.N) if o['paralog_eval'] else None
    trows, frows = [], []
    sources = [('mm', d) for d in sorted(f['runs'].glob('*/done.json'))]
    sources += [('gene_graph', f['gg_runs'] / n / 'done.json') for n in _reference_names(o)
                if (f['gg_runs'] / n / 'done.json').exists()]
    for src, d in sources:
        j = json.loads(d.read_text(encoding='utf-8'))
        z = np.load(d.parent / 'scores.npz')
        meta = {'key': f"{src}:{j['name']}", 'src': src, **{k: j.get(k) for k in gg._KEY}}
        if j['task'] == 'time':
            r, _ = gg.time_metrics(R, z['S'].astype(np.float32), j['name'], go)
            trows.append({**meta, **r})
        else:
            m = gg.fold_metrics(R, z, j['name'])
            if PARA is not None:
                m.update(fold_metrics_paralog(R, z, PARA))
            frows.append({**meta, **m, 'paralog_with_structure': (f['knowledge'] / 'structure.npz').exists()})
    T, F_ = pd.DataFrame(trows), pd.DataFrame(frows)
    t = f['tables']
    t.mkdir(parents=True, exist_ok=True)
    gg._write(T, t / 'time_split_metrics.csv')
    gg._write(F_, t / 'heldout_metrics_per_fold.csv')
    out = {'time_split_metrics': T, 'heldout_metrics_per_fold': F_}
    if not len(F_):
        return out
    F_['seed'] = gg.seed0(F_['seed'])
    num = [c for c in F_.columns if F_[c].dtype.kind in 'fi' and c != 'seed']
    g = F_.groupby(['src', 'model', 'arm', 'control', 'seed'])[num]
    out['heldout_summary'] = g.mean().add_suffix('_mean').join(g.std().add_suffix('_sd')).join(
        g.size().rename('folds')).reset_index()
    gg._write(out['heldout_summary'], t / 'heldout_summary.csv')
    out.update(results_and_winners(F_, T, o, t))
    return out


METRICS = ['link_AUPRC_2022', 'cat_micro_AUPRC', 'cat_macro_AUPRC', 'link_AUPRC_current',
           'link_AUPRC_2022_noparalog', 'cat_micro_AUPRC_noparalog']
PRIMARY = ['link_AUPRC_2022', 'cat_micro_AUPRC']


def results_and_winners(F_: pd.DataFrame, T: pd.DataFrame, o: Mapping[str, Any], tables: Path) -> Dict[str, Any]:
    """The results table, the paired fold deltas, the winner candidates and the winners.

    :param F_: per-fold metrics (``src``, ``model``, ``arm``, ``control``, ``task``).
    :param T: time-split metrics.
    :param o: unprefixed options.
    :param tables: the tables folder.
    :returns: ``results_table``, ``paired_deltas`` and ``winners``.
    """
    F_ = F_.copy()
    F_['fold'] = F_.task.str[4:].astype(int)
    F_['label'] = [run_label(r) for r in F_.itertuples()]
    metrics = [m for m in METRICS if m in F_]
    n_folds = int(F_.fold.nunique())
    g = F_.groupby(['src', 'model', 'arm', 'control', 'label'])
    S = g[metrics].mean().join(g[metrics].std().add_suffix('_sd')).join(g.size().rename('folds')).reset_index()
    if len(T):
        S = S.merge(T[['src', 'model', 'arm', 'control', 'AUPRC_any', 'AUPRC_any_lo', 'AUPRC_any_hi', 'P@1000',
                       'P@1000_degmatched', 'NT_enrichment_top1000']], on=['src', 'model', 'arm', 'control'], how='left')
    S['time_split_valid'] = 'yes'
    S.loc[S.arm.str.contains('ep|g|h|s3di|struct', regex=True) & ~S.arm.isin(['graph']), 'time_split_valid'] = \
        'approximate (ProstT5 2023, no annotations)'
    gg._write(S, tables / 'results_table.csv')

    def per_fold(src, model, arm, ctl='none'):
        """One run's per-fold rows, or None unless every fold is there."""
        d = F_[(F_.src == src) & (F_.model == model) & (F_.arm == arm) & (F_.control == ctl)].set_index('fold')
        return d if len(d) == n_folds else None

    ref_ops = per_fold('gene_graph', 'gps', 'noJUMP')
    ref_cos = per_fold('gene_graph', 'cosine', 'noJUMP')
    deltas = []
    for key, (noops, nosc, shuf) in o['relations'].items():
        model, arm = key.split(':')
        X = per_fold('mm', model, arm)
        if X is None:
            continue
        know = o['knowledge_only'].get(model)
        for m in metrics:
            for what, Y in (('vs noOPS', per_fold('mm', model, noops) if noops else None),
                            ('vs noScreen', (per_fold('gene_graph', 'gps', 'graph') if nosc == 'graph'
                                             else per_fold('mm', model, nosc)) if nosc else None),
                            ('vs shuffled', per_fold('mm', model, arm, shuf) if shuf else None),
                            ('vs OPS-only GPS (gene_graph)', ref_ops), ('vs OPS cosine vote (gene_graph)', ref_cos),
                            ('vs knowledge-only f', per_fold('mm', *know) if know else None)):
                if Y is None:
                    continue
                d = X[m] - Y[m]
                deltas.append({'model': model, 'arm': arm,
                               'label': ARM_LABEL.get(arm.replace('k_', '', 1) if model == 'cosine' else arm, arm),
                               'comparison': what, 'metric': m, 'mean_delta': d.mean(), 'min': d.min(), 'max': d.max(),
                               'folds_better': int((d > 0).sum())})
    Dl = pd.DataFrame(deltas, columns=['model', 'arm', 'label', 'comparison', 'metric', 'mean_delta', 'min', 'max',
                                       'folds_better'])
    gg._write(Dl, tables / 'paired_deltas.csv')
    win = {}
    for m in [x for x in PRIMARY + [p + '_noparalog' for p in PRIMARY] if x in metrics]:
        Q = winner_candidates(Dl, o, m, per_fold)
        gg._write(Q, tables / f'winner_candidates_{m}.csv')
        win[m] = choose_winner(Q, o, m, per_fold, ref_ops, ref_cos)
    gg._write_json(win, tables / 'winners.json')
    return {'results_table': S, 'paired_deltas': Dl, 'winners': win}


def arm_complexity(model: str, arm: str, o: Mapping[str, Any]) -> float:
    """The number of blocks of an arm, the winner rule's simplicity measure (late fusion counts 4.5).

    :param model: the model.
    :param arm: the arm.
    :param o: unprefixed options.
    :returns: the count.
    """
    if model == 'late':
        return 4.5
    arms = {**dict(o['gps_arms']), **dict(o['stack_arms'])}
    return len(arms.get(arm, [])) + (0 if model == 'gps' else -0.5)


def winner_candidates(Dl: pd.DataFrame, o: Mapping[str, Any], m: str, per_fold: Callable) -> pd.DataFrame:
    """Every related arm with the rule's three tests for one metric.

    :param Dl: paired deltas.
    :param o: unprefixed options.
    :param m: the metric.
    :param per_fold: (src, model, arm[, control]) -> per-fold rows.
    :returns: model, arm, mean, beats_reference, beats_knowledge_only, screen_dependent,
        has_shuffled_control, qualifies, blocks; best mean first.
    """
    need = int(o['winner_min_folds'])
    q = []
    for key in o['relations']:
        model, arm = key.split(':')
        sub = Dl[(Dl.model == model) & (Dl.arm == arm) & (Dl.metric == m)].set_index('comparison')
        if not len(sub):
            continue
        ok_ref = sub.folds_better.get('vs OPS-only GPS (gene_graph)', 0) >= need and (
            m.startswith('link') or sub.folds_better.get('vs OPS cosine vote (gene_graph)', 0) >= need)
        ok_k = sub.folds_better.get('vs knowledge-only f', 0) >= need
        ok_s = sub.folds_better.get('vs shuffled', -1) >= need
        q.append({'model': model, 'arm': arm, 'mean': float(per_fold('mm', model, arm)[m].mean()),
                  'beats_reference': bool(ok_ref), 'beats_knowledge_only': bool(ok_k), 'screen_dependent': bool(ok_s),
                  'has_shuffled_control': 'vs shuffled' in sub.index, 'qualifies': bool(ok_ref and ok_k and ok_s),
                  'blocks': arm_complexity(model, arm, o)})
    cols = ['model', 'arm', 'mean', 'beats_reference', 'beats_knowledge_only', 'screen_dependent',
            'has_shuffled_control', 'qualifies', 'blocks']
    return pd.DataFrame(q, columns=cols).sort_values('mean', ascending=False)


def choose_winner(Q: pd.DataFrame, o: Mapping[str, Any], m: str, per_fold: Callable, ref_ops: Optional[pd.DataFrame],
                  ref_cos: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """The winner of one metric: the best qualifying arm, or a simpler one close to it; else the reference.

    :param Q: :func:`winner_candidates`.
    :param o: unprefixed options.
    :param m: the metric.
    :param per_fold: (src, model, arm) -> per-fold rows.
    :param ref_ops: the gene graph GPS OPS-only per-fold rows.
    :param ref_cos: the gene graph OPS cosine per-fold rows.
    :returns: ``model``, ``arm`` and ``mean``.
    """
    quals = Q[Q.qualifies]
    if not len(quals):
        ref = ref_ops if m.startswith('link') else ref_cos
        return {'model': 'reference', 'arm': 'gene_graph GPS OPS-only (links) / OPS cosine vote (categories)',
                'mean': float(ref[m].mean()) if ref is not None else float('nan')}
    best = quals.iloc[0]
    for _, r in quals.sort_values('blocks').iterrows():
        if r.blocks < best.blocks and best['mean'] - r['mean'] <= float(o['winner_simpler_within']):
            a_, b_ = per_fold('mm', r.model, r.arm)[m], per_fold('mm', best.model, best.arm)[m]
            if (a_ >= b_).sum() >= int(o['winner_simpler_min_folds']):
                best = r
                break
    return {'model': best.model, 'arm': best.arm, 'mean': float(best['mean'])}


TXT_STOP = {'complex', 'pathway', 'process', 'regulation', 'of', 'the', 'and', 'kegg', 'corum', 'go', 'funk', 'cluster',
            'positive', 'negative', 'cellular', 'activity', 'subunit', 'protein', 'via', 'to', 'in', 'by', 'a', 'mean'}


def zscore_to_members(X: np.ndarray, H: np.ndarray, g: int, mem: np.ndarray, rng: np.random.Generator,
                      n: int = 1000) -> Tuple[float, List[int]]:
    """How much more similar a gene is to a class's members than to random same-size sets.

    :param X: the block.
    :param H: its presence.
    :param g: the gene.
    :param mem: the training members.
    :param rng: the null stream.
    :param n: null sets.
    :returns: the z-score and the three most similar members.
    """
    mem = mem[H[mem] & (mem != g)]
    if not H[g] or len(mem) < 2:
        return np.nan, []
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    s = Xn @ Xn[g]
    pool = np.flatnonzero(H)
    pool = pool[pool != g]
    obs = s[mem].mean()
    null = np.array([s[rng.choice(pool, len(mem), replace=False)].mean() for _ in range(n)])
    return float((obs - null.mean()) / (null.std() + 1e-9)), list(mem[np.argsort(-s[mem])[:3]])


def _interpro_names(path: Any) -> Dict[str, str]:
    """InterPro entry -> name from ``entry.list``."""
    names: Dict[str, str] = {}
    if path and os.path.exists(os.path.expanduser(path)):
        with open(os.path.expanduser(path), encoding='utf-8') as fh:
            for ln in fh:
                t = ln.rstrip('\n').split('\t')
                if len(t) >= 3:
                    names[t[0]] = t[2]
    return names


def _block(D: gg.GeneGraphData, b: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """A block's plain arrays from any source, or None."""
    try:
        X, H = D.blocks([b], None)
    except KeyError:
        return None
    return X[b], H[b]


def explain_gene_function(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Calibrated hypotheses with drivers, an explanation from the inputs only and new partners.

    Every call is out-of-fold. Calibration is cross-fitted isotonic and
    ``q`` comes from the non-targeting decoys (:mod:`spacr.gene_graph`).
    The explanation of a (gene, class) call reads only the class's
    training-fold members and the gene's own screen, image, sequence, domain
    and structure inputs, never its text, edges or memberships; the post-hoc
    column (does the gene's current UniProt text name the class?) is not an
    input.

    :param settings: the ``gene_function_*`` settings.
    :returns: ``calibration``, ``fdr``, ``hypotheses`` and ``ranked``.
    """
    import scipy.sparse as sp

    o = options(settings)
    f = folders(o)
    D = function_data(f)
    R = gg.Reference(D)
    go = trainer_options(o, f)
    rng = np.random.default_rng(int(o['seed']))
    nodes, classes = R.nodes, D.classes
    poor = gg.poorly_characterised(nodes, R.real, go['poorly_characterised'])
    sdeg, gos = nodes.string04_degree_genome.to_numpy(), nodes.go_terms.to_numpy()
    model, arm = o['explain_model'], o['explain_arm']
    win_runs = f['runs'] if o['explain_src'] == 'self' else f['gg_runs']
    screen_only = (f['gg_runs'], 'gps', 'noJUMP', 'none') if model == 'gps' else (f['runs'], 'cosine', 'k_ops', 'none')
    arms = {'winner': (win_runs, model, arm, 'none'), 'knowledge_only': (f['runs'], model, o['knowledge_arm'], 'none'),
            'screen_only': screen_only, 'winner_shuffled': (win_runs, model, arm, o['shuffled_control'])}
    n_folds = len([t for t in o['tasks'] if t != 'time'])
    calC, cal_rows = {}, []
    for k, (runs, m, a, ctl) in arms.items():
        oof = gg.load_oof(runs, m, a, n_folds, ctl)
        if oof is None:
            continue
        Pc, rc = gg.cat_calibrated(R, oof, rng)
        calC[k] = Pc
        cal_rows += [{**r, 'which': k, 'model': m, 'arm': a, 'control': ctl} for r in rc]
    tag = f'{model}_{arm}'
    t = f['tables']
    t.mkdir(parents=True, exist_ok=True)
    gg._write(pd.DataFrame(cal_rows), t / f'calibration_{tag}.csv')
    out: Dict[str, Any] = {'calibration': pd.DataFrame(cal_rows)}
    if 'winner' not in calC:
        return out
    Pc = calC['winner']
    cand_mask = R.real[:, None] & ~R.Y
    q_comb, q_nt, q_c = gg.qvalues(Pc[cand_mask], Pc[~R.real].ravel())
    QN, QC, QB = (np.full(Pc.shape, np.nan, np.float32) for _ in range(3))
    QN[cand_mask], QC[cand_mask], QB[cand_mask] = q_nt, q_c, q_comb
    fdr = {'arm': tag, 'candidates': int(cand_mask.sum()), 'decoys': int((~R.real).sum() * R.C),
           'pass_qNT0.1': int((q_nt < 0.1).sum()), 'pass_qNT0.1_poor': int((QN[poor] < 0.1).sum()),
           'poor_genes_qNT0.1': int((QN[poor] < 0.1).any(1).sum()), 'pass_combined0.1': int((q_comb < 0.1).sum()),
           'pass_combined0.1_poor': int((QB[poor] < 0.1).sum())}
    gg._write_json(fdr, t / f'fdr_summary_{tag}.json')
    out['fdr'] = fdr
    blocks = {b: _block(D, b) for b in ('ops', 'img_dino', 'esm')}
    k = f['knowledge']
    B = sp.load_npz(k / 'data' / 'dom_membership_current.npz').tocsr()
    vocab = json.loads((k / 'data' / 'dom_vocab_current.json').read_text(encoding='utf-8'))
    names = _interpro_names(o.get('interpro_entry_list'))
    SE = set(map(tuple, D.extra_edges)) if D.extra_edges is not None else set()
    unip = (gg._read(o['uniprot_function_cache'], sep='\t').set_index('gene').function.to_dict()
            if o.get('uniprot_function_cache') else {})
    partners = new_partners(R, f, o, poor, go)
    rows = []
    for g in np.flatnonzero(poor):
        pr = np.where(cand_mask[g], Pc[g], -1)
        for rank, c in enumerate(np.argsort(-pr)[:int(o['hypothesis_top_per_gene'])], 1):
            if pr[c] <= 0:
                continue
            mem = np.flatnonzero(R.Y[:, c] & (R.fold != R.fold[g]) & R.real)
            rec = {'gene': nodes.gene[g], 'rank_in_gene': rank, 'class_id': classes.class_id[c],
                   'class': classes.name[c], 'source': classes.source[c], 'class_size': int(R.Y[:, c].sum()),
                   'fold': int(R.fold[g]), 'p_calibrated': float(Pc[g, c]), 'q_NT': float(QN[g, c]),
                   'q_cal': float(QC[g, c]), 'q_combined': float(QB[g, c]), 'string04_degree': int(sdeg[g]),
                   'go_terms': int(gos[g])}
            for kk in ('knowledge_only', 'screen_only', 'winner_shuffled'):
                rec[f'p_{kk}'] = float(calC[kk][g, c]) if kk in calC else np.nan
            rec['delta_screen'] = rec['p_calibrated'] - rec['p_knowledge_only']
            rec['delta_knowledge'] = rec['p_calibrated'] - rec['p_screen_only']
            rec['delta_vs_shuffled'] = rec['p_calibrated'] - rec['p_winner_shuffled']
            rec.update(explain_call(g, mem, blocks, B, vocab, names, SE, nodes, R, rng, int(o['zscore_null'])))
            rec['top_new_partners_link_winner'] = partners.get(g, '')
            kw = [w for w in re.findall(r'[a-z0-9]+', str(classes.name[c]).lower()) if len(w) > 3 and w not in TXT_STOP]
            ft = str(unip.get(nodes.gene[g], '')).lower()
            rec['posthoc_uniprot_mentions_class'] = bool(kw) and any(w in ft for w in kw)
            drv = []
            if rec['delta_screen'] > 0.05:
                drv.append('screen')
            if rec['delta_knowledge'] > 0.05:
                drv.append('knowledge')
            if rec['delta_vs_shuffled'] <= 0.05 and rec['p_calibrated'] > 0.05:
                drv.append('(survives OPS shuffle: not screen-specific)')
            rec['drivers'] = '+'.join(drv) if drv else 'joint'
            rows.append(rec)
    T = pd.DataFrame(rows)
    if len(T):
        T = T.sort_values('p_calibrated', ascending=False)
    gg._write(T, t / f'hypotheses_{tag}.csv')
    out['hypotheses'] = T
    out['ranked'] = rank_hypotheses(T, f, o)
    return out


def explain_call(g: int, mem: np.ndarray, blocks: Mapping[str, Any], B: Any, vocab: Sequence[str],
                 names: Mapping[str, str], SE: set, nodes: pd.DataFrame, R: gg.Reference, rng: np.random.Generator,
                 n_null: int) -> Dict[str, Any]:
    """The explanation fields of one call, from the gene's inputs and the class's training members only.

    :param g: the gene.
    :param mem: the class's training-fold members.
    :param blocks: ``ops``, ``img_dino`` and ``esm`` arrays (or None).
    :param B: node x domain membership.
    :param vocab: domain entries.
    :param names: entry names.
    :param SE: structural-similarity pairs.
    :param nodes: node table.
    :param R: the labels (real genes).
    :param rng: the z-score null stream.
    :param n_null: null sets.
    :returns: OPS_z, OPS_similar_members, IMG_z, ESM_nearest_member, ESM_nearest_pct, shared_domains,
        struct_hit_members and explanation.
    """
    rec: Dict[str, Any] = {}
    ev = []
    for b, lab in (('ops', 'OPS'), ('img_dino', 'IMG')):
        if blocks.get(b) is None:
            rec[f'{lab}_z'] = np.nan
            continue
        zz, tm = zscore_to_members(blocks[b][0], blocks[b][1], g, mem, rng, n_null)
        rec[f'{lab}_z'] = zz
        if lab == 'OPS':
            rec['OPS_similar_members'] = ','.join(nodes.gene[tm]) if tm else ''
        if np.isfinite(zz) and zz >= 2:
            ev.append(f'{lab} phenotype like members (z={zz:.1f}' + (f"; {rec['OPS_similar_members']})" if lab == 'OPS' else ')'))
    if blocks.get('esm') is not None:
        X, H = blocks['esm']
        if H[g] and len(mem):
            Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
            s = Xn @ Xn[g]
            s[g] = -np.inf
            ok = np.flatnonzero(H & R.real)
            ok = ok[ok != g]
            mm = mem[H[mem]]
            if len(mm):
                j = mm[np.argmax(s[mm])]
                pct = float((s[ok] < s[j]).mean())
                rec['ESM_nearest_member'], rec['ESM_nearest_pct'] = nodes.gene[j], pct
                if pct >= 0.99:
                    ev.append(f'sequence embedding close to {nodes.gene[j]} (top {100 * (1 - pct):.1f} %)')
    dg = set(B[g].indices)
    if dg and len(mem):
        cnt: Dict[int, int] = {}
        for m_ in mem:
            for t in dg & set(B[m_].indices):
                cnt[t] = cnt.get(t, 0) + 1
        sh = [vocab[t] for t, n in sorted(cnt.items(), key=lambda x: -x[1]) if n >= 2][:3]
        rec['shared_domains'] = ';'.join(f"{x} {names.get(x, '')}".strip() for x in sh)
        if sh:
            ev.append('shares domains with members: ' + ', '.join(names.get(x, x) for x in sh))
    sm = [nodes.gene[m_] for m_ in mem if (min(g, m_), max(g, m_)) in SE]
    rec['struct_hit_members'] = ','.join(sm[:5])
    if sm:
        ev.append(f"Foldseek 3Di hits to {len(sm)} member(s): {', '.join(sm[:3])}")
    ev.append('own UniProt text hidden (held-out)')
    rec['explanation'] = '; '.join(ev)
    return rec


def new_partners(R: gg.Reference, f: Mapping[str, Path], o: Mapping[str, Any], poor: np.ndarray,
                 go: Mapping[str, Any]) -> Dict[int, str]:
    """The five best new partners of each poorly characterised gene from the link winner's out-of-fold scores.

    Partners already known (current references, the known-extra references,
    2022 edges), decoys and same-fold genes are excluded.

    :param R: the labels.
    :param f: :func:`folders`.
    :param o: unprefixed options.
    :param poor: poorly characterised genes.
    :param go: trainer options (known-extra references).
    :returns: gene -> comma-joined partners.
    """
    lm, la = o['link_partners_from'].split(':')
    lo = gg.load_oof(f['runs'], lm, la, len([t for t in o['tasks'] if t != 'time']))
    s04 = np.zeros((R.N, R.N), bool)
    for name in go['known_extra'] or []:
        s04 |= R.matrix(f'E_cur_{gg.known_key(name)}')
    partners: Dict[int, str] = {}
    if lo is None:
        return partners
    for d in lo.values():
        test = np.zeros(R.N, bool)
        test[d['rows']] = True
        for r_i, g in enumerate(d['rows']):
            if not poor[g]:
                continue
            sc = d['S'][r_i].astype(np.float64).copy()
            sc[~R.real | test | R.cur[g] | s04[g] | R.train[g]] = -np.inf
            partners[g] = ','.join(R.nodes.gene[np.argsort(-sc)[:5]])
    return partners


def _markdown(frame: pd.DataFrame) -> str:
    """A Markdown table of a frame, floats to three decimals."""
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '|' + '---|' * len(frame.columns)]
    for r in frame.itertuples(index=False):
        lines.append('| ' + ' | '.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in r) + ' |')
    return '\n'.join(lines)


def rank_hypotheses(T: pd.DataFrame, f: Mapping[str, Path], o: Mapping[str, Any]) -> pd.DataFrame:
    """The ranked hypothesis table: screen-specific calls first, with the gene graph's own call and flags.

    Writes ``hypotheses_ranked.csv`` and ``hypotheses_top.md``.

    :param T: the hypothesis table.
    :param f: :func:`folders`.
    :param o: unprefixed options.
    :returns: the ranked table.
    """
    t = f['tables']
    if not len(T):
        gg._write(T, t / 'hypotheses_ranked.csv')
        return T
    T = T.copy()
    delta = float(o['screen_specific_delta'])
    T['screen_specific'] = (T.delta_screen > delta) & (T.delta_vs_shuffled > delta)
    gp = f['gg_tables'] / 'predictions_gene_category_gps_noJUMP.csv'
    if gp.exists() and gp.stat().st_size > 1:
        gq = gg._read(gp, sep=',')[['gene', 'class_id', 'q_NT', 'p_calibrated']].rename(
            columns={'q_NT': 'gene_graph_q_NT', 'p_calibrated': 'gene_graph_p'})
        T = T.merge(gq, on=['gene', 'class_id'], how='left')
    else:
        T['gene_graph_q_NT'], T['gene_graph_p'] = np.nan, np.nan
    if o.get('specificity_table'):
        S = gg._read(o['specificity_table'])
        col = [c for c in S.columns if c != 'gene']
        flag = S.set_index('gene')
        T['spec_flag'] = [';'.join(c for c in col if g in flag.index and flag[c].dtype == bool and bool(flag.loc[g, c]))
                          for g in T.gene]
    T['call_type'] = ['screen-specific' if s else ('knowledge (homology/structure)' if d > delta else 'joint')
                      for s, d in zip(T.screen_specific, T.delta_knowledge)]
    T = T.sort_values(['screen_specific', 'p_calibrated'], ascending=[False, False])
    gg._write(T, t / 'hypotheses_ranked.csv')
    cols = ['gene', 'class', 'p_calibrated', 'p_knowledge_only', 'p_screen_only', 'p_winner_shuffled', 'call_type',
            'gene_graph_q_NT', 'posthoc_uniprot_mentions_class', 'explanation', 'top_new_partners_link_winner']
    top1 = T[T.rank_in_gene == 1]
    A, Bm = top1[top1.screen_specific].head(30), top1[~top1.screen_specific].head(15)
    (t / 'hypotheses_top.md').write_text(
        f'rows {len(T)}, genes {T.gene.nunique()}, screen-specific top calls {int(top1.screen_specific.sum())}\n\n'
        '## Screen-specific calls (top call per gene)\n\n' + _markdown(A[cols]) + '\n\n'
        '## Knowledge-driven calls (top call per gene, contrast)\n\n' + _markdown(Bm[cols]) + '\n', encoding='utf-8')
    return T


FIG_ROWS = [
    ['gene_graph', 'gps', 'graph', 'none', 'GPS graph only'], ['gene_graph', 'gps', 'noJUMP', 'none', 'GPS OPS only (gene_graph)'],
    ['gene_graph', 'cosine', 'noJUMP', 'none', 'OPS cosine vote (gene_graph)'],
    ['mm', 'gps', 'f', 'none', 'GPS knowledge only (ESM+dom+text)'], ['mm', 'gps', 'c', 'none', 'GPS knowledge + OPS'],
    ['mm', 'gps', 'c', 'shufOPS', 'GPS knowledge + OPS shuffled'], ['mm', 'gps', 'e', 'none', 'GPS all (+ images)'],
    ['mm', 'gps', 'ep', 'none', 'GPS all + structure'], ['mm', 'gps', 'ep', 'shufScreen', 'GPS all + structure, screen shuffled'],
    ['mm', 'late', 'late_opsf', 'none', 'late fusion: GPS OPS + GPS knowledge'],
    ['mm', 'late', 'late_opsf', 'shufScreen', 'late fusion, screen model shuffled'],
    ['mm', 'cosine', 'k_fp', 'none', 'kNN knowledge + structure (no screen)'], ['mm', 'cosine', 'k_g', 'none', 'kNN 3Di + OPS'],
    ['mm', 'cosine', 'k_h', 'none', 'kNN Foldseek graph + OPS'], ['mm', 'cosine', 'k_ep', 'none', 'kNN all + structure'],
    ['mm', 'cosine', 'k_ep', 'shufScreen', 'kNN all + structure, screen shuffled'],
    ['mm', 'late', 'late_opsf_kNNe', 'none', 'late fusion: GPS knowledge + kNN all'],
    ['mm', 'late', 'late_opsf_kNNe', 'shufScreen', 'late fusion, kNN screen shuffled']]


def _fig_colour(src: str, model: str, arm: str, control: str) -> str:
    """Bar colour: controls grey, screen arms navy, fusions teal, knowledge or graph only light."""
    if control != 'none':
        return '#9aa5b1'
    if model in ('late',) or (model == 'cosine' and src == 'mm' and arm not in ('k_fp', 'k_f')):
        return '#2a9d8f'
    if arm in ('f', 'graph', 'k_fp', 'k_f'):
        return '#8ecae6'
    return '#1f3a5f' if src == 'mm' else '#3a7ca5'


def gene_function_figures(settings: Mapping[str, Any]) -> List[str]:
    """Held-out link and category bars (all and paralog-excluded) and the screen-contribution deltas.

    Bars are fold means with jittered fold points, both at 60 % opacity;
    controls are grey and hatched. Source tables are written beside them.

    :param settings: the ``gene_function_*`` settings.
    :returns: the files written.
    """
    import matplotlib

    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt

    o = options(settings)
    f = folders(o)
    t, figs = f['tables'], f['figures']
    figs.mkdir(parents=True, exist_ok=True)
    F_ = gg._read(t / 'heldout_metrics_per_fold.csv', sep=',')
    files: List[str] = []
    rc = {'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'spacr'}
    for fig_id, (m1, m2, b1, b2, name) in {
            'fig1': ('link_AUPRC_2022', 'link_AUPRC_2022_noparalog', 'link_base_2022', 'link_base_2022_noparalog', 'link'),
            'fig2': ('cat_micro_AUPRC', 'cat_micro_AUPRC_noparalog', 'cat_base', 'cat_base_noparalog', 'category')}.items():
        with plt.rc_context(rc):
            fig, axs = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
            pts = []
            for ax, met, base, title in ((axs[0], m1, b1, f'{name}: all pairs'),
                                         (axs[1], m2, b2, f'{name}: paralog pairs removed')):
                pts += _panel(ax, F_, met, base, title)
            fig.tight_layout()
            stem = figs / f'{fig_id}_heldout_{name}'
            fig.savefig(stem.with_suffix('.png'), dpi=200)
            fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
            plt.close(fig)
        files += [str(stem.with_suffix('.png')), str(stem.with_suffix('.svg'))]
        gg._write(pd.DataFrame(pts), t / f'{fig_id}_heldout_{name}_points.csv')
    D = gg._read(t / 'paired_deltas.csv', sep=',')
    sel = [('gps', 'c', 'vs noOPS'), ('gps', 'c', 'vs shuffled'), ('gps', 'ep', 'vs noScreen'), ('gps', 'ep', 'vs shuffled'),
           ('late', 'late_opsf', 'vs knowledge-only f'), ('late', 'late_opsf', 'vs shuffled'),
           ('cosine', 'k_ep', 'vs noScreen'), ('cosine', 'k_ep', 'vs shuffled'), ('cosine', 'k_g', 'vs noOPS'),
           ('cosine', 'k_h', 'vs noOPS')]
    out = []
    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, met in zip(axs, ['link_AUPRC_2022_noparalog', 'cat_micro_AUPRC_noparalog']):
            for i, (m, a, cmp) in enumerate(sel):
                r = D[(D.model == m) & (D.arm == a) & (D.comparison == cmp) & (D.metric == met)]
                if not len(r):
                    continue
                r = r.iloc[0]
                shuffled = 'shuffled' in cmp
                ax.barh(i, r.mean_delta, color='#9aa5b1' if shuffled else '#2a9d8f', alpha=0.6,
                        hatch='///' if shuffled else None, edgecolor='#1f3a5f', linewidth=0.4)
                ax.plot([r['min'], r['max']], [i, i], color='#0b1f33', lw=1, alpha=0.6)
                out.append(r.to_dict())
            ax.axvline(0, color='#9aa5b1', lw=0.8)
            ax.set_yticks(range(len(sel)))
            ax.set_yticklabels([f'{m} {a}: {c}' for m, a, c in sel])
            ax.invert_yaxis()
            ax.set_xlabel(f"delta {met.replace('_', ' ')} (mean; line = fold range)")
        fig.tight_layout()
        stem = figs / 'fig3_screen_contribution'
        fig.savefig(stem.with_suffix('.png'), dpi=200)
        fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
        plt.close(fig)
    gg._write(pd.DataFrame(out), t / 'fig3_screen_contribution.csv')
    return files + [str(stem.with_suffix('.png')), str(stem.with_suffix('.svg'))]


def _panel(ax: Any, F_: pd.DataFrame, metric: str, base_col: str, title: str) -> List[Dict[str, Any]]:
    """One horizontal bar panel of :func:`gene_function_figures`; returns its points."""
    rng = np.random.default_rng(0)
    out = []
    for i, (src, m, a, c, lab) in enumerate(FIG_ROWS):
        d = F_[(F_.src == src) & (F_.model == m) & (F_.arm == a) & (F_.control == c)]
        if metric not in d or not len(d):
            continue
        v = d[metric].to_numpy()
        ctrl = c != 'none'
        col = _fig_colour(src, m, a, c)
        ax.barh(i, v.mean(), color=col, alpha=0.6, hatch='///' if ctrl else None, edgecolor='#1f3a5f' if ctrl else col,
                linewidth=0.6)
        ax.scatter(v, i + rng.uniform(-0.22, 0.22, len(v)), s=8, color='#0b1f33', alpha=0.6, linewidths=0, zorder=3)
        out += [{'label': lab, 'src': src, 'model': m, 'arm': a, 'control': c, 'metric': metric, 'fold': int(tk[4:]),
                 'value': x} for tk, x in zip(d.task, v)]
    if base_col in F_:
        ax.axvline(F_[base_col].mean(), color='#9aa5b1', lw=0.8, ls='--')
    ax.set_yticks(range(len(FIG_ROWS)))
    ax.set_yticklabels([r[4] for r in FIG_ROWS])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('AUPRC (held-out genes, folds)')
    return out


def run_gene_function(settings: Mapping[str, Any], over_limit: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """kNN stack, trained arms, late fusion, evaluation, explanation and figures, in order.

    The settings are written to ``<src>/gene_function/settings/gene_function_settings.csv``.
    A run that yielded the card stops after training; run it again to resume.

    :param settings: the ``gene_function_*`` settings.
    :param over_limit: the yield test of the trained arms.
    :returns: the step results.
    """
    full = get_gene_function_default_settings(settings)
    o = options(full)
    f = folders(o)
    f['root'].mkdir(parents=True, exist_ok=True)
    gg.save_settings_file(full, f['root'], 'gene_function_settings')
    res: Dict[str, Any] = {'stack': run_knn_stack(full), 'train': train_gene_function(full, over_limit)}
    if res['train']['status'] != 'done':
        return res
    res['late'] = run_late_fusion(full)
    res['evaluate'] = evaluate_gene_function(full)
    res['explain'] = explain_gene_function(full)
    if o['figures']:
        res['figures'] = gene_function_figures(full)
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run :func:`run_gene_function` from a settings file.

    :param argv: the arguments; None reads :data:`sys.argv`.
    :returns: 0 when finished, 75 when the trained arms yielded the card.
    """
    from .cli import load_settings_file

    parser = argparse.ArgumentParser(prog='python -m spacr.gene_function', description=_DESCRIPTION)
    parser.add_argument('--settings', required=True)
    args = parser.parse_args(argv)
    res = run_gene_function(load_settings_file(args.settings))
    return 0 if res['train']['status'] == 'done' else gg.YIELD_RC


if __name__ == '__main__':
    sys.exit(main())
