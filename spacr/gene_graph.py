"""Learn gene-gene links and gene-to-category membership from screen gene profiles.

An optical pooled screen gives one profile per gene. Whether those profiles
say something about gene function that the published interaction maps did
not already know is judged here by two protocols, each with its controls:

* the **time split**: the model is trained on the gene-gene edges known in
  2022 (STRING v11.5, BioGRID 4.4.214, CORUM 3.0) and scored on the edges
  added since (``gene_graph_new_edges``), among every gene pair that was not
  a 2022 edge;
* **held-out genes**: for fold ``k`` every edge and every category membership
  of the fold's genes is hidden; the model scores those genes against every
  gene and every category.

The steps, each callable on its own and all four in order by
:func:`run_gene_graph` (what ``spacr-run gene_graph`` calls):

* :func:`build_gene_graph` assembles the nodes (screen genes and
  non-targeting pseudo-genes), the feature blocks with their missing
  indicators, the typed edges, the categories and the folds;
* :func:`train_gene_graph` runs the models: a graph transformer (``gps``),
  GraphSAGE (``sage``), GCN (``gcn``), Hadamard logistic regression (``lr``)
  and a Laplacian-eigenmap embedding (``spectral``), plus the training-free
  baselines ``cosine``, ``adamicadar`` and ``prior``, each over feature arms
  and under the controls ``shufX`` (features permuted among genes),
  ``shufY`` (labels relabelled by a gene permutation) and ``rewire``
  (degree-preserving swaps);
* :func:`evaluate_gene_graph` writes the time-split and held-out metrics,
  the per-fold summaries, the ablation table and the paired fold deltas;
* :func:`predict_gene_functions` calibrates the out-of-fold scores, controls
  the false discovery rate with the non-targeting pseudo-genes as decoys and
  writes the explained prediction lists.

The models are plain PyTorch, NumPy, SciPy and scikit-learn. They run on the
device named by ``gene_graph_device``; a CUDA device that is not available
falls back to the CPU. On CUDA, the card rule of the shared workstation
applies: a run waits until the card total plus its cap stays below
``gene_graph_gpu_limit_gb`` for ``gene_graph_admission_hold_s`` seconds,
yields (saving a resume file) when the card goes over the limit, and stops
when ``gene_graph_gpu_budget_h`` GPU hours are used. On the CPU the
computation is deterministic, so a rerun from the saved settings file
writes the same tables.
"""
from __future__ import annotations

import argparse
import ast
import functools
import gzip
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

APP_KEY = 'gene_graph'
PREFIX = 'gene_graph_'

#: The trained models; every other model name is a training-free baseline.
TRAINED_MODELS: Tuple[str, ...] = ('gps', 'sage', 'gcn', 'lr')

#: The exit status of a run that yielded the card.
YIELD_RC = 75

_DEFAULTS: Dict[str, Any] = {
    'gene_graph_src': '',
    'gene_graph_data_dir': '',
    'gene_graph_runs_dir': '',
    'gene_graph_ops_gene_table': '',
    'gene_graph_ops_guide_table': '',
    'gene_graph_ops_min_cells': 10,
    'gene_graph_ops_feature_prefix': 'w_',
    'gene_graph_ops_scale': 'global_rms',
    'gene_graph_fitness_feature': True,
    'gene_graph_nt_grouping': '',
    'gene_graph_extra_blocks': {},
    'gene_graph_extra_block_nt': {},
    'gene_graph_extra_block_pca': {'jorf': 128},
    'gene_graph_sensitivity_blocks': {},
    'gene_graph_sensitivity_dims': 64,
    'gene_graph_reference_dir': '',
    'gene_graph_train_edges': ['STRING11.5>=0.7', 'BioGRID2022', 'CORUM3'],
    'gene_graph_new_edges': ['STRINGnew>=0.7', 'BioGRIDnew', 'CORUMnew'],
    'gene_graph_current_edges': ['STRING>=0.7', 'BioGRID', 'CORUM'],
    'gene_graph_known_extra': ['STRING>=0.4'],
    'gene_graph_go_annotations': '',
    'gene_graph_class_table': '',
    'gene_graph_class_members': '',
    'gene_graph_gene_folds': '',
    'gene_graph_n_folds': 5,
    'gene_graph_tasks': ['time', 'fold0', 'fold1', 'fold2', 'fold3', 'fold4'],
    'gene_graph_models': ['gps', 'sage', 'gcn', 'lr', 'spectral', 'cosine', 'adamicadar', 'prior'],
    'gene_graph_arms': {'full': ['ops', 'jcr', 'jorf'], 'noOPS': ['jcr', 'jorf'],
                        'noJUMP': ['ops'], 'graph': []},
    'gene_graph_controls': ['shufX', 'shufY', 'rewire'],
    'gene_graph_control_runs': ['gps:full', 'gps:noJUMP:shufY', 'lr:full', 'cosine:noJUMP:shufX'],
    'gene_graph_configs': [],
    'gene_graph_seeds': [0],
    'gene_graph_hidden': 128,
    'gene_graph_layers': 3,
    'gene_graph_heads': 4,
    'gene_graph_dropout': 0.2,
    'gene_graph_rwse_steps': 16,
    'gene_graph_node_mask': 0.15,
    'gene_graph_edge_mask': 0.2,
    'gene_graph_neg_per_pos': 3,
    'gene_graph_pos_weight_pair': 1.5,
    'gene_graph_neg_weight_pair': 0.5,
    'gene_graph_pos_weight_cat': 5.0,
    'gene_graph_lr_graph': 1e-3,
    'gene_graph_lr_linear': 3e-3,
    'gene_graph_wd_graph': 1e-4,
    'gene_graph_wd_linear': 1e-3,
    'gene_graph_lr_pos_minibatch': 50000,
    'gene_graph_max_epochs': 400,
    'gene_graph_eval_every': 10,
    'gene_graph_patience': 6,
    'gene_graph_min_epochs': 60,
    'gene_graph_val_edge_frac': 0.05,
    'gene_graph_val_gene_frac': 0.10,
    'gene_graph_knn_vote_k': 25,
    'gene_graph_spectral_dims': 64,
    'gene_graph_rewire_swaps': 2.0,
    'gene_graph_device': 'cuda',
    'gene_graph_gpu_cap_gb': 3.0,
    'gene_graph_gpu_limit_gb': 19.0,
    'gene_graph_admission_hold_s': 180,
    'gene_graph_gpu_budget_h': 6.0,
    'gene_graph_gpu_log': '~/.spacr/gpu/log',
    'gene_graph_bootstrap_reps': 20,
    'gene_graph_topk': [100, 1000, 10000],
    'gene_graph_fdr': 0.1,
    'gene_graph_fdr_rule': 'nt',
    'gene_graph_poorly_characterised': {'string04_degree_q': 0.25, 'go_terms_q': 0.25},
    'gene_graph_prediction_model': 'gps',
    'gene_graph_prediction_arm': 'noJUMP',
    'gene_graph_driver_arms': ['full', 'noJUMP', 'noOPS', 'graph'],
    'gene_graph_jump_blocks': {'CRISPR': 'jcr', 'ORF': 'jorf'},
    'gene_graph_jump_null': 2000,
    'gene_graph_specificity_table': '',
    'gene_graph_paired_deltas': [
        'gps/noJUMP|gps/graph', 'gps/noJUMP|cosine/noJUMP', 'gps/noJUMP|lr/noJUMP',
        'gps/full|gps/noOPS', 'gps/full|gps/noJUMP', 'sage/noJUMP|cosine/noJUMP',
        'gps/noJUMP|prior/graph', 'gps/full|gps/full/shufX'],
    'gene_graph_figures': True,
    'gene_graph_seed': 20261108,
}

_EXPECTED_TYPES: Dict[str, Any] = {
    key: ((int, float) if isinstance(value, float) else type(value))
    for key, value in _DEFAULTS.items()
}

_EXPECTED_TYPES['gene_graph_reference_dir'] = (str, list)

_TOOLTIPS: Dict[str, str] = {
    'gene_graph_src': "(str) - Output folder of the analysis; a gene_graph/ folder with data, runs, tables, figures and the saved settings is created under it. Blank stops the run with an error unless both the data and runs folders are given; the outputs then go next to the runs folder. Default blank.",
    'gene_graph_data_dir': "(str) - Folder of an assembled graph (graph.npz, nodes.parquet, classes.parquet) to use instead of <src>/gene_graph/data. Default blank.",
    'gene_graph_runs_dir': "(str) - Folder of model runs to evaluate instead of <src>/gene_graph/runs, such as the runs of an earlier analysis. Default blank.",
    'gene_graph_ops_gene_table': "(str) - Gene table of the screen: gene, n_cells and the feature columns (S02-weighted recommended). Parquet or CSV. Default blank.",
    'gene_graph_ops_guide_table': "(str) - Guide table in the same feature space: guide, gene, n_cells and the features; used for the non-targeting pseudo-genes and the fitness feature. Default blank.",
    'gene_graph_ops_min_cells': "(int) - Minimum cell count of a gene in the screen gene table; only rows with n_cells strictly above it become nodes. Higher drops more sparsely sampled genes and gives cleaner profiles, but fewer genes are scored. Lower keeps more genes with noisier profiles. Default 10.",
    'gene_graph_ops_feature_prefix': "(str) - Start of the column names that count as screen features in the gene and guide tables; every other column is ignored. A prefix that matches no column leaves the screen block empty, and a shorter prefix may pull in count or metadata columns. Default 'w_'.",
    'gene_graph_ops_scale': "(str) - 'global_rms' divides the whole screen block by one root mean square (keeps relative feature weights); 'none' leaves it. Default 'global_rms'.",
    'gene_graph_fitness_feature': "(bool) - When on, each gene gets one extra screen feature: the log of its mean cells per guide, centred and scaled. This lets the models use how well cells grow after the knockout. Off leaves the screen profile without it. Default True.",
    'gene_graph_nt_grouping': "(str) - Table of non-targeting guides grouped into pseudo-genes (pseudogene, sgRNA, fold); each pseudo-gene's profile is the mean of its guide rows. Default blank (no decoys).",
    'gene_graph_extra_blocks': "(dict) - More feature blocks as name -> gene x feature table (index = gene symbol), such as JUMP CRISPR and ORF profiles. Default {}.",
    'gene_graph_extra_block_nt': "(dict) - Name -> table of non-targeting profiles of that block; pseudo-genes receive one at the block's gene coverage rate. Default {}.",
    'gene_graph_extra_block_pca': "(dict) - Name -> PCA dimensions of that block (fit on the genes, seed 0); 0 keeps every column. Default {'jorf': 128}.",
    'gene_graph_sensitivity_blocks': "(dict) - Name -> [gene table, guide table] of an alternative screen profile, embedded (standardise + PCA 99 %) and cut to the first gene_graph_sensitivity_dims components. Default {}.",
    'gene_graph_sensitivity_dims': "(int) - How many leading embedding components of each alternative screen profile are kept as that block's features. Higher keeps more detail and noise and makes the block wider. Lower keeps only the strongest shared signal. Has no effect without sensitivity blocks. Default 64.",
    'gene_graph_reference_dir': "(str or list) - Folder, or list of folders, holding the reference tables read by name (see spacr.ops_reference). Default blank.",
    'gene_graph_train_edges': "(list) - Message-passing edge sources, each a reference name or a table of gene pairs (a, b); each source is its own edge type. Default the 2022 references.",
    'gene_graph_new_edges': "(list) - Time-split positives, one per training source in the same order; pairs already in training are removed. Default the evidence added after 2022.",
    'gene_graph_current_edges': "(list) - Current references, the second held-out reference and the 'already known' filter of the outputs. Default STRING v12, BioGRID, CORUM.",
    'gene_graph_known_extra': "(list) - References that also count as known when filtering outputs; the first gives the genome-wide degree of the poorly-characterised rule. Default ['STRING>=0.4'].",
    'gene_graph_go_annotations': "(str) - GO annotation file (GAF, may be gzipped); direct terms per gene, ND, NOT and protein binding excluded. Default blank (no GO counts).",
    'gene_graph_class_table': "(str) - Table of the gene categories to predict, with class_id, name and source columns; read together with the members table. Blank means no categories, so held-out folds score gene-gene links only and the gene-to-category lists stay empty. Default blank.",
    'gene_graph_class_members': "(str) - Table listing which genes belong to which category, one row per class_id and gene. Only used when a category table is given; rows naming a gene not in the screen or an unknown class are skipped. Default blank.",
    'gene_graph_gene_folds': "(str) - Gene folds (gene, fold); genes missing from it get a seeded random fold. Default blank.",
    'gene_graph_n_folds': "(int) - Number of gene folds for the held-out protocol; genes without a fold in the folds table get a random one of these. More folds hide fewer genes per fold and give more training data each time, but add runs. List one foldK task per fold to score them all. Default 5.",
    'gene_graph_tasks': "(list) - Which protocols to run: 'time' trains on 2022 edges and scores the edges added since, and each foldK hides fold K's genes and scores them. Removing entries skips those runs and leaves their tables and predictions empty. Default all.",
    'gene_graph_models': "(list) - Which models to run: the trained gps, sage, gcn and lr, the lr on a spectral embedding, and the training-free cosine, adamicadar and prior baselines. Removing a model skips all its runs, controls and comparisons, and saves time. Default all.",
    'gene_graph_arms': "(dict) - Feature arms, name -> blocks. Default full (ops, jcr, jorf), noOPS, noJUMP (ops) and graph (no features).",
    'gene_graph_controls': "(list) - Which negative controls to run on the control runs: shufX permutes features among genes, shufY permutes gene labels, rewire swaps edges keeping each degree. Removing one skips it; an empty list runs no controls. Default all three.",
    'gene_graph_control_runs': "(list) - The model and arm pairs that get controls, as 'model:arm' for every listed control or 'model:arm:control' for one. Entries whose model or arm is not run are skipped. More entries test more runs but add training time. Default the reference set.",
    'gene_graph_configs': "(list) - Explicit run names <task>__<model>__<arm>__<control>__s<seed>; blank derives them from the tasks, models, arms and controls. Default [].",
    'gene_graph_seeds': "(list) - Random seeds of the trained models; each seed repeats every trained run. More seeds show how much results vary between training runs, but multiply training time. Training-free baselines always run once. Default [0].",
    'gene_graph_hidden': "(int) - Size of the gene embedding inside gps, sage and gcn. Wider models can learn more detailed patterns but use more GPU memory and time and may overfit small screens. Narrower models are faster but may underfit. Default 128.",
    'gene_graph_layers': "(int) - Number of message-passing layers in gps, sage and gcn; each layer lets a gene draw on neighbours one more step away. More layers reach further but are slower and can blur genes together. Fewer layers stay local. Default 3.",
    'gene_graph_heads': "(int) - Number of attention heads in each gps layer's global self-attention. More heads let the model attend to several kinds of gene similarity at once, at the same total width. Has no effect on sage, gcn or lr. Default 4.",
    'gene_graph_dropout': "(float) - Share of activations randomly set to zero during training of gps, sage and gcn. Higher regularises more and can reduce overfitting, but too high slows learning and lowers accuracy. Zero turns dropout off. Default 0.2.",
    'gene_graph_rwse_steps': "(int) - Number of random-walk steps whose return probabilities describe each node's position in the graph and feed the graph models' encoder. More steps capture larger structures around a gene but add input features and compute. Default 16.",
    'gene_graph_node_mask': "(float) - In held-out folds, the share of training genes hidden each epoch: their edges and memberships are removed from the message graph and become the training targets. Higher mimics unseen genes more closely but leaves a sparser graph. Default 0.15.",
    'gene_graph_edge_mask': "(float) - Share of training edges taken out of the message graph each epoch and used as positive targets instead. Higher gives more targets but less graph to pass messages over. Lower keeps the graph fuller with fewer targets. Default 0.2.",
    'gene_graph_neg_per_pos': "(int) - How many non-linked gene pairs are drawn for each positive pair per epoch, half at random and half degree-weighted. More negatives teach the model to reject unrelated pairs better but cost more time per epoch. Default 3.",
    'gene_graph_pos_weight_pair': "(float) - Weight of known gene-gene pairs in the link loss. Raising it relative to the negative weight pushes the model to score more pairs as linked, which raises recall but can add false links. Lowering it does the opposite. Default 1.5.",
    'gene_graph_neg_weight_pair': "(float) - Weight of sampled non-linked pairs in the link loss. Raising it makes the model stricter, so fewer pairs score high and false links drop, but true links may be missed. Lowering it makes the model more permissive. Default 0.5.",
    'gene_graph_pos_weight_cat': "(float) - Extra weight on true category memberships in the held-out category loss, to balance their rarity. Higher makes the model predict memberships more readily, with more recall and more false calls. Lower makes it more conservative. Default 5.0.",
    'gene_graph_lr_graph': "(float) - Learning rate of the AdamW optimiser for gps, sage and gcn. Higher learns faster but can make training unstable or overshoot. Lower is steadier but may need more epochs and can stop early before converging. Default 0.001.",
    'gene_graph_lr_linear': "(float) - Learning rate of the AdamW optimiser for the lr model, also used for lr on the spectral embedding. Higher learns faster but can be unstable. Lower is steadier but may need more epochs to converge. Default 0.003.",
    'gene_graph_wd_graph': "(float) - Weight decay of the AdamW optimiser for gps, sage and gcn, which shrinks the weights at each step. Higher regularises more and can curb overfitting, but too high weakens the model. Zero turns it off. Default 0.0001.",
    'gene_graph_wd_linear': "(float) - Weight decay of the AdamW optimiser for the lr model and lr on the spectral embedding. Higher shrinks the weights more and can curb overfitting, but too high weakens the fit. Zero turns it off. Default 0.001.",
    'gene_graph_lr_pos_minibatch': "(int) - Most known pairs the lr model trains on per epoch; when there are more, a random subset of this size is drawn. Higher uses more pairs per step but costs more time and memory. Lower is faster but noisier. Default 50000.",
    'gene_graph_max_epochs': "(int) - Upper limit on training epochs of each trained run; early stopping may end a run sooner. Higher gives slow runs room to converge but can cost more time. Lower caps time, but runs may stop before they learn fully. Default 400.",
    'gene_graph_eval_every': "(int) - Number of epochs between checks on the inner validation set; the best checked state is kept. Smaller checks more often and picks the best epoch more precisely, but each check takes time. Larger is faster but coarser. Default 10.",
    'gene_graph_patience': "(int) - Number of validation checks in a row without improvement before training stops early. Higher waits longer for a late improvement but takes more time. Lower stops sooner and saves time, but may cut a run short. Default 6.",
    'gene_graph_min_epochs': "(int) - Early stopping cannot end a run before this many epochs, even without validation gains. Higher protects slow starters but costs time on runs that have stopped improving. Lower lets poor runs end sooner. Default 60.",
    'gene_graph_val_edge_frac': "(float) - On the time split, the share of training edges held back to judge early stopping and pick the best epoch. Higher gives a more reliable validation score but leaves fewer edges to train on. Lower trains on more but validates noisier. Default 0.05.",
    'gene_graph_val_gene_frac': "(float) - In held-out folds, the share of non-test genes held back with all their edges to judge early stopping. Higher gives a steadier validation score but fewer training genes. Lower trains on more genes with a noisier validation. Default 0.1.",
    'gene_graph_knn_vote_k': "(int) - In held-out folds, the cosine baseline scores each gene's categories by a similarity-weighted vote of this many most similar training genes. Higher smooths the vote over more genes. Lower relies on the closest few and is noisier. Default 25.",
    'gene_graph_spectral_dims': "(int) - Number of Laplacian-eigenmap dimensions built from the training graph for the spectral model, lr on graph position alone. More dimensions capture finer graph structure but can add noise and time. Fewer keep only broad structure. Default 64.",
    'gene_graph_rewire_swaps': "(float) - Number of degree-preserving edge swaps in the rewire control, as a multiple of the edge count; memberships are rewired the same way. Higher scrambles the graph more completely. Lower leaves more of the real structure in place. Default 2.0.",
    'gene_graph_device': "(str) - 'cuda' or 'cpu'; CUDA falls back to the CPU when no card is visible. Default 'cuda'.",
    'gene_graph_gpu_cap_gb': "(float) - Most GPU memory the training process may use, in GB; admission waits until the card has this plus 0.5 GB free under the limit. Higher allows bigger models but waits longer for room. Lower may run out of memory. Default 3.",
    'gene_graph_gpu_limit_gb': "(float) - Card memory total, in GB, that this run must stay under. A run waits to start until there is room for its cap, and yields with a resume file when the card total goes over it. Higher shares less with other jobs. Default 19.",
    'gene_graph_admission_hold_s': "(int) - Seconds the card must stay under the limit, with room for this run's cap, before training starts. Higher waits longer and avoids starting during a brief lull of other jobs. Lower starts sooner but may then yield. Default 180.",
    'gene_graph_gpu_budget_h': "(float) - Total GPU hours of finished CUDA runs in the runs folder after which the queue stops before the next trained run. Higher lets more runs finish in one go. Lower stops sooner; running again later resumes the queue. Default 6.",
    'gene_graph_gpu_log': "(str) - File the START / YIELD / FINISH / STOP lines are appended to. Default '~/.spacr/gpu/log'.",
    'gene_graph_bootstrap_reps': "(int) - Number of gene bootstrap resamples used for the 95 % interval of the time-split AUPRC. More replicates give a steadier interval but take longer to evaluate. Fewer are faster but the interval is rougher. Default 20.",
    'gene_graph_topk': "(list) - Cut-offs at which the time split reports precision and recall of the top-ranked pairs, each with a degree-matched random-pair control. Adding values reports more cut-offs. Smaller values judge only the most confident calls. Default [100, 1000, 10000].",
    'gene_graph_fdr': "(float) - False discovery rate cut-off of the prediction lists: only calls with a q-value below it are written. Higher keeps more predictions but more of them may be false. Lower keeps fewer, more reliable calls. Default 0.1.",
    'gene_graph_fdr_rule': "(str) - 'nt' (target-decoy q from the non-targeting pseudo-genes) or 'combined' (the larger of that and the calibration q). Default 'nt'.",
    'gene_graph_poorly_characterised': "(dict) - Quantiles of genome STRING >= 0.4 degree and GO terms at or below which a gene counts as poorly characterised. Default 0.25 and 0.25.",
    'gene_graph_prediction_model': "(str) - Which model's scores make the calibrated prediction lists; its runs must exist for the chosen arm. Choosing another model, such as sage or lr, writes that model's lists instead and names the files after it. Default 'gps'.",
    'gene_graph_prediction_arm': "(str) - Which feature arm of the prediction model makes the output lists; it is also compared with the driver arms to explain each call. Choosing 'full' adds the JUMP blocks to the predictions; the default keeps them as independent support. Default 'noJUMP'.",
    'gene_graph_driver_arms': "(list) - Feature arms whose calibrated probabilities are added to each prediction, to show which data block drives the call. Removing an arm drops its probability column and the drivers that depend on it. Adding arms needs their runs. Default full, noJUMP, noOPS, graph.",
    'gene_graph_jump_blocks': "(dict) - Feature blocks used as independent support for each prediction, as label to block name; each adds a cosine and a p-value column per call. Labels whose block is missing from the graph are skipped. An empty dict adds no support columns. Default {'CRISPR': 'jcr', 'ORF': 'jorf'}.",
    'gene_graph_jump_null': "(int) - Number of random gene sets, the same size as the category, drawn to give the p-value of each category call's independent support. More sets give finer, steadier p-values but take longer. Fewer are faster but coarser. Default 2000.",
    'gene_graph_specificity_table': "(str) - Table of per-gene guide-specificity flags (gene, flag_multi_copy, flag_near_paralog, flag_adjacent_stain_target), added to every prediction list for the gene and its partner. Blank marks every gene as not_in_table. Default blank.",
    'gene_graph_paired_deltas': "(list) - Comparisons written as A|B, each side model/arm with an optional control; the table gives the mean fold-by-fold difference of A over B and how many folds A wins. Adding entries reports more comparisons; empty writes none. Default the reference set.",
    'gene_graph_figures': "(bool) - When on, the held-out, time-split and calibration figures and their source tables are drawn after the predictions are written. Off skips the figures and saves time; the result tables are still written. Default True.",
    'gene_graph_seed': "(int) - Main random seed of the analysis; random folds, validation splits, masking, negative sampling, controls and calibration all derive from it. Changing it gives a different but equally valid draw, so results shift slightly. Keeping it repeats a run exactly on the CPU. Default 20261108.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Gene graph inputs': [k for k in _DEFAULTS if k.startswith((
        'gene_graph_src', 'gene_graph_data_dir', 'gene_graph_runs_dir', 'gene_graph_ops_',
        'gene_graph_fitness', 'gene_graph_nt_', 'gene_graph_extra', 'gene_graph_sensitivity',
        'gene_graph_reference', 'gene_graph_train_edges', 'gene_graph_new_edges',
        'gene_graph_current_edges', 'gene_graph_known', 'gene_graph_go_', 'gene_graph_class',
        'gene_graph_gene_folds', 'gene_graph_n_folds'))],
    'Gene graph models': [k for k in _DEFAULTS if k.startswith((
        'gene_graph_tasks', 'gene_graph_models', 'gene_graph_arms', 'gene_graph_control',
        'gene_graph_configs', 'gene_graph_seeds', 'gene_graph_hidden', 'gene_graph_layers',
        'gene_graph_heads', 'gene_graph_dropout', 'gene_graph_rwse', 'gene_graph_node_mask',
        'gene_graph_edge_mask', 'gene_graph_neg_per', 'gene_graph_pos_weight',
        'gene_graph_neg_weight', 'gene_graph_lr_', 'gene_graph_wd_', 'gene_graph_max_epochs',
        'gene_graph_eval_every', 'gene_graph_patience', 'gene_graph_min_epochs', 'gene_graph_val_',
        'gene_graph_knn', 'gene_graph_spectral', 'gene_graph_rewire', 'gene_graph_seed'))],
    'Gene graph compute': [k for k in _DEFAULTS if k.startswith((
        'gene_graph_device', 'gene_graph_gpu', 'gene_graph_admission'))],
    'Gene graph evaluation': [k for k in _DEFAULTS if k.startswith((
        'gene_graph_bootstrap', 'gene_graph_topk', 'gene_graph_fdr', 'gene_graph_poorly',
        'gene_graph_prediction', 'gene_graph_driver', 'gene_graph_jump', 'gene_graph_specificity',
        'gene_graph_paired', 'gene_graph_figures'))],
}

_DESCRIPTION = (
    "Learn gene-gene links and gene-to-category membership from screen gene "
    "profiles with a graph transformer, GraphSAGE, GCN and logistic regression "
    "against cosine, Adamic-Adar and degree baselines, on a 2022 time split and "
    "held-out genes, with shuffled and rewired controls, calibration and a "
    "non-targeting decoy FDR."
)


def get_gene_graph_default_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_gene_graph`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every ``gene_graph_*`` key.
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
    register_defaults(APP_KEY, get_gene_graph_default_settings,
                      expected_types=_EXPECTED_TYPES, tooltips=_TOOLTIPS,
                      categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def _literal(value: Any) -> Any:
    """A list or dict setting that arrived as its text (a settings CSV cell)."""
    if isinstance(value, str) and value.strip()[:1] in ('[', '{', '('):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return json.loads(value)
    return value


def options(settings: Optional[Mapping[str, Any]] = None, prefix: str = PREFIX,
            defaults: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The settings without their prefix, list and dict values parsed.

    :param settings: settings with prefixed keys; missing keys take defaults.
    :param prefix: the key prefix to strip.
    :param defaults: the defaults function; :func:`get_gene_graph_default_settings`.
    :returns: ``{unprefixed key: value}``.
    """
    merged = (defaults or get_gene_graph_default_settings)(settings)
    return {k[len(prefix):]: _literal(v) for k, v in merged.items() if k.startswith(prefix)}


def _read(path: Any, **kwargs) -> pd.DataFrame:
    """Read a table exactly as written, through :func:`spacr.tabular.read_table`."""
    from .tabular import read_table

    return read_table(os.fspath(path), canonicalise=False, report=None, **kwargs)


def _write(frame: pd.DataFrame, path: Any) -> str:
    """Write a table through :func:`spacr.tabular.write_table`, names as given."""
    from .tabular import write_table

    return write_table(frame, os.fspath(path), canonicalise=False)


def _write_json(obj: Any, path: Any) -> None:
    """Write a JSON file, indented, keys in insertion order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default), encoding='utf-8')


def _json_default(value: Any) -> Any:
    """JSON form of NumPy scalars and arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'{type(value).__name__} is not JSON serialisable')


def is_nt(names: Sequence[Any]) -> np.ndarray:
    """True for non-targeting guides and the pseudo-genes made from them.

    :param names: gene or guide names.
    :returns: a boolean array.
    """
    from .ops_reference import is_control

    return is_control(names)


def gene_symbol(name: Any) -> str:
    """The gene symbol of a row name; a shared guide ``"A&B"`` gives ``"A"``.

    :param name: the row name.
    :returns: the symbol.
    """
    return str(name).split('&')[0]


def paths(o: Mapping[str, Any]) -> Dict[str, Path]:
    """The output folders of a run.

    :param o: unprefixed options (see :func:`options`).
    :returns: ``root``, ``data``, ``runs``, ``tables`` and ``figures``.
    :raises ValueError: without ``src`` and without both override folders.
    """
    if not o.get('src') and not (o.get('data_dir') and o.get('runs_dir')):
        raise ValueError('gene_graph_src is blank: name the output folder')
    if o.get('src'):
        root = Path(os.path.expanduser(o['src'])) / 'gene_graph'
    else:
        root = Path(os.path.expanduser(o['runs_dir'])).parent
    data = Path(os.path.expanduser(o['data_dir'])) if o.get('data_dir') else root / 'data'
    runs = Path(os.path.expanduser(o['runs_dir'])) if o.get('runs_dir') else root / 'runs'
    return {'root': root, 'data': data, 'runs': runs, 'tables': root / 'tables', 'figures': root / 'figures'}


def source_label(name: str) -> str:
    """The edge source a reference name stands for: ``STRINGnew>=0.7`` gives ``STRING``.

    :param name: a reference name or a pair table's path.
    :returns: the source label.
    """
    if os.path.exists(os.path.expanduser(name)):
        return Path(name).stem
    return re.split(r'new|[\d>=@.]', name)[0] or name


def known_key(name: str) -> str:
    """The ``E_cur_<key>`` name of a known-extra reference: ``STRING>=0.4`` gives ``STRING04``.

    :param name: a reference name or a pair table's path.
    :returns: the key.
    """
    return re.sub(r'[^A-Za-z0-9]', '', source_label(name) if os.path.exists(os.path.expanduser(name)) else name)


def reference_pairs(name: str, reference_dir: Any) -> List[Tuple[str, str]]:
    """The gene pairs of a reference name, or of a table with columns ``a`` and ``b``.

    :param name: a :mod:`spacr.ops_reference` name or a path to a pair table.
    :param reference_dir: the folder the named references are read from.
    :returns: the pairs.
    """
    if os.path.exists(os.path.expanduser(name)):
        table = _read(os.path.expanduser(name))
        return list(zip(table['a'].astype(str), table['b'].astype(str)))
    from .ops_reference import reference_pairs as named

    return list(named(name, reference_dir))


def pairs_to_idx(pairs: Sequence[Tuple[str, str]], gidx: Mapping[str, int]) -> set:
    """Gene pairs as unordered node-index pairs ``(i, j)``, ``i < j``; unknown genes dropped.

    :param pairs: symbol pairs.
    :param gidx: symbol -> node index.
    :returns: the set of index pairs.
    """
    out = set()
    for a, b in pairs:
        ia, ib = gidx.get(a), gidx.get(b)
        if ia is not None and ib is not None and ia != ib:
            out.add((min(ia, ib), max(ia, ib)))
    return out


def go_counts(path: Any) -> Dict[str, int]:
    """Direct GO terms per gene symbol from a GAF file (ND, NOT and GO:0005515 excluded).

    :param path: the GAF, plain or gzipped.
    :returns: ``{symbol: number of terms}``.
    """
    counts = defaultdict(set)
    path = os.path.expanduser(os.fspath(path))
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.startswith('!'):
                continue
            t = line.split('\t')
            if len(t) < 7 or t[6] == 'ND' or t[4] == 'GO:0005515' or 'NOT' in t[3]:
                continue
            counts[t[2]].add(t[4])
    return {k: len(v) for k, v in counts.items()}


def _nt_groups(o: Mapping[str, Any], guide_index: pd.Index) -> List[Tuple[str, List[str], int]]:
    """The pseudo-genes with at least two guides present: (name, guides, fold)."""
    if not o.get('nt_grouping'):
        return []
    ntp = _read(o['nt_grouping'])
    out = []
    for pg, d in ntp.groupby('pseudogene'):
        k = [s for s in d.sgRNA if s in guide_index]
        if len(k) >= 2:
            out.append((pg, k, int(d.fold.iloc[0])))
    return out


def ops_block(o: Mapping[str, Any]) -> Dict[str, Any]:
    """The screen block: real genes, then the non-targeting pseudo-genes.

    :param o: unprefixed options.
    :returns: ``genes``, ``X`` (float32), ``log_fitness``, ``nt_fold`` and ``n_real``.
    """
    g = _read(o['ops_gene_table'])
    g = g[g.n_cells > o['ops_min_cells']].reset_index(drop=True)
    cols = [c for c in g.columns if str(c).startswith(o['ops_feature_prefix'])]
    gu = _read(o['ops_guide_table'])
    real = g[~is_nt(g.gene)].reset_index(drop=True)
    cpg = gu[~is_nt(gu.gene)].groupby('gene').n_cells.mean()
    gi = gu.set_index('guide')
    rows, names, fit, folds = [], [], [], []
    for pg, k, fold in _nt_groups(o, gi.index):
        rows.append(gi.loc[k, cols].to_numpy(float).mean(0))
        names.append(pg)
        fit.append(gi.loc[k, 'n_cells'].mean())
        folds.append(fold)
    X = real[cols].to_numpy(float)
    if rows:
        X = np.vstack([X, np.array(rows)])
    genes = list(real.gene) + names
    with np.errstate(divide='ignore'):
        f = np.log(np.r_[[cpg.get(x, np.nan) for x in real.gene], fit])
    f = np.where(np.isfinite(f), f, np.nanmedian(f))
    if o['ops_scale'] == 'global_rms':
        X = X / np.sqrt((X[: len(real)] ** 2).mean())
    if o['fitness_feature']:
        X = np.c_[X, (f - f.mean()) / f.std()]
    return {'genes': genes, 'X': X.astype(np.float32), 'log_fitness': f,
            'nt_fold': dict(zip(names, folds)), 'n_real': len(real)}


def sensitivity_block(genes: Sequence[str], o: Mapping[str, Any], gene_table: Any,
                      guide_table: Any) -> Tuple[np.ndarray, np.ndarray]:
    """An alternative screen block, embedded and cut to its first components.

    :param genes: node names.
    :param o: unprefixed options.
    :param gene_table: gene x feature table.
    :param guide_table: guide x feature table (for the pseudo-genes).
    :returns: ``(X, present)``.
    """
    from .ops_reference import embed_gene_profiles

    g = _read(gene_table)
    gu = _read(guide_table)
    meta = {'gene', 'guide', 'n_guides', 'n_cells', 'mitotic_cells', 'mitotic_index', 'mitotic_fraction'}
    feats = [c for c in g.columns if c not in meta]
    g = g[~is_nt(g.gene)]
    gi = gu.set_index('guide')
    nt_rows = {pg: gi.loc[k, feats].to_numpy(float).mean(0) for pg, k, _ in _nt_groups(o, gi.index)}
    tab = pd.concat([g.set_index('gene')[feats], pd.DataFrame(nt_rows, index=feats).T])
    E = embed_gene_profiles(tab.to_numpy(float))[:, : int(o['sensitivity_dims'])]
    E = E / np.sqrt((E ** 2).mean())
    return _place(genes, pd.Index(tab.index), E, lambda x: x)


def _place(genes: Sequence[str], index: pd.Index, E: np.ndarray,
           key: Callable[[str], str]) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of ``E`` placed at the nodes whose ``key(name)`` is in ``index``."""
    X = np.zeros((len(genes), E.shape[1]), np.float32)
    have = np.zeros(len(genes), bool)
    for i, name in enumerate(genes):
        j = index.get_indexer([key(name)])[0]
        if j >= 0:
            X[i] = E[j]
            have[i] = True
    return X, have


def extra_block(genes: Sequence[str], n_real: int, gene_table: Any, nt_table: Any,
                ncomp: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """A gene x feature block (index = symbol) with pseudo-profiles for the decoys.

    Centred on the genes, optionally reduced by a PCA fit on the genes (seed 0)
    and scaled to unit RMS. Each pseudo-gene receives a non-targeting profile
    with the probability of the genes' coverage, so coverage does not reveal
    a decoy.

    :param genes: node names, real genes first.
    :param n_real: number of real genes.
    :param gene_table: the block's gene table.
    :param nt_table: the block's non-targeting profiles, or blank.
    :param ncomp: PCA dimensions; 0 keeps every column.
    :param rng: the seeded draw of the decoy coverage.
    :returns: ``(X, present)``.
    """
    G = _read(gene_table)
    A = G.to_numpy(float)
    mu = A.mean(0)
    pca = None
    if ncomp and A.shape[1] > ncomp:
        from sklearn.decomposition import PCA

        pca = PCA(ncomp, random_state=0).fit(A - mu)
    Ag = pca.transform(A - mu) if pca is not None else A - mu
    s = np.sqrt((Ag ** 2).mean())
    Ag = Ag / s
    X, have = _place(list(genes[:n_real]) + [''] * (len(genes) - n_real), pd.Index(G.index), Ag, gene_symbol)
    have[n_real:] = False
    nnt = len(genes) - n_real
    if nt_table and nnt:
        NT = _read(nt_table).to_numpy(float)
        An = (pca.transform(NT - mu) if pca is not None else NT - mu) / s
        cov = have[:n_real].mean()
        pick = rng.random(nnt) < cov
        draw = rng.choice(len(An), nnt, replace=False)
        for k in range(nnt):
            if pick[k]:
                X[n_real + k] = An[draw[k]]
                have[n_real + k] = True
    return X, have


def build_gene_graph(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Assemble the nodes, feature blocks, typed edges, categories and folds.

    Writes ``graph.npz`` (``X_<block>``, ``H_<block>``,
    ``E_{train,new,cur}_<source>``, ``E_cur_<known extra>``, ``GC``),
    ``nodes.parquet``, ``classes.parquet`` and ``data_summary.json`` to the
    data folder.

    :param settings: the ``gene_graph_*`` settings.
    :returns: the data summary.
    """
    o = options(settings)
    out = paths(o)['data']
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(o['seed']))
    ops = ops_block(o)
    genes, n_real = ops['genes'], ops['n_real']
    N = len(genes)
    sym = [gene_symbol(g) for g in genes]
    gidx: Dict[str, int] = {}
    for i, s in enumerate(sym[:n_real]):
        gidx.setdefault(s, i)
    blocks = {'ops': (ops['X'], np.ones(N, bool))}
    for name, tables in (o['sensitivity_blocks'] or {}).items():
        blocks[name] = sensitivity_block(genes, o, tables[0], tables[1])
    for name, table in (o['extra_blocks'] or {}).items():
        blocks[name] = extra_block(genes, n_real, table, (o['extra_block_nt'] or {}).get(name, ''),
                                   int((o['extra_block_pca'] or {}).get(name, 0)), rng)
    edges: Dict[Tuple[str, str], set] = {}
    sources = [source_label(n) for n in o['train_edges']]
    for split, names in (('train', o['train_edges']), ('new', o['new_edges']), ('cur', o['current_edges'])):
        for name, label in zip(names, sources):
            edges[(split, label)] = pairs_to_idx(reference_pairs(name, o['reference_dir']), gidx)
    train_all = set().union(*[edges[('train', k)] for k in sources])
    for k in sources:
        if ('new', k) in edges:
            edges[('new', k)] = edges[('new', k)] - train_all
    degree: Dict[str, int] = {}
    for n, name in enumerate(o['known_extra'] or []):
        pairs = reference_pairs(name, o['reference_dir'])
        edges[('cur', known_key(name))] = pairs_to_idx(pairs, gidx)
        if n == 0:
            degree = pd.Series([p for pair in pairs for p in pair]).value_counts().to_dict() if pairs else {}
    go = go_counts(o['go_annotations']) if o.get('go_annotations') else {}
    if o.get('class_table'):
        ct = _read(o['class_table'])
        cm = _read(o['class_members'])
        cid = {c: i for i, c in enumerate(ct.class_id)}
        gc = sorted({(gidx[g], cid[c]) for c, g in zip(cm.class_id, cm.gene) if g in gidx and c in cid})
    else:
        ct = pd.DataFrame(columns=['class_id', 'name', 'source'])
        gc = []
    folds = _read(o['gene_folds']).set_index('gene').fold.to_dict() if o.get('gene_folds') else {}
    fold = np.array([folds.get(g, folds.get(s, -1)) for g, s in zip(genes[:n_real], sym[:n_real])]
                    + [ops['nt_fold'][g] for g in genes[n_real:]], dtype=np.int64)
    miss = fold < 0
    fold[miss] = rng.integers(0, int(o['n_folds']), miss.sum())
    nodes = pd.DataFrame({
        'gene': genes, 'sym': sym, 'is_nt': np.arange(N) >= n_real, 'fold': fold,
        'log_cells_per_guide': ops['log_fitness'],
        'string04_degree_genome': [int(degree.get(s, 0)) if i < n_real else 0 for i, s in enumerate(sym)],
        'go_terms': [int(go.get(s, 0)) if i < n_real else 0 for i, s in enumerate(sym)]})
    for b, (X, h) in blocks.items():
        nodes[f'has_{b}'] = h
    _write(nodes, out / 'nodes.parquet')
    _write(ct, out / 'classes.parquet')
    arrs = {f'X_{b}': X for b, (X, h) in blocks.items()}
    arrs.update({f'H_{b}': h for b, (X, h) in blocks.items()})
    for (split, k), s in edges.items():
        arrs[f'E_{split}_{k}'] = np.array(sorted(s), dtype=np.int32).reshape(-1, 2)
    arrs['GC'] = np.array(gc, dtype=np.int32).reshape(-1, 2)
    np.savez_compressed(out / 'graph.npz', **arrs)
    new_union = set().union(*[edges.get(('new', k), set()) for k in sources])
    summary = {
        'nodes': N, 'real_genes': n_real, 'nt_pseudogenes': N - n_real, 'classes': len(ct),
        'gene_class_edges': len(gc),
        'blocks': {b: {'dims': int(X.shape[1]), 'coverage_real': float(h[:n_real].mean()),
                       'coverage_nt': float(h[n_real:].mean()) if N > n_real else float('nan')}
                   for b, (X, h) in blocks.items()},
        'edges': {f'{s}_{k}': len(v) for (s, k), v in edges.items()},
        'train_union': len(train_all), 'new_union': len(new_union),
        'genes_missing_from_gene_folds': int(miss[:n_real].sum())}
    _write_json(summary, out / 'data_summary.json')
    return summary


class GeneGraphData:
    """The assembled graph of :func:`build_gene_graph`, read for training and scoring.

    :param data_dir: the data folder.
    :param block_sources: more ``.npz`` files with ``X_<block>`` / ``H_<block>``
        arrays; a task-specific array ``X_<block>_f<k>`` or ``X_<block>_time``
        is used for that task when it exists.
    :param extra_edges: an extra message-passing edge type (index pairs),
        used by the pseudo-block ``struct``; never supervised.
    """

    def __init__(self, data_dir: Any, block_sources: Sequence[Any] = (),
                 extra_edges: Optional[np.ndarray] = None):
        """Load the graph arrays, the nodes and the training edges with their source bits."""
        data_dir = Path(data_dir)
        self.z = dict(np.load(data_dir / 'graph.npz'))
        self.extra = [dict(np.load(p)) for p in block_sources]
        self.extra_edges = None if extra_edges is None else np.asarray(extra_edges, np.int64).reshape(-1, 2)
        self.nodes = _read(data_dir / 'nodes.parquet')
        self.classes = _read(data_dir / 'classes.parquet')
        self.N = len(self.nodes)
        self.is_nt = self.nodes.is_nt.to_numpy(bool)
        self.real = ~self.is_nt
        self.fold = self.nodes.fold.to_numpy()
        self.C = len(self.classes)
        self.sources = [k[len('E_train_'):] for k in self.z if k.startswith('E_train_')]
        bits_of: Dict[Tuple[int, int], int] = {}
        for t, k in enumerate(self.sources):
            for a, b in self.z[f'E_train_{k}']:
                key = (int(a), int(b))
                bits_of[key] = bits_of.get(key, 0) | (1 << t)
        self.U = np.array(list(bits_of.keys()), dtype=np.int64).reshape(-1, 2)
        self.Ubits = np.array(list(bits_of.values()), dtype=np.int64)
        self.GC = self.z['GC'].astype(np.int64).reshape(-1, 2)
        self.task: Optional[str] = None

    def blocks(self, names: Sequence[str], task: Optional[str] = None) -> Tuple[Dict[str, np.ndarray],
                                                                               Dict[str, np.ndarray]]:
        """The feature arrays and presence flags of blocks, task-specific where present.

        :param names: block names; ``struct`` is skipped (it is an edge type).
        :param task: ``time`` or ``fold<k>``; None uses the plain arrays.
        :returns: ``(X, H)`` dicts.
        :raises KeyError: for a block no source holds.
        """
        task = task if task is not None else self.task
        tag = 'time' if task == 'time' else (f'f{task[4:]}' if task else None)
        X, H = {}, {}
        for b in names:
            if b == 'struct':
                continue
            src = next((s for s in self.extra if f'X_{b}' in s or f'X_{b}_{tag}' in s), self.z)
            key = f'X_{b}_{tag}' if f'X_{b}_{tag}' in src else f'X_{b}'
            if key not in src:
                raise KeyError(f'block {b!r} is in no graph or block source')
            X[b] = src[key]
            H[b] = src['H_' + key[2:]]
        return X, H

    @property
    def n_types(self) -> int:
        """Edge types: one per training source plus member->category and category->member."""
        return len(self.sources) + 2


def rewire_pairs(E: np.ndarray, rng: np.random.Generator, n_swaps: int, bipartite: bool = False) -> np.ndarray:
    """Degree-preserving double-edge swaps, with no self-loop and no duplicate.

    :param E: index pairs; gene-gene pairs are stored sorted.
    :param rng: the random stream.
    :param n_swaps: swaps to make (at most ``3 * n_swaps`` attempts).
    :param bipartite: swap gene-category pairs (keeps both sides' degrees).
    :returns: the rewired pairs.
    """
    E = E.copy()
    have = set(map(tuple, E))
    m = len(E)
    done = 0
    if m < 2:
        return E
    for _ in range(n_swaps * 3):
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if bipartite:
            n1, n2 = (a, d), (c, b)
        else:
            if rng.random() < 0.5:
                n1, n2 = (a, d), (c, b)
            else:
                n1, n2 = (a, c), (b, d)
            n1, n2 = tuple(sorted(n1)), tuple(sorted(n2))
        if (not bipartite and (n1[0] == n1[1] or n2[0] == n2[1])) or n1 in have or n2 in have or n1 == n2:
            continue
        have.discard(tuple(E[i]))
        have.discard(tuple(E[j]))
        have.add(n1)
        have.add(n2)
        E[i], E[j] = n1, n2
        done += 1
        if done >= n_swaps:
            break
    return E


def spectral_block(D: GeneGraphData, U: np.ndarray, GC: Optional[np.ndarray],
                   k: int = 64) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Laplacian-eigenmap embedding of the training graph only.

    Isolated nodes (held-out genes, decoys) get zeros and the missing flag.

    :param D: the graph.
    :param U: training gene-gene pairs.
    :param GC: training memberships, or None on the time split.
    :param k: dimensions.
    :returns: ``({'spec': X}, {'spec': present})``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    N = D.N
    C = D.C if GC is not None else 0
    r = [U[:, 0], U[:, 1]]
    c = [U[:, 1], U[:, 0]]
    if C:
        r += [GC[:, 0], N + GC[:, 1]]
        c += [N + GC[:, 1], GC[:, 0]]
    r, c = np.concatenate(r), np.concatenate(c)
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(N + C, N + C))
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    dinv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0)
    An = sp.diags(dinv) @ A @ sp.diags(dinv)
    k = min(k, N + C - 2)
    vals, vecs = eigsh(An, k=k + 1, which='LA', v0=np.random.default_rng(0).random(N + C))
    o = np.argsort(-vals)
    E = vecs[:, o[1:]]
    E = E[:N] / np.sqrt((E[:N][d[:N] > 0] ** 2).mean())
    have = d[:N] > 0
    E[~have] = 0
    return {'spec': E.astype(np.float32)}, {'spec': have}


def prepare(D: GeneGraphData, task: str, blocks: Sequence[str], control: str, seed: int,
            o: Mapping[str, Any], *, spectral: bool = False, screen_blocks: Sequence[str] = ('ops',),
            control_seed: Optional[int] = None) -> Dict[str, Any]:
    """The arrays of one run: features under the control, edges, test and validation sets.

    :param D: the graph.
    :param task: ``time`` or ``fold<k>``.
    :param blocks: feature blocks of the arm (``struct`` adds the extra edge type).
    :param control: ``none``, ``shufX``, ``shufY``, ``rewire``, ``shufOPS``
        (``ops`` rows permuted among real genes) or ``shufScreen``
        (every block of ``screen_blocks`` permuted with one permutation).
    :param seed: the run seed.
    :param o: unprefixed options.
    :param spectral: replace the features by :func:`spectral_block`.
    :param screen_blocks: the screen blocks of ``shufScreen``.
    :param control_seed: master seed of the ``shufOPS`` / ``shufScreen``
        permutation; the gene graph seed when None.
    :returns: a dict of arrays.
    """
    master = int(o['seed'])
    rng = np.random.default_rng([master, seed, 7])
    Xn, Hn = D.blocks(blocks, task)
    Xn = {b: x.copy() for b, x in Xn.items()}
    Hn = {b: h.copy() for b, h in Hn.items()}
    U, bits, GC = D.U.copy(), D.Ubits.copy(), D.GC.copy()
    real_idx = np.flatnonzero(D.real)
    if control == 'shufX':
        perm = rng.permutation(real_idx)
        for b in Xn:
            Xn[b][real_idx] = Xn[b][perm]
            Hn[b][real_idx] = Hn[b][perm]
    elif control == 'shufY':
        pi = np.arange(D.N)
        pi[real_idx] = rng.permutation(real_idx)
        U = np.sort(pi[U], axis=1)
        GC = np.c_[pi[GC[:, 0]], GC[:, 1]]
    elif control == 'rewire':
        swaps = float(o['rewire_swaps'])
        U = rewire_pairs(U, rng, int(swaps * len(U)))
        GC = rewire_pairs(GC, rng, int(swaps * len(GC)), bipartite=True)
    use_cat = task != 'time'
    U_all = U.copy()
    test = np.zeros(D.N, bool)
    val_gene = np.zeros(D.N, bool)
    if use_cat:
        k = int(task.replace('fold', ''))
        test = D.fold == k
        keep = ~(test[U[:, 0]] | test[U[:, 1]])
        U, bits = U[keep], bits[keep]
        GC = GC[~test[GC[:, 0]]]
        cand = np.flatnonzero(D.real & ~test)
        val_gene[rng.choice(cand, int(float(o['val_gene_frac']) * len(cand)), replace=False)] = True
        val_edge = np.zeros(len(U), bool)
    else:
        val_edge = rng.random(len(U)) < float(o['val_edge_frac'])
    if spectral:
        Xn, Hn = spectral_block(D, U, GC if use_cat else None, int(o['spectral_dims']))
    if control in ('shufOPS', 'shufScreen'):
        prng = np.random.default_rng([master if control_seed is None else int(control_seed), seed, 13])
        perm = prng.permutation(real_idx)
        for b in Xn:
            if b == 'ops' or (control == 'shufScreen' and b in screen_blocks):
                Xn[b][real_idx] = Xn[b][perm]
                Hn[b][real_idx] = Hn[b][perm]
    S = dict(X=Xn, H=Hn, U=U, U_all=U_all, bits=bits, GC=GC if use_cat else np.zeros((0, 2), np.int64),
             use_cat=use_cat, test=test, val_gene=val_gene, val_edge=val_edge, struct='struct' in blocks)
    if S['struct']:
        S['E_struct'] = D.extra_edges if D.extra_edges is not None else np.zeros((0, 2), np.int64)
    return S


def spmm(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse times dense in float32 (no bf16 sparse kernel), result in ``x``'s dtype.

    :param A: sparse matrix.
    :param x: dense matrix.
    :returns: ``A @ x``.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        return torch.sparse.mm(A, x.float()).to(x.dtype)


def rwse(M: int, ei: torch.Tensor, k: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random-walk return probabilities ``diag(P^t)``, ``t = 1..k``, and log degree.

    :param M: number of nodes.
    :param ei: edge index (2 x E), both directions.
    :param k: steps.
    :returns: ``(M x k, M x 1)``; an isolated node has a row of zeros.
    """
    A = torch.zeros(M, M, device=ei.device)
    if ei.shape[1]:
        A[ei[0], ei[1]] = 1.0
    deg = A.sum(1)
    Pm = A / deg.clamp(min=1)[:, None]
    out = []
    Pk = Pm
    for _ in range(k):
        out.append(torch.diagonal(Pk))
        Pk = Pk @ Pm
    return torch.stack(out, 1), torch.log1p(deg)[:, None]


def mean_adj(M: int, ei: torch.Tensor, et: torch.Tensor, n_types: int) -> List[Optional[torch.Tensor]]:
    """Row-normalised sparse adjacency per edge type (rows aggregate from columns).

    :param M: number of nodes.
    :param ei: edge index.
    :param et: edge type of each edge.
    :param n_types: number of edge types.
    :returns: one sparse matrix per type, None for a type without edges.
    """
    mats: List[Optional[torch.Tensor]] = []
    for t in range(n_types):
        k = et == t
        r, c = ei[0, k], ei[1, k]
        if r.numel() == 0:
            mats.append(None)
            continue
        deg = torch.zeros(M, device=ei.device).index_add_(0, r, torch.ones_like(r, dtype=torch.float))
        mats.append(torch.sparse_coo_tensor(torch.stack([r, c]), 1.0 / deg[r], (M, M)).coalesce())
    return mats


def gcn_adj(M: int, ei: torch.Tensor) -> torch.Tensor:
    """Symmetric-normalised adjacency with self-loops.

    :param M: number of nodes.
    :param ei: edge index, both directions.
    :returns: the sparse matrix.
    """
    loops = torch.arange(M, device=ei.device)
    r = torch.cat([ei[0], loops])
    c = torch.cat([ei[1], loops])
    deg = torch.zeros(M, device=ei.device).index_add_(0, r, torch.ones_like(r, dtype=torch.float))
    v = deg[r].rsqrt() * deg[c].rsqrt()
    return torch.sparse_coo_tensor(torch.stack([r, c]), v, (M, M)).coalesce()


class Encoder(nn.Module):
    """Gene input encoder: per-block projection plus missing-indicator embedding, plus structure.

    :param dims: block -> input dimensions.
    :param h: hidden width.
    :param n_cat: number of category nodes (learnable embeddings).
    :param use_struct: add ``Linear(RWSE + log degree)``.
    :param pe_k: RWSE steps.
    """

    def __init__(self, dims: Mapping[str, int], h: int, n_cat: int, use_struct: bool, pe_k: int = 16):
        """Create the projections, missing embeddings and structural encoder."""
        super().__init__()
        self.blocks = list(dims)
        self.proj = nn.ModuleDict({b: nn.Linear(d, h) for b, d in dims.items()})
        self.miss = nn.ModuleDict({b: nn.Embedding(2, h) for b in dims})
        self.use_struct = use_struct
        if use_struct:
            self.struct = nn.Linear(pe_k + 1, h)
        self.bias = nn.Parameter(torch.zeros(h))
        self.cat = nn.Embedding(n_cat, h) if n_cat else None

    def forward(self, X, H, N, pe=None, n_cat=0):
        """Node inputs: genes first, then the category nodes.

        :param X: block -> gene features (N x block dimensions).
        :param H: block -> per-gene indicator of a present block (N).
        :param N: number of genes.
        :param pe: structural encodings (RWSE and log degree) of every node.
        :param n_cat: number of category nodes appended after the genes.
        :returns: the node inputs ((N + n_cat) x h).
        """
        z = self.bias.expand(N, -1)
        for b in self.blocks:
            z = z + self.proj[b](X[b]) * H[b][:, None].float() + self.miss[b](H[b].long())
        if self.use_struct:
            z = z + self.struct(pe[:N])
        if n_cat:
            c = self.cat.weight
            if self.use_struct:
                c = c + self.struct(pe[N:N + n_cat])
            z = torch.cat([z, c], 0)
        return z


class RSage(nn.Module):
    """Relational GraphSAGE layer: root weight plus mean aggregation per edge type.

    :param h: width.
    :param n_types: edge types.
    """

    def __init__(self, h: int, n_types: int):
        """Create the root and per-type weights."""
        super().__init__()
        self.root = nn.Linear(h, h)
        self.rel = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(n_types)])

    def forward(self, x, mats):
        """One layer of message passing.

        :param x: node states (nodes x h).
        :param mats: one sparse mean-aggregation matrix per edge type, or None.
        :returns: the new node states.
        """
        out = self.root(x)
        for t, A in enumerate(mats):
            if A is not None:
                out = out + self.rel[t](spmm(A, x))
        return out


class GPSLayer(nn.Module):
    """GraphGPS-style layer: relational SAGE plus full self-attention, summed, then a feed-forward block.

    :param h: width.
    :param heads: attention heads.
    :param drop: dropout.
    :param n_types: edge types.
    """

    def __init__(self, h: int, heads: int, drop: float, n_types: int):
        """Create the local, global and feed-forward parts."""
        super().__init__()
        self.local = RSage(h, n_types)
        self.qkv = nn.Linear(h, 3 * h)
        self.o = nn.Linear(h, h)
        self.heads = heads
        self.n1, self.n2, self.n3 = nn.LayerNorm(h), nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * h, h))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mats):
        """One layer.

        :param x: node states (nodes x h).
        :param mats: one sparse mean-aggregation matrix per edge type, or None.
        :returns: the new node states.
        """
        hl = self.n1(x + self.drop(self.local(x, mats)))
        M, h = x.shape
        q, k, v = self.qkv(x).view(M, 3, self.heads, h // self.heads).permute(1, 2, 0, 3)
        a = F.scaled_dot_product_attention(q[None], k[None], v[None], dropout_p=0.0)[0]
        ha = self.n2(x + self.drop(self.o(a.permute(1, 0, 2).reshape(M, h))))
        y = hl + ha
        return self.n3(y + self.drop(self.ff(y)))


class GeneGraphModel(nn.Module):
    """One model of the gene graph: ``lr``, ``gcn``, ``sage`` or ``gps``.

    All share the decoders: pairs by DistMult,
    ``s(u, v) = scale * sum(z_u * r * z_v) + b``, and categories by
    ``scale * (z_g * q) @ z_c + b_c``. For ``lr`` the encoder is the raw
    concatenation of the blocks and their indicators, so the pair score is
    Hadamard logistic regression and the category head is linear.

    :param kind: ``lr``, ``gcn``, ``sage`` or ``gps``.
    :param dims: block -> input dimensions.
    :param n_cat: number of categories (0 on the time split).
    :param h: hidden width.
    :param layers: layers.
    :param heads: attention heads (gps).
    :param drop: dropout.
    :param n_types: edge types.
    :param pe_k: RWSE steps.
    """

    def __init__(self, kind: str, dims: Mapping[str, int], n_cat: int, h: int = 128, layers: int = 3,
                 heads: int = 4, drop: float = 0.2, n_types: int = 5, pe_k: int = 16):
        """Create the encoder, the layers and the decoders."""
        super().__init__()
        self.kind = kind
        self.n_cat = n_cat
        self.uses_graph = kind != 'lr'
        if kind == 'lr':
            d = sum(dims.values()) + len(dims)
            self.enc = None
            self.dims = dict(dims)
            hz = max(d, 1)
            self.cat_w = nn.Linear(hz, n_cat) if n_cat else None
        else:
            if kind not in ('gps', 'sage', 'gcn'):
                raise KeyError(f'unknown model {kind!r}')
            self.enc = Encoder(dims, h, n_cat, use_struct=True, pe_k=pe_k)
            hz = h
            if kind == 'gps':
                self.layers = nn.ModuleList([GPSLayer(h, heads, drop, n_types) for _ in range(layers)])
            elif kind == 'sage':
                self.layers = nn.ModuleList([RSage(h, n_types) for _ in range(layers)])
                self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])
            else:
                self.layers = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
                self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])
            self.drop = nn.Dropout(drop)
            self.cat_q = nn.Parameter(torch.ones(hz)) if n_cat else None
            self.cat_b = nn.Parameter(torch.full((n_cat,), -3.0)) if n_cat else None
        self.r = nn.Parameter(torch.ones(hz) * (0.1 if kind == 'lr' else 1.0))
        self.b = nn.Parameter(torch.tensor(-2.0))
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(hz)))

    def embed(self, X, H, graph):
        """Gene embeddings (N x h) and category embeddings, from a message graph of :func:`build_graph`.

        :param X: block -> gene features (N x block dimensions).
        :param H: block -> per-gene indicator of a present block (N).
        :param graph: the message graph of :func:`build_graph`.
        :returns: ``(gene embeddings, category embeddings or None)``.
        """
        if self.kind == 'lr':
            parts = []
            for b in self.dims:
                parts += [X[b] * H[b][:, None].float(), H[b][:, None].float()]
            z = torch.cat(parts, 1) if parts else torch.zeros(graph['N'], 1, device=self.r.device)
            return z, None
        z = self.enc(X, H, graph['N'], graph['pe'], self.n_cat)
        for i, layer in enumerate(self.layers):
            if self.kind == 'gps':
                z = layer(z, graph['mats'])
            elif self.kind == 'sage':
                z = self.norms[i](z + self.drop(F.gelu(layer(z, graph['mats']))))
            else:
                z = self.norms[i](z + self.drop(F.gelu(spmm(graph['gadj'], layer(z)))))
        N = z.shape[0] - self.n_cat
        return z[:N], (z[N:] if self.n_cat else None)

    def pair_logit(self, zu, zv):
        """DistMult logits of matched pairs.

        :param zu: embeddings of the first gene of each pair.
        :param zv: embeddings of the second gene of each pair.
        :returns: one logit per pair.
        """
        return self.scale * (zu * self.r * zv).sum(-1) + self.b

    def pair_matrix(self, zr, zc):
        """DistMult logits of every row x column pair.

        :param zr: row gene embeddings.
        :param zc: column gene embeddings.
        :returns: the rows x columns logit matrix.
        """
        return self.scale * (zr * self.r) @ zc.T + self.b

    def cat_logits(self, zg, zc):
        """Category logits of genes.

        :param zg: gene embeddings.
        :param zc: category embeddings (unused by ``lr``).
        :returns: the genes x categories logit matrix.
        """
        if self.kind == 'lr':
            return self.cat_w(zg)
        return self.scale * (zg * self.cat_q) @ zc.T + self.cat_b


def build_graph(D: GeneGraphData, S: Mapping[str, Any], pairs: np.ndarray, bits: np.ndarray, gc: np.ndarray,
                dev: torch.device, kind: str, pe_k: int = 16) -> Dict[str, Any]:
    """Message-graph tensors from typed gene pairs, memberships and the extra edge type.

    :param D: the graph.
    :param S: the run arrays of :func:`prepare`.
    :param pairs: gene-gene pairs passing messages.
    :param bits: their source bits.
    :param gc: member -> category pairs passing messages.
    :param dev: the device.
    :param kind: the model; ``lr`` needs no graph.
    :param pe_k: RWSE steps.
    :returns: ``N`` and, for graph models, ``pe`` and ``mats`` or ``gadj``.
    """
    N, C = D.N, (D.C if S['use_cat'] else 0)
    M = N + C
    ns = len(D.sources)
    rows, cols, types = [], [], []
    for t in range(ns):
        k = (bits >> t) & 1 == 1
        a, b = pairs[k, 0], pairs[k, 1]
        rows += [a, b]
        cols += [b, a]
        types += [np.full(2 * k.sum(), t)]
    if C and len(gc):
        rows += [N + gc[:, 1], gc[:, 0]]
        cols += [gc[:, 0], N + gc[:, 1]]
        types += [np.full(len(gc), ns), np.full(len(gc), ns + 1)]
    if S.get('struct'):
        E = S['E_struct']
        rows += [E[:, 0], E[:, 1]]
        cols += [E[:, 1], E[:, 0]]
        types += [np.full(2 * len(E), ns + 2)]
    g: Dict[str, Any] = {'N': N}
    if kind == 'lr':
        return g
    ei = torch.as_tensor(np.stack([np.concatenate(rows), np.concatenate(cols)]).astype(np.int64), device=dev)
    et = torch.as_tensor(np.concatenate(types).astype(np.int64), device=dev)
    pe, ld = rwse(M, ei, pe_k)
    g['pe'] = torch.cat([pe, ld], 1)
    if kind == 'gcn':
        und = np.concatenate([pairs, gc + np.array([0, N])]) if C and len(gc) else pairs
        u = torch.as_tensor(np.concatenate([und, und[:, ::-1]]).T.copy(), device=dev, dtype=torch.long)
        g['gadj'] = gcn_adj(M, u)
    else:
        g['mats'] = mean_adj(M, ei, et, D.n_types + int(bool(S.get('struct'))))
    return g


def sample_negatives(rng: np.random.Generator, pos: np.ndarray, pool: np.ndarray, deg: np.ndarray, n: int,
                     keyset: np.ndarray, N: int) -> np.ndarray:
    """Negative pairs: half uniform, half degree-proportional corruptions; never a known pair.

    :param rng: the random stream.
    :param pos: positive pairs.
    :param pool: genes allowed in a negative.
    :param deg: degree of each node (corruption weights ``deg + 1``).
    :param n: negatives to draw before filtering.
    :param keyset: sorted keys ``i * N + j`` of every known pair.
    :param N: number of nodes.
    :returns: sorted pairs, self pairs and known pairs removed.
    """
    half = n // 2
    a = rng.choice(pool, half)
    b = rng.choice(pool, half)
    p = deg[pool] + 1.0
    p = p / p.sum()
    src = pos[rng.integers(len(pos), size=n - half)] if len(pos) else rng.choice(pool, (n - half, 2))
    side = rng.integers(2, size=n - half)
    c = src[np.arange(n - half), side]
    d = rng.choice(pool, n - half, p=p)
    na = np.concatenate([a, c])
    nb = np.concatenate([b, d])
    lo, hi = np.minimum(na, nb), np.maximum(na, nb)
    ok = (lo != hi) & ~np.isin(lo * N + hi, keyset)
    return np.stack([lo[ok], hi[ok]], 1)


class GpuYield(Exception):
    """Raised when a run yields the card; its resume file is saved."""


def gpu_log(path: Any, label: str, msg: str) -> None:
    """Append one line to the shared GPU log.

    :param path: the log file.
    :param label: the job label.
    :param msg: the message.
    """
    path = Path(os.path.expanduser(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(f"{datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')} [{label}] {msg}\n")


def card_used_gb() -> float:
    """Memory in use on the current CUDA card by every process (GB)."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 30


def card_over_limit(limit_gb: float) -> bool:
    """True when the card total is above ``limit_gb``.

    :param limit_gb: the card limit (GB).
    :returns: whether to yield.
    """
    return card_used_gb() > limit_gb


def admit(expected_peak_gb: float, limit_gb: float, hold_s: float, poll_s: float = 15.0,
          used: Callable[[], float] = card_used_gb, sleep: Callable[[float], None] = time.sleep) -> float:
    """Wait until the card total plus the expected peak stays within the limit for ``hold_s`` seconds.

    :param expected_peak_gb: this process's peak.
    :param limit_gb: the card limit.
    :param hold_s: seconds the condition must hold.
    :param poll_s: seconds between readings.
    :param used: reads the card total.
    :param sleep: waits.
    :returns: the card total at admission.
    """
    ok_since = None
    waited = 0.0
    while True:
        u = used()
        if u + expected_peak_gb <= limit_gb:
            ok_since = waited if ok_since is None else ok_since
            if waited - ok_since >= hold_s:
                return u
        else:
            ok_since = None
        sleep(poll_s)
        waited += poll_s


def _device(o: Mapping[str, Any]) -> torch.device:
    """The device of the trained models; CUDA falls back to the CPU when unavailable."""
    want = str(o.get('device') or 'cpu')
    if want.startswith('cuda') and not torch.cuda.is_available():
        logger.warning('gene_graph_device=%s but no CUDA device is visible: running on the CPU', want)
        return torch.device('cpu')
    return torch.device(want)


def parse_name(name: str) -> Tuple[str, str, str, str, int]:
    """Split a run name ``<task>__<model>__<arm>__<control>__s<seed>``.

    :param name: the run name.
    :returns: ``(task, model, arm, control, seed)``.
    """
    task, kind, arm, control, s = name.split('__')
    return task, kind, arm, control, int(s[1:])


def _arm_blocks(o: Mapping[str, Any], arm: str) -> List[str]:
    """The blocks of an arm; ``spectral`` has none (it makes its own)."""
    arms = o['arms'] or {}
    if arm == 'spectral':
        return []
    if arm not in arms:
        raise KeyError(f'arm {arm!r} is not in gene_graph_arms')
    return list(arms[arm])


def _hparams(o: Mapping[str, Any], kind: str, n_cat: int, max_epochs: int) -> Dict[str, Any]:
    """The hyperparameters recorded in ``done.json``."""
    return {'h': int(o['hidden']), 'layers': int(o['layers']), 'heads': int(o['heads']),
            'dropout': float(o['dropout']), 'lr': float(o['lr_linear'] if kind == 'lr' else o['lr_graph']),
            'weight_decay': float(o['wd_linear'] if kind == 'lr' else o['wd_graph']),
            'node_mask': float(o['node_mask']) if n_cat else 0.0, 'edge_mask': float(o['edge_mask']),
            'neg_per_pos': int(o['neg_per_pos']), 'rwse_steps': int(o['rwse_steps']),
            'max_epochs': max_epochs, 'patience_evals': int(o['patience'])}


def _average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Step-wise average precision; NaN without positives."""
    from sklearn.metrics import average_precision_score

    y = np.asarray(y).ravel()
    return float(average_precision_score(y, np.asarray(s).ravel())) if y.any() else float('nan')


class _Validation:
    """The fixed inner validation of one run (time split: hidden edges; held-out: hidden genes)."""

    def __init__(self, D, S, task, o, seed, pool, ukeys, trainable, n_cat, Y, dev, kind):
        """Draw the validation positives and negatives and build the validation message graph."""
        N = D.N
        U, bits, GC = S['U'], S['bits'], S['GC']
        self.task, self.n_cat = task, n_cat
        vr = np.random.default_rng([int(o['seed']), seed, 11])
        gc_train = GC[~S['val_gene'][GC[:, 0]]] if len(GC) else GC
        self.graph = build_graph(D, S, U[trainable], bits[trainable], gc_train, dev, kind, int(o['rwse_steps']))
        if task == 'time':
            self.vpos = U[S['val_edge']]
            self.vneg = sample_negatives(vr, self.vpos, pool, np.zeros(N), 20 * len(self.vpos), ukeys, N)
            return
        touch_val = S['val_gene'][U[:, 0]] | S['val_gene'][U[:, 1]]
        self.vg = np.flatnonzero(S['val_gene'])
        self.vlab = np.zeros((len(self.vg), N), bool)
        pos_of = {g: i for i, g in enumerate(self.vg)}
        for a, b in U[touch_val]:
            if a in pos_of:
                self.vlab[pos_of[a], b] = True
            if b in pos_of:
                self.vlab[pos_of[b], a] = True
        self.vcols = np.flatnonzero(~S['test'] & ~D.is_nt)
        self.vcat = Y[torch.as_tensor(self.vg, device=dev)].cpu().numpy() if n_cat else None

    def score(self, model, X, H, dev) -> float:
        """Validation AP (time: pairs; held-out: mean of link and category AP)."""
        model.eval()
        with torch.no_grad():
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                zg, zc = model.embed(X, H, self.graph)
            zg = zg.float()
            if self.task == 'time':
                pr = torch.as_tensor(np.concatenate([self.vpos, self.vneg]), device=dev)
                s = model.pair_logit(zg[pr[:, 0]], zg[pr[:, 1]]).float().cpu().numpy()
                return _average_precision(np.r_[np.ones(len(self.vpos)), np.zeros(len(self.vneg))], s)
            vg = torch.as_tensor(self.vg, device=dev)
            Sv = model.pair_matrix(zg[vg], zg).float().cpu().numpy()[:, self.vcols]
            ap_l = _average_precision(self.vlab[:, self.vcols], Sv)
            if not self.n_cat:
                return ap_l
            cl = model.cat_logits(zg[vg], zc.float() if zc is not None else None).float().cpu().numpy()
            return 0.5 * (ap_l + _average_precision(self.vcat, cl))


def run_config(D: GeneGraphData, name: str, o: Mapping[str, Any], dev: torch.device, runs_dir: Any,
               label: str = 'gene-graph', over_limit: Optional[Callable[[], bool]] = None,
               prepared: Optional[Dict[str, Any]] = None) -> str:
    """Train and score one trained-model run, resuming from ``resume.pt`` when it exists.

    Per epoch (graph models), ``node_mask`` of the supervisable genes are
    node-masked (held-out tasks) and ``edge_mask`` of the other training
    edges are hidden and supervised; negatives are ``neg_per_pos`` per
    positive, half uniform and half degree-proportional, never a known pair.
    ``lr`` supervises every training edge (a minibatch of positives) and
    every training gene. Early stopping on the inner validation keeps the
    best state. The final scoring graph holds every training edge with the
    validation genes restored; test genes stay isolated. Every epoch draws
    from ``default_rng([seed, s, epoch])`` and the resume file carries the
    torch random state, so a resumed run scores like an uninterrupted one.

    :param D: the graph.
    :param name: the run name.
    :param o: unprefixed options.
    :param dev: the device.
    :param runs_dir: the runs folder.
    :param label: the job label in the GPU log.
    :param over_limit: returns True when the run must yield; on CUDA the card
        total against ``gpu_limit_gb`` when None.
    :param prepared: the run arrays when the caller made them (fusion controls).
    :returns: ``done`` (already finished) or ``ran``.
    :raises GpuYield: when the run yielded; rerun to resume.
    """
    task, kind, arm, control, seed = parse_name(name)
    out = Path(runs_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        return 'done'
    if over_limit is None and dev.type == 'cuda':
        over_limit = functools.partial(card_over_limit, float(o['gpu_limit_gb']))
    t0 = time.time()
    S = prepared or prepare(D, task, _arm_blocks(o, arm), control, seed, o, spectral=arm == 'spectral')
    torch.manual_seed(seed)
    dims = {b: S['X'][b].shape[1] for b in S['X']}
    n_cat = D.C if S['use_cat'] else 0
    pe_k = int(o['rwse_steps'])
    model = GeneGraphModel(kind, dims, n_cat, h=int(o['hidden']), layers=int(o['layers']), heads=int(o['heads']),
                           drop=float(o['dropout']), n_types=D.n_types + int(bool(S.get('struct'))),
                           pe_k=pe_k).to(dev)
    wd = float(o['wd_linear'] if kind == 'lr' else o['wd_graph'])
    opt = torch.optim.AdamW(model.parameters(), lr=float(o['lr_linear'] if kind == 'lr' else o['lr_graph']),
                            weight_decay=wd)
    X = {b: torch.as_tensor(x, device=dev) for b, x in S['X'].items()}
    H = {b: torch.as_tensor(h, device=dev) for b, h in S['H'].items()}
    U, bits, GC = S['U'], S['bits'], S['GC']
    N = D.N
    excl = S['test'] | S['val_gene'] | D.is_nt
    pool = np.flatnonzero(~excl)
    ukeys = np.sort(S['U_all'][:, 0] * N + S['U_all'][:, 1])
    touch_val = S['val_gene'][U[:, 0]] | S['val_gene'][U[:, 1]]
    trainable = ~S['val_edge'] & ~touch_val
    deg = np.bincount(U[trainable].ravel(), minlength=N).astype(float)
    gc_train = GC[~S['val_gene'][GC[:, 0]]] if len(GC) else GC
    Y = torch.zeros(N, n_cat, device=dev) if n_cat else None
    if n_cat and len(GC):
        Y[torch.as_tensor(GC[:, 0], device=dev), torch.as_tensor(GC[:, 1], device=dev)] = 1.0
    val = _Validation(D, S, task, o, seed, pool, ukeys, trainable, n_cat, Y, dev, kind)
    max_epochs = int(o['max_epochs'])
    every, patience, min_epochs = int(o['eval_every']), int(o['patience']), int(o['min_epochs'])
    state = {'epoch': 0, 'best': -1.0, 'bad': 0, 'best_state': None, 'log': []}
    rp = out / 'resume.pt'
    if rp.exists():
        ck = torch.load(rp, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        state = ck['state']
        torch.set_rng_state(ck['torch_rng'])
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    w_pos, w_neg = float(o['pos_weight_pair']), float(o['neg_weight_pair'])
    pos_cat = float(o['pos_weight_cat'])
    while state['epoch'] < max_epochs:
        ep = state['epoch']
        rng = np.random.default_rng([int(o['seed']), seed, ep])
        model.train()
        if kind == 'lr':
            msg = np.zeros(len(U), bool)
            pos = U[trainable]
            mb = int(o['lr_pos_minibatch'])
            if len(pos) > mb:
                pos = pos[rng.choice(len(pos), mb, replace=False)]
            masked_genes = np.flatnonzero(~excl) if n_cat else np.zeros(0, int)
            gc_msg = gc_train[:0]
        else:
            nm = np.zeros(N, bool)
            if n_cat:
                nm[rng.choice(pool, int(float(o['node_mask']) * len(pool)), replace=False)] = True
            touch_nm = nm[U[:, 0]] | nm[U[:, 1]]
            em = (rng.random(len(U)) < float(o['edge_mask'])) & ~touch_nm
            msg = trainable & ~touch_nm & ~em
            pos = U[trainable & (touch_nm | em)]
            masked_genes = np.flatnonzero(nm)
            gc_msg = gc_train[~nm[gc_train[:, 0]]] if len(gc_train) else gc_train
        g = build_graph(D, S, U[msg], bits[msg], gc_msg, dev, kind, pe_k)
        neg = sample_negatives(rng, pos, pool, deg, int(o['neg_per_pos']) * len(pos), ukeys, N)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            zg, zc = model.embed(X, H, g)
            zg = zg.float()
            pr = torch.as_tensor(np.concatenate([pos, neg]).reshape(-1, 2), device=dev)
            lab = torch.cat([torch.ones(len(pos), device=dev), torch.zeros(len(neg), device=dev)])
            w = torch.cat([torch.full((len(pos),), w_pos, device=dev), torch.full((len(neg),), w_neg, device=dev)])
            loss = F.binary_cross_entropy_with_logits(model.pair_logit(zg[pr[:, 0]], zg[pr[:, 1]]), lab, weight=w)
            if n_cat and len(masked_genes):
                mg = torch.as_tensor(masked_genes, device=dev)
                cl = model.cat_logits(zg[mg], zc.float() if zc is not None else None).float()
                loss = loss + F.binary_cross_entropy_with_logits(cl, Y[mg], pos_weight=torch.tensor(pos_cat, device=dev))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        state['epoch'] = ep + 1
        if (ep + 1) % every == 0 or (state['best_state'] is None and ep + 1 == max_epochs):
            v = val.score(model, X, H, dev)
            state['log'].append({'epoch': ep + 1, 'loss': float(loss), 'val': v, 't': time.time() - t0})
            if v > state['best'] + 1e-4 or state['best_state'] is None:
                state['best'], state['bad'] = (v if v == v else state['best']), 0
                state['best_state'] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                state['bad'] += 1
            if state['bad'] >= patience and ep + 1 >= min_epochs:
                break
        if over_limit is not None and state['epoch'] < max_epochs and over_limit():
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'state': state,
                        'torch_rng': torch.get_rng_state()}, rp)
            if dev.type == 'cuda':
                gpu_log(o['gpu_log'], label, f"YIELD card total > {o['gpu_limit_gb']} GB at {name} epoch {ep + 1}; "
                                             f"resume.pt saved, exit {YIELD_RC}")
            raise GpuYield(name)
    model.load_state_dict(state['best_state'])
    model.eval()
    gfin = build_graph(D, S, U, bits, GC, dev, kind, pe_k)
    with torch.no_grad():
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            zg, zc = model.embed(X, H, gfin)
        zg = zg.float()
        zc = zc.float() if zc is not None else None
        if task == 'time':
            np.savez_compressed(out / 'scores.npz', S=model.pair_matrix(zg, zg).float().cpu().numpy())
        else:
            rows = np.flatnonzero(S['test'])
            r = torch.as_tensor(rows, device=dev)
            np.savez_compressed(out / 'scores.npz', rows=rows, S=model.pair_matrix(zg[r], zg).float().cpu().numpy(),
                                C=model.cat_logits(zg[r], zc).float().cpu().numpy())
    peak = torch.cuda.max_memory_reserved() / 2 ** 30 if dev.type == 'cuda' else 0.0
    _write(pd.DataFrame(state['log'], columns=['epoch', 'loss', 'val', 't']), out / 'log.csv')
    _write_json({'name': name, 'task': task, 'model': kind, 'arm': arm, 'control': control, 'seed': seed,
                 'blocks': [b for b in S['X']] + (['struct'] if S.get('struct') else []), 'dims': dims,
                 'epochs': state['epoch'], 'best_val': state['best'], 'seconds': time.time() - t0,
                 'device': dev.type, 'peak_reserved_gb': peak, 'hparams': _hparams(o, kind, n_cat, max_epochs)},
                out / 'done.json')
    if rp.exists():
        rp.unlink()
    return 'ran'


def _cos_block(X: np.ndarray, H: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine of rows x all nodes within one block, and where both have the block."""
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = Xn[rows] @ Xn.T
    M = (H[rows][:, None] & H[None, :]).astype(np.float32)
    return S * M, M


def cosine_scores(X: Mapping[str, np.ndarray], H: Mapping[str, np.ndarray], rows: np.ndarray,
                  N: int) -> np.ndarray:
    """Per-block cosine averaged over the blocks both genes have.

    :param X: block -> features.
    :param H: block -> presence.
    :param rows: the scored rows.
    :param N: number of nodes.
    :returns: rows x N similarities.
    """
    num = np.zeros((len(rows), N), np.float32)
    den = np.zeros((len(rows), N), np.float32)
    for b in X:
        sb, mb = _cos_block(np.asarray(X[b], np.float32), np.asarray(H[b], bool), rows)
        num += sb
        den += mb
    return num / np.maximum(den, 1)


def knn_vote(Sm: np.ndarray, GC: np.ndarray, N: int, C: int, k: int = 25) -> np.ndarray:
    """Category scores: similarity-weighted vote of the ``k`` most similar training genes.

    :param Sm: rows x N similarities.
    :param GC: training memberships.
    :param N: number of nodes.
    :param C: number of categories.
    :param k: neighbours.
    :returns: rows x C scores.
    """
    Y = np.zeros((N, C), np.float32)
    Y[GC[:, 0], GC[:, 1]] = 1
    train = np.zeros(N, bool)
    train[np.unique(GC[:, 0])] = True
    Sk = np.where(train[None, :], Sm, -np.inf)
    k = min(k, max(int(train.sum()) - 1, 1))
    top = np.argpartition(-Sk, k, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(Sm, top, 1), 0, None)
    return np.einsum('rk,rkc->rc', w, Y[top]) / np.maximum(w.sum(1, keepdims=True), 1e-6)


def run_baseline(D: GeneGraphData, name: str, o: Mapping[str, Any], runs_dir: Any,
                 prepared: Optional[Dict[str, Any]] = None) -> str:
    """Score one training-free run: ``cosine``, ``adamicadar`` or ``prior``.

    :param D: the graph.
    :param name: the run name.
    :param o: unprefixed options.
    :param runs_dir: the runs folder.
    :param prepared: the run arrays when the caller made them.
    :returns: ``done`` or ``ran``.
    :raises KeyError: for an unknown baseline.
    """
    task, kind, arm, control, seed = parse_name(name)
    out = Path(runs_dir) / name
    if (out / 'done.json').exists():
        return 'done'
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    arms = o['arms'] or {}
    S = prepared or prepare(D, task, list(arms[arm]) if arm in arms else [], control, seed, o)
    N = D.N
    rows = np.arange(N) if task == 'time' else np.flatnonzero(S['test'])
    U = S['U']
    deg = np.bincount(U.ravel(), minlength=N).astype(np.float32)
    C = None
    if kind == 'cosine':
        Sm = cosine_scores(S['X'], S['H'], rows, N)
        if task != 'time':
            C = knn_vote(Sm, S['GC'], N, D.C, int(o['knn_vote_k']))
    elif kind == 'adamicadar':
        import scipy.sparse as sp

        A = sp.csr_matrix((np.ones(2 * len(U), np.float32), (np.r_[U[:, 0], U[:, 1]], np.r_[U[:, 1], U[:, 0]])),
                          shape=(N, N))
        w = 1 / np.log(np.maximum(deg, 2))
        Sm = (A @ sp.diags(w) @ A).toarray()[rows].astype(np.float32)
    elif kind == 'prior':
        Sm = (deg[rows][:, None] * deg[None, :]) if task == 'time' else np.repeat(deg[None, :], len(rows), 0)
        if task != 'time':
            size = np.bincount(S['GC'][:, 1], minlength=D.C).astype(np.float32)
            C = np.repeat(size[None, :], len(rows), 0)
    else:
        raise KeyError(f'unknown baseline {kind!r}')
    if task == 'time':
        np.savez_compressed(out / 'scores.npz', S=Sm.astype(np.float32))
    else:
        np.savez_compressed(out / 'scores.npz', rows=rows, S=Sm.astype(np.float32),
                            C=(C if C is not None else np.zeros((len(rows), D.C))).astype(np.float32))
    _write_json({'name': name, 'task': task, 'model': kind, 'arm': arm, 'control': control,
                 'blocks': list(arms.get(arm, [])), 'seconds': time.time() - t0, 'device': 'cpu',
                 'k_vote': int(o['knn_vote_k']) if kind == 'cosine' else None}, out / 'done.json')
    return 'ran'


def default_configs(o: Mapping[str, Any]) -> List[str]:
    """The run names derived from the tasks, models, arms, controls and seeds.

    :param o: unprefixed options.
    :returns: run names; ``gene_graph_configs`` when it is not empty.
    """
    if o.get('configs'):
        return list(o['configs'])
    arms = list((o['arms'] or {}).keys())
    names: List[str] = []
    for task in o['tasks']:
        for model in o['models']:
            if model == 'spectral':
                names += [f'{task}__lr__spectral__none__s{s}' for s in o['seeds']]
            elif model in TRAINED_MODELS:
                names += [f'{task}__{model}__{a}__none__s{s}' for a in arms for s in o['seeds']]
            elif model == 'cosine':
                names += [f'{task}__cosine__{a}__none__s0' for a in arms if (o['arms'] or {})[a]]
            elif model == 'prior' or (model == 'adamicadar' and task == 'time'):
                names.append(f'{task}__{model}__graph__none__s0')
        for entry in o['control_runs'] or []:
            parts = entry.split(':')
            model, arm = parts[0], parts[1]
            controls = [parts[2]] if len(parts) > 2 else list(o['controls'])
            if model not in o['models'] or arm not in arms:
                continue
            seeds = o['seeds'] if model in TRAINED_MODELS else [0]
            names += [f'{task}__{model}__{arm}__{c}__s{s}' for c in controls if c in o['controls'] for s in seeds]
    return list(dict.fromkeys(names))


def _cuda_hours(runs_dir: Path) -> float:
    """GPU hours of the finished CUDA runs of a runs folder."""
    used = 0.0
    for f in sorted(runs_dir.glob('*/done.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        if j.get('device') == 'cuda':
            used += float(j.get('seconds', 0)) / 3600
    return used


def train_runs(D: GeneGraphData, names: Sequence[str], o: Mapping[str, Any], runs_dir: Any,
               label: str = 'gene-graph', over_limit: Optional[Callable[[], bool]] = None,
               prepare_fn: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Run a queue of configs: baselines on the CPU, trained models on the chosen device.

    On CUDA the queue is admitted under the card rule, stops at the GPU
    budget and yields when the card goes over the limit.

    :param D: the graph.
    :param names: run names.
    :param o: unprefixed options.
    :param runs_dir: the runs folder.
    :param label: the job label in the GPU log.
    :param over_limit: the yield test (see :func:`run_config`).
    :param prepare_fn: makes the run arrays from a name (fusion arms); None
        uses :func:`prepare`.
    :returns: ``status`` (``done``, ``yielded`` or ``budget``), ``ran`` and ``device``.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (runs_dir / n / 'done.json').exists()]
    trained = [n for n in todo if parse_name(n)[1] in TRAINED_MODELS]
    dev = _device(o) if trained else torch.device('cpu')
    ran: List[str] = []
    status = 'done'
    if dev.type == 'cuda':
        cap = float(o['gpu_cap_gb'])
        u = admit(cap + 0.5, float(o['gpu_limit_gb']), float(o['admission_hold_s']))
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / (torch.cuda.mem_get_info()[1] / 2 ** 30)))
        torch.backends.cuda.matmul.allow_tf32 = True
        gpu_log(o['gpu_log'], label, f"START single process, cap {cap:.0f} GB, card admission/yield "
                                     f"{float(o['gpu_limit_gb']):.0f} GB (card {u:.1f} GB at admission), "
                                     f"{len(trained)} configs, {runs_dir}")
    previous = torch.are_deterministic_algorithms_enabled()
    if dev.type == 'cpu':
        torch.use_deterministic_algorithms(True)
    try:
        for n in todo:
            kind = parse_name(n)[1]
            S = prepare_fn(n) if prepare_fn is not None else None
            if kind not in TRAINED_MODELS:
                run_baseline(D, n, o, runs_dir, prepared=S)
                ran.append(n)
                continue
            if dev.type == 'cuda':
                used = _cuda_hours(runs_dir)
                if used > float(o['gpu_budget_h']):
                    gpu_log(o['gpu_log'], label, f"STOP GPU budget {o['gpu_budget_h']} h reached ({used:.2f} h); "
                                                 "ask before continuing")
                    status = 'budget'
                    break
            try:
                run_config(D, n, o, dev, runs_dir, label, over_limit, prepared=S)
            except GpuYield:
                status = 'yielded'
                break
            ran.append(n)
            logger.info('%s ran', n)
    finally:
        torch.use_deterministic_algorithms(previous)
        if dev.type == 'cuda':
            gpu_log(o['gpu_log'], label, f'FINISH (GPU time used so far {_cuda_hours(runs_dir):.2f} h)')
    return {'status': status, 'ran': ran, 'device': dev.type}


def train_gene_graph(settings: Mapping[str, Any], over_limit: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Run every configured model on the assembled graph.

    :param settings: the ``gene_graph_*`` settings.
    :param over_limit: the yield test (tests force a yield with it).
    :returns: see :func:`train_runs`.
    """
    o = options(settings)
    p = paths(o)
    D = GeneGraphData(p['data'])
    return train_runs(D, default_configs(o), o, p['runs'], over_limit=over_limit)


def wap(order: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Average precision of pre-sorted scores, with optional sample weights.

    :param order: indices in descending score order.
    :param y: labels.
    :param w: sample weights.
    :returns: the AP; NaN without positives.
    """
    yy = y[order].astype(np.float64)
    ww = np.ones_like(yy) if w is None else w[order]
    tp = np.cumsum(yy * ww)
    tot = np.cumsum(ww)
    pos = (yy * ww).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(((yy * ww) * (tp / tot)).sum() / pos) if pos > 0 else float('nan')


class Reference:
    """The evaluation labels of an assembled graph: training, new and current edges, categories, strata.

    :param D: the graph.
    """

    def __init__(self, D: GeneGraphData):
        """Build the label matrices, the degree deciles, the fitness tertiles and the time-split candidates."""
        z = D.z
        self.D = D
        self.nodes = D.nodes
        self.N = N = D.N
        self.real = D.real
        self.ridx = np.flatnonzero(self.real)
        self.C = D.C
        self.sources = list(D.sources)
        self.train = self._mat([f'E_train_{k}' for k in self.sources])
        self.new = {k: self._mat([f'E_new_{k}']) for k in self.sources}
        anyn = np.zeros((N, N), bool)
        for m in self.new.values():
            anyn |= m
        self.new['any'] = anyn
        self.cur = self._mat([f'E_cur_{k}' for k in self.sources])
        self.GC = D.GC
        self.Y = np.zeros((N, self.C), bool)
        self.Y[self.GC[:, 0], self.GC[:, 1]] = True
        self.deg = self.train.sum(1)
        f = self.nodes.log_cells_per_guide.to_numpy()
        q = np.quantile(f[self.real], [1 / 3, 2 / 3])
        self.fit_t = np.searchsorted(q, f)
        dq = np.quantile(self.deg[self.real], np.linspace(0, 1, 11)[1:-1])
        self.deg_dec = np.searchsorted(dq, self.deg, side='right')
        self.fold = self.nodes.fold.to_numpy()
        r = self.ridx
        iu = np.triu_indices(len(r), 1)
        i, j = r[iu[0]], r[iu[1]]
        keep = ~self.train[i, j]
        self.ti, self.tj = i[keep].astype(np.int32), j[keep].astype(np.int32)
        self.tlab = {k: v[self.ti, self.tj] for k, v in self.new.items()}
        dd = self.deg_dec
        self.tcombo = (np.minimum(dd[self.ti], dd[self.tj]) * 10 + np.maximum(dd[self.ti], dd[self.tj])).astype(np.int16)
        self.tfit = np.minimum(self.fit_t[self.ti], self.fit_t[self.tj])
        self.combo_index = {c: np.flatnonzero(self.tcombo == c) for c in np.unique(self.tcombo)}
        self.nt_idx = np.flatnonzero(~self.real)

    def _mat(self, keys: Sequence[str]) -> np.ndarray:
        """Symmetric boolean matrix of the edges stored under ``keys``."""
        M = np.zeros((self.N, self.N), bool)
        for k in keys:
            e = self.D.z.get(k)
            if e is not None and len(e):
                M[e[:, 0], e[:, 1]] = True
        return M | M.T

    def matrix(self, key: str) -> np.ndarray:
        """Symmetric boolean matrix of one stored edge array, such as ``E_cur_STRING04``.

        :param key: the array name.
        :returns: the matrix (all False when the array is absent).
        """
        return self._mat([key])


def time_metrics(R: Reference, S: np.ndarray, name: str, o: Mapping[str, Any],
                 seed: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Time-split metrics of one score matrix.

    AUPRC overall and per source with a gene-bootstrap interval, precision
    and recall at k, the degree-matched random-pair control, AUPRC within
    fitness tertiles, the hub and fitness fractions of the top 1,000 and the
    non-targeting enrichment above the real top-1,000 and top-10,000
    thresholds. Ties are broken by a seeded jitter of 1e-12 times max |s|.

    :param R: the labels.
    :param S: N x N scores.
    :param name: the run name.
    :param o: unprefixed options (``topk``, ``bootstrap_reps``).
    :param seed: the jitter, control and bootstrap seed.
    :returns: the metrics row and the bootstrap rows.
    """
    s = S[R.ti, R.tj].astype(np.float64)
    rng = np.random.default_rng(seed)
    s = s + rng.random(len(s)) * 1e-12 * (np.abs(s).max() + 1)
    order = np.argsort(-s, kind='stable')
    row: Dict[str, Any] = {'name': name, 'candidates': len(s)}
    for k, y in R.tlab.items():
        row[f'AUPRC_{k}'] = wap(order, y)
        row[f'base_{k}'] = float(y.mean())
    y = R.tlab['any']
    for k in o['topk']:
        k = int(k)
        top = order[:k]
        row[f'P@{k}'] = float(y[top].mean())
        row[f'R@{k}'] = float(y[top].sum() / y.sum()) if y.sum() else float('nan')
        combos, cnt = np.unique(R.tcombo[top], return_counts=True)
        hits, tot = 0, 0
        for c, n in zip(combos, cnt):
            pool = R.combo_index[c]
            d = pool[rng.integers(len(pool), size=10 * n)]
            hits += y[d].sum()
            tot += len(d)
        row[f'P@{k}_degmatched'] = hits / tot
        row[f'enrich@{k}_vs_degmatched'] = row[f'P@{k}'] / max(row[f'P@{k}_degmatched'], 1e-9)
    top = order[:1000]
    row['top1000_frac_top_degree_decile'] = float(((R.deg_dec[R.ti[top]] == 9) | (R.deg_dec[R.tj[top]] == 9)).mean())
    row['all_frac_top_degree_decile'] = float(((R.deg_dec[R.ti] == 9) | (R.deg_dec[R.tj] == 9)).mean())
    row['top1000_frac_fitness_low_tertile'] = float((R.tfit[top] == 0).mean())
    row['all_frac_fitness_low_tertile'] = float((R.tfit == 0).mean())
    for t in range(3):
        k = R.tfit == t
        o_t = np.argsort(-s[k], kind='stable')
        row[f'AUPRC_any_fit{t}'] = wap(o_t, y[k])
        row[f'base_any_fit{t}'] = float(y[k].mean()) if k.any() else float('nan')
    n1k, n10k = min(1000, len(s)), min(10000, len(s))
    nts = S[np.ix_(R.nt_idx, R.ridx)].ravel()
    thr = s[order[n1k - 1]]
    row['NT_frac_above_top1000'] = float((nts >= thr).mean()) if len(nts) else float('nan')
    row['NT_enrichment_top1000'] = row['NT_frac_above_top1000'] / (n1k / len(s))
    thr2 = s[order[n10k - 1]]
    row['NT_enrichment_top10000'] = (float((nts >= thr2).mean()) if len(nts) else float('nan')) / (n10k / len(s))
    boots = []
    for b in range(int(o['bootstrap_reps'])):
        w = np.bincount(rng.integers(0, len(R.ridx), len(R.ridx)), minlength=len(R.ridx)).astype(np.float64)
        wn = np.zeros(R.N)
        wn[R.ridx] = w
        boots.append({'name': name, 'b': b, 'AUPRC_any': wap(order, y, wn[R.ti] * wn[R.tj])})
    bb = np.array([x['AUPRC_any'] for x in boots], float)
    ok = bb[np.isfinite(bb)]
    row['AUPRC_any_lo'] = float(np.percentile(ok, 2.5)) if len(ok) else float('nan')
    row['AUPRC_any_hi'] = float(np.percentile(ok, 97.5)) if len(ok) else float('nan')
    return row, boots


def fold_metrics(R: Reference, z: Mapping[str, np.ndarray], name: str) -> Dict[str, Any]:
    """Held-out metrics of one fold's scores.

    Links of test genes to real non-test genes against the 2022 and the
    current edges (pooled AUPRC, gene mAP), the top-5 partners' hub fraction,
    precision and degree-matched precision; categories (micro and macro
    AUPRC, recall at 10, micro AUPRC per fitness tertile) and the
    non-targeting enrichment of categories (top 1 %) and links (top 0.1 %).

    :param R: the labels.
    :param z: ``rows``, ``S`` (rows x N) and ``C`` (rows x categories).
    :param name: the run name.
    :returns: the metrics row.
    """
    from sklearn.metrics import average_precision_score

    rows, S, C = z['rows'], z['S'].astype(np.float64), z['C'].astype(np.float64)
    test = np.zeros(R.N, bool)
    test[rows] = True
    cols = np.flatnonzero(R.real & ~test)
    rr = R.real[rows]
    out: Dict[str, Any] = {'name': name, 'test_genes': int(rr.sum()), 'test_nt': int((~rr).sum())}
    Sr = S[rr][:, cols]
    rng = np.random.default_rng(0)
    Sr = Sr + rng.random(Sr.shape) * 1e-12 * (np.abs(Sr).max() + 1)
    for ref, M in (('2022', R.train), ('current', R.cur)):
        L = M[np.ix_(rows[rr], cols)]
        out[f'link_AUPRC_{ref}'] = float(average_precision_score(L.ravel(), Sr.ravel())) if L.any() else np.nan
        out[f'link_base_{ref}'] = float(L.mean())
        aps = [average_precision_score(L[i], Sr[i]) for i in range(len(L)) if L[i].any()]
        out[f'link_geneMAP_{ref}'] = float(np.mean(aps)) if aps else np.nan
    L = R.train[np.ix_(rows[rr], cols)]
    top = np.argsort(-Sr, 1)[:, :5]
    out['link_top5_frac_top_degree_decile'] = float((R.deg_dec[cols[top]] == 9).mean())
    out['link_top5_precision'] = float(np.take_along_axis(L, top, 1).mean())
    dd = R.deg_dec[cols]
    hit = []
    for i in range(len(top)):
        for c in top[i]:
            pool = np.flatnonzero(dd == dd[c])
            d = pool[rng.integers(len(pool), size=10)]
            hit.append(L[i, d].mean())
    out['link_top5_precision_degmatched'] = float(np.mean(hit)) if hit else np.nan
    Y = R.Y[rows[rr]]
    Cr = C[rr] + rng.random(C[rr].shape) * 1e-12 * (np.abs(C).max() + 1)
    has = Y.sum(1) > 0
    out['cat_micro_AUPRC'] = float(average_precision_score(Y[has].ravel(), Cr[has].ravel())) if has.any() else np.nan
    out['cat_base'] = float(Y[has].mean()) if has.any() else np.nan
    cl = [average_precision_score(Y[has][:, c], Cr[has][:, c]) for c in range(R.C) if Y[has][:, c].any()]
    out['cat_macro_AUPRC'] = float(np.mean(cl)) if cl else np.nan
    if has.any():
        t10 = np.argsort(-Cr[has], 1)[:, :10]
        out['cat_recall@10'] = float((np.take_along_axis(Y[has], t10, 1).sum(1) / Y[has].sum(1)).mean())
    else:
        out['cat_recall@10'] = np.nan
    for t in range(3):
        k = has & (R.fit_t[rows[rr]] == t)
        out[f'cat_micro_AUPRC_fit{t}'] = float(average_precision_score(Y[k].ravel(), Cr[k].ravel())) if Y[k].any() else np.nan
    if (~rr).any() and has.any():
        thr = np.quantile(Cr[has].ravel(), 0.99)
        out['NT_cat_enrichment_top1pct'] = float((C[~rr] >= thr).mean() / 0.01)
        thr_l = np.quantile(Sr.ravel(), 0.999)
        out['NT_link_enrichment_top0.1pct'] = float((S[~rr][:, cols] >= thr_l).mean() / 0.001)
    return out


_KEY = ('task', 'model', 'arm', 'control', 'seed')

PAL = {'full': '#1f3a5f', 'noJUMP': '#2a9d8f', 'noOPS': '#3a7ca5', 'graph': '#8ecae6', 'spectral': '#8ecae6',
       'tvnfull': '#264653', 'tvnnoJUMP': '#4fb3a9', 'control': '#9aa5b1', 'prior': '#c3ccd6'}
ARM_LABEL = {'full': 'OPS + JUMP', 'noJUMP': 'OPS only', 'noOPS': 'JUMP only', 'graph': 'graph only',
             'spectral': 'graph only', 'tvnfull': 'OPS(TVN) + JUMP', 'tvnnoJUMP': 'OPS(TVN) only'}
CTRL_LABEL = {'shufX': 'shuffled features', 'shufY': 'shuffled labels', 'rewire': 'rewired edges',
              'shufOPS': 'shuffled screen profile', 'shufScreen': 'shuffled screen blocks'}
MODEL_LABEL = {'gps': 'graph transformer', 'sage': 'GraphSAGE', 'gcn': 'GCN', 'lr': 'logistic regr.',
               'cosine': 'cosine similarity', 'prior': 'degree / size prior', 'adamicadar': 'Adamic-Adar',
               'late': 'late fusion'}
ORDER_M = ['gps', 'sage', 'gcn', 'lr', 'cosine', 'late', 'adamicadar', 'prior']
ORDER_A = ['full', 'noJUMP', 'noOPS', 'graph', 'spectral', 'tvnfull', 'tvnnoJUMP']


def run_label(r: Any) -> str:
    """The plain-language label of a metrics row (model, arm, control, seed).

    :param r: a row with ``model``, ``arm``, ``control`` and ``seed``.
    :returns: the label.
    """
    s = MODEL_LABEL.get(r.model, r.model)
    if r.model in ('prior', 'adamicadar'):
        return s
    a = ARM_LABEL.get(r.arm, r.arm)
    if r.model == 'lr' and r.arm == 'spectral':
        return 'spectral emb. + LR (graph only)'
    if r.control != 'none':
        return f'{s}, {a}: {CTRL_LABEL.get(r.control, r.control)}'
    seed = int(r.seed) if pd.notna(getattr(r, 'seed', 0)) else 0
    return f'{s}, {a}' + (f' (seed {seed})' if seed else '')


def sort_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Rows in figure order: controls last, then by model, arm and seed.

    :param d: rows with ``model``, ``arm``, ``control`` and ``seed``.
    :returns: the sorted rows.
    """
    d = d.copy()
    d['_m'] = d.model.map({m: i for i, m in enumerate(ORDER_M)}).fillna(99)
    d['_a'] = d.arm.map({m: i for i, m in enumerate(ORDER_A)}).fillna(99)
    d['_c'] = (d.control != 'none').astype(int)
    return d.sort_values(['_c', '_m', '_a', 'seed'], kind='stable').drop(columns=['_m', '_a', '_c'])


def seed0(seed: pd.Series) -> pd.Series:
    """Seeds as numbers, a missing seed (training-free baselines) as 0.

    :param seed: the seed column.
    :returns: the filled column.
    """
    return pd.to_numeric(seed, errors='coerce').fillna(0)


def heldout_summary(F_: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over folds per model x arm x control x seed.

    :param F_: per-fold metrics.
    :returns: the summary with ``folds``.
    """
    F_ = F_.copy()
    F_['seed'] = seed0(F_['seed'])
    num = [c for c in F_.columns if c not in ('task', 'model', 'arm', 'control', 'seed', 'name')]
    g = F_.groupby(['model', 'arm', 'control', 'seed'])[num]
    return g.mean().add_suffix('_mean').join(g.std().add_suffix('_sd')).join(g.size().rename('folds')).reset_index()


def paired_deltas(F_: pd.DataFrame, comparisons: Sequence[str],
                  metrics: Sequence[str] = ('link_AUPRC_2022', 'cat_micro_AUPRC', 'cat_macro_AUPRC'),
                  seed: int = 0) -> pd.DataFrame:
    """Paired fold deltas of A over B, with the count of folds where A is better.

    :param F_: per-fold metrics.
    :param comparisons: ``'model/arm[/control]|model/arm[/control]'``.
    :param metrics: metric columns.
    :param seed: the seed of both runs.
    :returns: metric, A, B, A_mean, B_mean, delta, delta_min, delta_max, folds_A_better, n.
    """
    F_ = F_.copy()
    F_['seed'] = seed0(F_['seed'])
    rows = []
    for metric in metrics:
        for comp in comparisons:
            a, b = comp.split('|')
            va, vb = _fold_series(F_, a, metric, seed), _fold_series(F_, b, metric, seed)
            j = pd.concat([va.rename('A'), vb.rename('B')], axis=1).dropna()
            if not len(j):
                continue
            d = j.A - j.B
            rows.append({'metric': metric, 'A': a, 'B': b, 'A_mean': float(j.A.mean()), 'B_mean': float(j.B.mean()),
                         'delta': float(d.mean()), 'delta_min': float(d.min()), 'delta_max': float(d.max()),
                         'folds_A_better': int((d > 0).sum()), 'n': int(len(d))})
    return pd.DataFrame(rows, columns=['metric', 'A', 'B', 'A_mean', 'B_mean', 'delta', 'delta_min', 'delta_max',
                                       'folds_A_better', 'n'])


def _fold_series(F_: pd.DataFrame, spec: str, metric: str, seed: int) -> pd.Series:
    """One run's metric per task, from ``model/arm[/control]``."""
    parts = spec.split('/')
    control = parts[2] if len(parts) > 2 else 'none'
    k = (F_.model == parts[0]) & (F_.arm == parts[1]) & (F_.control == control) & (F_.seed == seed)
    return F_[k].set_index('task')[metric] if metric in F_ else pd.Series(dtype=float)


def ablation_table(T: pd.DataFrame, S: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """The headline table: time split and held-out metrics of every model x arm x control.

    :param T: time-split metrics.
    :param S: held-out summary.
    :returns: the table and its Markdown.
    """
    T, S = T.copy(), S.copy()
    for d in (T, S):
        if len(d):
            d['seed'] = seed0(d.seed)
    key = ['model', 'arm', 'control', 'seed']
    tc = ['AUPRC_any', 'AUPRC_any_lo', 'AUPRC_any_hi'] + [f'AUPRC_{k}' for k in ('STRING', 'BioGRID', 'CORUM')] + [
        'P@1000', 'P@1000_degmatched', 'enrich@1000_vs_degmatched', 'R@10000', 'NT_enrichment_top1000',
        'top1000_frac_fitness_low_tertile', 'top1000_frac_top_degree_decile']
    sc = ['cat_micro_AUPRC_mean', 'cat_micro_AUPRC_sd', 'cat_macro_AUPRC_mean', 'cat_recall@10_mean',
          'link_AUPRC_2022_mean', 'link_AUPRC_2022_sd', 'link_AUPRC_current_mean', 'link_geneMAP_2022_mean',
          'link_top5_precision_mean', 'link_top5_precision_degmatched_mean', 'link_top5_frac_top_degree_decile_mean',
          'NT_cat_enrichment_top1pct_mean', 'NT_link_enrichment_top0.1pct_mean']
    Tk = T[key + [c for c in tc if c in T]] if len(T) else pd.DataFrame(columns=key)
    Sk = S[key + [c for c in sc if c in S]] if len(S) else pd.DataFrame(columns=key)
    A = Tk.merge(Sk, on=key, how='outer')
    for c in tc + sc:
        if c not in A:
            A[c] = np.nan
    A = A[key + tc + sc]
    A['label'] = A.apply(run_label, axis=1) if len(A) else []
    A = sort_rows(A) if len(A) else A
    show = {'label': 'model, arm', 'AUPRC_any': 'time AUPRC', 'P@1000': 'time P@1k', 'P@1000_degmatched': 'P@1k deg-matched',
            'NT_enrichment_top1000': 'NT enr. (time)', 'cat_micro_AUPRC_mean': 'held-out cat. micro AUPRC',
            'cat_macro_AUPRC_mean': 'cat. macro AUPRC', 'link_AUPRC_2022_mean': 'held-out link AUPRC',
            'link_AUPRC_current_mean': 'link AUPRC (current)', 'NT_cat_enrichment_top1pct_mean': 'NT enr. (cat.)'}
    M = A[list(show)].rename(columns=show)
    lines = ['| ' + ' | '.join(M.columns) + ' |', '|' + '---|' * len(M.columns)]
    for r in M.itertuples(index=False):
        lines.append('| ' + ' | '.join(x if isinstance(x, str) else ('–' if pd.isna(x) else f'{x:.4f}') for x in r) + ' |')
    base_t = T.base_any.mean() if 'base_any' in T else float('nan')
    base_c = S.cat_base_mean.mean() if 'cat_base_mean' in S else float('nan')
    base_l = S.link_base_2022_mean.mean() if 'link_base_2022_mean' in S else float('nan')
    lines.append(f'\nBase rates: time split {base_t:.4f}; held-out category {base_c:.4f}; '
                 f'held-out link (2022) {base_l:.4f}.')
    return A, '\n'.join(lines) + '\n'


def evaluate_runs(R: Reference, runs_dir: Any, o: Mapping[str, Any],
                  names: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """Metrics of every finished run of a runs folder.

    :param R: the labels.
    :param runs_dir: the runs folder.
    :param o: unprefixed options.
    :param names: only these runs; every finished run when None.
    :returns: ``time``, ``boot`` and ``fold`` frames.
    """
    trows, frows, brows = [], [], []
    for d in sorted(Path(runs_dir).glob('*/done.json')):
        j = json.loads(d.read_text(encoding='utf-8'))
        if names is not None and j['name'] not in names:
            continue
        z = np.load(d.parent / 'scores.npz')
        meta = {k: j.get(k) for k in _KEY}
        if j['task'] == 'time':
            r, b = time_metrics(R, z['S'].astype(np.float32), j['name'], o)
            trows.append({**meta, **r})
            brows += b
        else:
            frows.append({**meta, **fold_metrics(R, z, j['name'])})
    return {'time': pd.DataFrame(trows), 'boot': pd.DataFrame(brows, columns=['name', 'b', 'AUPRC_any']),
            'fold': pd.DataFrame(frows)}


def run_configs_table(runs_dir: Any) -> pd.DataFrame:
    """One row per finished run with its ``done.json`` fields.

    :param runs_dir: the runs folder.
    :returns: the table.
    """
    rows = []
    for d in sorted(Path(runs_dir).glob('*/done.json')):
        j = json.loads(d.read_text(encoding='utf-8'))
        rows.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in j.items()})
    return pd.DataFrame(rows)


def evaluate_gene_graph(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Write the time-split and held-out metrics, summaries, ablation table and paired deltas.

    :param settings: the ``gene_graph_*`` settings.
    :returns: the frames written, by table name.
    """
    o = options(settings)
    p = paths(o)
    R = Reference(GeneGraphData(p['data']))
    ev = evaluate_runs(R, p['runs'], o)
    t = p['tables']
    t.mkdir(parents=True, exist_ok=True)
    out = {'time_split_metrics': ev['time'], 'time_split_bootstrap': ev['boot'],
           'heldout_metrics_per_fold': ev['fold']}
    out['heldout_summary'] = heldout_summary(ev['fold']) if len(ev['fold']) else pd.DataFrame()
    out['ablation_table'], md = ablation_table(ev['time'], out['heldout_summary'])
    out['heldout_paired_deltas'] = (paired_deltas(ev['fold'], o['paired_deltas'] or []) if len(ev['fold'])
                                    else pd.DataFrame())
    out['run_configs'] = run_configs_table(p['runs'])
    for name, frame in out.items():
        _write(frame, t / f'{name}.csv')
    (t / 'ablation_table.md').write_text(md, encoding='utf-8')
    return out


def load_oof(runs_dir: Any, model: str, arm: str, n_folds: int, control: str = 'none',
             seed: int = 0) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
    """The out-of-fold scores of one model x arm, or None when a fold is missing.

    :param runs_dir: the runs folder.
    :param model: the model.
    :param arm: the arm.
    :param n_folds: number of folds.
    :param control: the control.
    :param seed: the seed.
    :returns: ``{fold: {rows, S, C}}``.
    """
    out = {}
    for f in range(n_folds):
        d = Path(runs_dir) / f'fold{f}__{model}__{arm}__{control}__s{seed}'
        if not (d / 'done.json').exists():
            return None
        z = np.load(d / 'scores.npz')
        out[f] = {'rows': z['rows'], 'S': z['S'].astype(np.float32), 'C': z['C'].astype(np.float32)}
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins.

    :param p: probabilities.
    :param y: labels.
    :param bins: number of bins.
    :returns: the ECE.
    """
    o = np.argsort(p)
    e = 0.0
    for b in np.array_split(o, bins):
        if len(b):
            e += len(b) / len(p) * abs(p[b].mean() - y[b].mean())
    return float(e)


def _isotonic(x: np.ndarray, y: np.ndarray):
    """An isotonic map of score to probability, clipped to [0, 1]."""
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x, y)


def cat_calibrated(R: Reference, oof: Mapping[int, Mapping[str, np.ndarray]], rng: np.random.Generator,
                   max_fit: int = 3_000_000) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Cross-fitted calibrated category probabilities (each fold mapped by the other folds' fit).

    :param R: the labels.
    :param oof: out-of-fold scores.
    :param rng: the subsample stream.
    :param max_fit: at most this many scores per fit.
    :returns: N x C probabilities and the calibration rows.
    """
    Pm = np.full((R.N, R.C), np.nan, np.float32)
    folds = sorted(oof)
    cal_rows = []
    for f in folds:
        xs, ys = [], []
        for g in folds:
            if g == f:
                continue
            rows, Cm = oof[g]['rows'], oof[g]['C']
            rr = R.real[rows]
            xs.append(Cm[rr].ravel())
            ys.append(R.Y[rows[rr]].ravel())
        x, y = np.concatenate(xs), np.concatenate(ys)
        k = rng.choice(len(x), min(len(x), max_fit), replace=False)
        iso = _isotonic(x[k], y[k])
        rows, Cm = oof[f]['rows'], oof[f]['C']
        Pm[rows] = iso.predict(Cm.ravel()).reshape(Cm.shape)
        rr = R.real[rows]
        p, yy = Pm[rows[rr]].ravel(), R.Y[rows[rr]].ravel()
        cal_rows.append({'task': 'category', 'fold': f, 'ECE': ece(p, yy), 'Brier': float(((p - yy) ** 2).mean()),
                         'mean_p': float(p.mean()), 'base': float(yy.mean())})
    return Pm, cal_rows


def pair_calibrated(R: Reference, oof: Mapping[int, Mapping[str, np.ndarray]], rng: np.random.Generator,
                    frac: float = 0.1) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Cross-fitted calibrated P(known pair), symmetrised as the mean of both directions.

    :param R: the labels (known = current references).
    :param oof: out-of-fold scores.
    :param rng: the subsample stream.
    :param frac: pair subsample of each fit.
    :returns: N x N probabilities and the calibration rows.
    """
    N = R.N
    Pm = np.full((N, N), np.nan, np.float32)
    cal_rows = []
    known = R.cur
    folds = sorted(oof)
    for f in folds:
        xs, ys = [], []
        for g in folds:
            if g == f:
                continue
            z = oof[g]
            rows = z['rows'][R.real[z['rows']]]
            test = np.zeros(N, bool)
            test[z['rows']] = True
            cols = np.flatnonzero(R.real & ~test)
            Sg = z['S'][R.real[z['rows']]][:, cols]
            k = rng.random(Sg.shape) < frac
            xs.append(Sg[k])
            ys.append(known[np.ix_(rows, cols)][k])
        iso = _isotonic(np.concatenate(xs), np.concatenate(ys))
        z = oof[f]
        test = np.zeros(N, bool)
        test[z['rows']] = True
        cols = np.flatnonzero(R.real & ~test)
        Pf = iso.predict(z['S'][:, cols].ravel()).reshape(len(z['rows']), len(cols)).astype(np.float32)
        Pm[np.ix_(z['rows'], cols)] = Pf
        rr = R.real[z['rows']]
        p, yy = Pf[rr].ravel(), known[np.ix_(z['rows'][rr], cols)].ravel()
        cal_rows.append({'task': 'pair', 'fold': f, 'ECE': ece(p, yy), 'Brier': float(((p - yy) ** 2).mean()),
                         'mean_p': float(p.mean()), 'base': float(yy.mean())})
    A = Pm[np.ix_(R.ridx, R.ridx)]
    B = A.T
    Pm[np.ix_(R.ridx, R.ridx)] = np.where(np.isnan(A), B, np.where(np.isnan(B), A, 0.5 * (A + B)))
    return Pm, cal_rows


def qvalues(cand: np.ndarray, decoy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target-decoy and calibration q-values of candidate probabilities.

    ``q_NT = (d(t) + 1) / n_decoy * n_cand / n(t)`` with ``d(t)`` the decoys
    at or above ``t``; ``q_cal = 1 - mean calibrated P of the candidates
    above t``; both made monotone by a running minimum from the bottom.

    :param cand: candidate probabilities.
    :param decoy: decoy probabilities on the same scale.
    :returns: ``(q_combined, q_NT, q_cal)``, aligned with ``cand``.
    """
    o = np.argsort(-cand, kind='stable')
    c = cand[o]
    n = np.arange(1, len(c) + 1)
    ds = np.sort(decoy)
    d_ge = len(ds) - np.searchsorted(ds, c, side='left')
    q_nt = (d_ge + 1) / max(len(ds), 1) * len(c) / n
    q_cal = 1 - np.cumsum(c) / n
    q = np.maximum(q_nt, q_cal)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = q
    qn = np.minimum.accumulate(q_nt[::-1])[::-1]
    o_nt = np.empty_like(qn)
    o_nt[o] = qn
    qc = np.minimum.accumulate(q_cal[::-1])[::-1]
    o_c = np.empty_like(qc)
    o_c[o] = qc
    return np.clip(out, 0, 1), np.clip(o_nt, 0, 1), np.clip(o_c, 0, 1)


def jump_support_cat(Xj: np.ndarray, Hj: np.ndarray, g: int, members: Sequence[int], rng: np.random.Generator,
                     n: int = 2000) -> Tuple[float, float, int]:
    """Independent support of a gene -> category call: mean cosine to the members against random sets.

    :param Xj: the support block.
    :param Hj: its presence.
    :param g: the gene.
    :param members: the category's members.
    :param rng: the null stream.
    :param n: null sets.
    :returns: ``(cosine, p, members used)``; ``p = (1 + #null >= obs) / (1 + n)``.
    """
    if not Hj[g]:
        return np.nan, np.nan, 0
    mem = np.array([m for m in members if Hj[m] and m != g])
    if len(mem) < 2:
        return np.nan, np.nan, len(mem)
    Xn = Xj / np.clip(np.linalg.norm(Xj, axis=1, keepdims=True), 1e-12, None)
    obs = float((Xn[mem] @ Xn[g]).mean())
    pool = np.flatnonzero(Hj)
    pool = pool[pool != g]
    sims = Xn[pool] @ Xn[g]
    null = np.array([sims[rng.choice(len(pool), len(mem), replace=False)].mean() for _ in range(n)])
    return obs, float((1 + (null >= obs).sum()) / (1 + n)), len(mem)


def jump_support_pair(Xj: np.ndarray, Hj: np.ndarray, g: int, u: int) -> Tuple[float, float]:
    """Independent support of a gene-gene call: the partner's rank among the gene's cosines.

    :param Xj: the support block.
    :param Hj: its presence.
    :param g: the gene.
    :param u: the partner.
    :returns: ``(cosine, p)``.
    """
    if not (Hj[g] and Hj[u]):
        return np.nan, np.nan
    Xn = Xj / np.clip(np.linalg.norm(Xj, axis=1, keepdims=True), 1e-12, None)
    pool = np.flatnonzero(Hj)
    pool = pool[(pool != g) & (pool != u)]
    sims = Xn[pool] @ Xn[g]
    obs = float(Xn[u] @ Xn[g])
    return obs, float((1 + (sims >= obs).sum()) / (1 + len(sims)))


def drivers(T: pd.DataFrame) -> pd.DataFrame:
    """Add the arm deltas and the driver label of each prediction.

    ``delta_OPS_over_graph = p_noJUMP - p_graph``, ``delta_JUMP = p_full -
    p_noJUMP``, ``delta_OPS_given_JUMP = p_full - p_noOPS``; the label lists
    the blocks with a delta above 0.05, plus ``graph`` when the graph-only
    probability is at least half the call's.

    :param T: predictions with ``p_<arm>`` columns.
    :returns: the same frame with the new columns.
    """
    if {'p_noJUMP', 'p_graph'} <= set(T.columns):
        T['delta_OPS_over_graph'] = T.p_noJUMP - T.p_graph
    if {'p_full', 'p_noJUMP'} <= set(T.columns):
        T['delta_JUMP'] = T.p_full - T.p_noJUMP
    if {'p_full', 'p_noOPS'} <= set(T.columns):
        T['delta_OPS_given_JUMP'] = T.p_full - T.p_noOPS
    T['drivers'] = pd.Series([_driver_label(r) for r in T.to_dict('records')], index=T.index, dtype=object)
    return T


def _driver_label(r: Mapping[str, Any]) -> str:
    """The driver label of one prediction row."""
    d = []
    if r.get('delta_OPS_over_graph', 0) > 0.05:
        d.append('OPS')
    if r.get('delta_JUMP', 0) > 0.05:
        d.append('JUMP')
    if r.get('p_graph', 0) >= 0.5 * r.get('p_calibrated', 1):
        d.append('graph')
    return '+'.join(d) or 'mixed'


def poorly_characterised(nodes: pd.DataFrame, real: np.ndarray, rule: Mapping[str, float]) -> np.ndarray:
    """Real genes in the bottom quantiles of both genome STRING degree and GO terms.

    :param nodes: the node table.
    :param real: real-gene mask.
    :param rule: ``string04_degree_q`` and ``go_terms_q``.
    :returns: the mask.
    """
    sdeg, go = nodes.string04_degree_genome.to_numpy(), nodes.go_terms.to_numpy()
    return (real & (sdeg <= np.quantile(sdeg[real], float(rule['string04_degree_q'])))
            & (go <= np.quantile(go[real], float(rule['go_terms_q']))))


def specificity_flag(table: Optional[pd.DataFrame], gene: Any) -> str:
    """The guide-specificity flag of a gene: multi_copy, near_paralog, adjacent_stain_target, none.

    :param table: the flags, indexed by gene; None gives ``not_in_table``.
    :param gene: the gene.
    :returns: the flags joined with ``+``, ``none`` or ``not_in_table``.
    """
    g = gene_symbol(gene)
    if table is None or g not in table.index:
        return 'not_in_table'
    r = table.loc[g]
    f = [k for k in ('flag_multi_copy', 'flag_near_paralog', 'flag_adjacent_stain_target') if bool(r[k])]
    return '+'.join(x.replace('flag_', '') for x in f) or 'none'


def _jump_blocks(D: GeneGraphData, o: Mapping[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The independent-support blocks present in the graph."""
    return {lab: (D.z[f'X_{b}'], D.z[f'H_{b}']) for lab, b in (o['jump_blocks'] or {}).items() if f'X_{b}' in D.z}


def predict_heldout(R: Reference, runs_dir: Any, o: Mapping[str, Any], tables: Path) -> Dict[str, Any]:
    """Calibrated, FDR-controlled held-out predictions with their support and drivers.

    :param R: the labels.
    :param runs_dir: the runs folder.
    :param o: unprefixed options.
    :param tables: the tables folder.
    :returns: the frames written.
    """
    D = R.D
    model, main_arm = o['prediction_model'], o['prediction_arm']
    rule, fdr, n_folds = o['fdr_rule'], float(o['fdr']), int(o['n_folds'])
    rng = np.random.default_rng(int(o['seed']))
    nodes, classes = R.nodes, D.classes
    J = _jump_blocks(D, o)
    sdeg, go = nodes.string04_degree_genome.to_numpy(), nodes.go_terms.to_numpy()
    poor = poorly_characterised(nodes, R.real, o['poorly_characterised'])
    arms = [main_arm] + [x for x in o['driver_arms'] if x != main_arm]
    calP, calC, cal_rows = {}, {}, []
    for arm in arms:
        oof = load_oof(runs_dir, model, arm, n_folds)
        if oof is None:
            continue
        Pc, rc = cat_calibrated(R, oof, rng)
        Pp, rp = pair_calibrated(R, oof, rng)
        calC[arm], calP[arm] = Pc, Pp
        cal_rows += [{**r, 'model': model, 'arm': arm} for r in rc + rp]
    out: Dict[str, Any] = {'calibration': pd.DataFrame(cal_rows)}
    _write(out['calibration'], tables / 'calibration.csv')
    if main_arm not in calC:
        return out
    Pc, Pp = calC[main_arm], calP[main_arm]
    cand_mask = R.real[:, None] & ~R.Y
    cand = Pc[cand_mask]
    decoy = Pc[~R.real].ravel()
    q_comb, q_nt, q_c = qvalues(cand, decoy)
    q = q_nt if rule == 'nt' else q_comb
    Q = np.full(Pc.shape, np.nan, np.float32)
    Q[cand_mask] = q
    QN = np.full(Pc.shape, np.nan, np.float32)
    QN[cand_mask] = q_c
    QB = np.full(Pc.shape, np.nan, np.float32)
    QB[cand_mask] = q_comb
    thr = cand[q < fdr].min() if (q < fdr).any() else np.inf
    fdr_rows = [{'task': 'category', 'arm': main_arm, 'candidates': int(cand.size), 'decoys': int(decoy.size),
                 'rule': rule, 'pass_q0.1': int((q < fdr).sum()), 'pass_q0.1_poor': int((Q[poor] < fdr).sum()),
                 'pass_qNT0.1': int((q_nt < fdr).sum()), 'pass_combined0.1': int((q_comb < fdr).sum()),
                 'threshold_p': float(thr), 'NT_rows_above_threshold': int((decoy >= thr).sum()),
                 'NT_genes_with_any': int((Pc[~R.real] >= thr).any(1).sum())}]
    rows = []
    gi, ci = np.nonzero(poor[:, None] & (Q < fdr))
    for g, c in zip(gi, ci):
        mem = np.flatnonzero(R.Y[:, c])
        rec = {'gene': nodes.gene[g], 'class_id': classes.class_id[c], 'class': classes.name[c],
               'source': classes.source[c], 'class_size': len(mem), 'p_calibrated': float(Pc[g, c]),
               'q_NT': float(Q[g, c]) if rule == 'nt' else np.nan, 'q_cal': float(QN[g, c]),
               'q_combined': float(QB[g, c]), 'q': float(Q[g, c]), 'string04_degree': int(sdeg[g]),
               'go_terms': int(go[g]), 'fold': int(R.fold[g])}
        for arm in arms:
            rec[f'p_{arm}'] = float(calC[arm][g, c]) if arm in calC else np.nan
        for m, (Xj, Hj) in J.items():
            rec[f'JUMP_{m}_cos'], rec[f'JUMP_{m}_p'], rec[f'JUMP_{m}_members'] = jump_support_cat(
                Xj, Hj, g, mem, rng, int(o['jump_null']))
        rows.append(rec)
    T = pd.DataFrame(rows, columns=[
        'gene', 'class_id', 'class', 'source', 'class_size', 'p_calibrated', 'q_NT', 'q_cal', 'q_combined', 'q',
        'string04_degree', 'go_terms', 'fold'] + [f'p_{a}' for a in arms]
        + [f'JUMP_{m}_{x}' for m in J for x in ('cos', 'p', 'members')])
    out['category'] = drivers(T).sort_values(['q', 'p_calibrated'], ascending=[True, False], kind='stable')
    s04 = np.zeros((R.N, R.N), bool)
    for name in o['known_extra'] or []:
        s04 |= R.matrix(f'E_cur_{known_key(name)}')
    iu = np.triu_indices(len(R.ridx), 1)
    i, j = R.ridx[iu[0]], R.ridx[iu[1]]
    pv = Pp[i, j]
    ok = ~np.isnan(pv)
    known = R.cur[i, j] | s04[i, j] | R.train[i, j]
    ci_ = ok & ~known
    cand = pv[ci_]
    dec = Pp[np.ix_(np.flatnonzero(~R.real), R.ridx)].ravel()
    dec = dec[~np.isnan(dec)]
    q_comb, q_nt, q_c = qvalues(cand, dec)
    q = q_nt if rule == 'nt' else q_comb
    thr = cand[q < fdr].min() if (q < fdr).any() else np.inf
    pi_, pj_ = i[ci_], j[ci_]
    sel = (q < fdr) & (poor[pi_] | poor[pj_])
    fdr_rows.append({'task': 'pair', 'arm': main_arm, 'candidates': int(cand.size), 'decoys': int(dec.size),
                     'rule': rule, 'pass_q0.1': int((q < fdr).sum()), 'pass_qNT0.1': int((q_nt < fdr).sum()),
                     'pass_combined0.1': int((q_comb < fdr).sum()), 'pass_q0.1_poor': int(sel.sum()),
                     'threshold_p': float(thr), 'NT_rows_above_threshold': int((dec >= thr).sum()),
                     'NT_genes_with_any': int((Pp[np.ix_(np.flatnonzero(~R.real), R.ridx)] >= thr).any(1).sum())})
    rows = []
    order = np.argsort(-cand[sel])[:5000]
    for k in np.flatnonzero(sel)[order]:
        g, u = pi_[k], pj_[k]
        if not poor[g]:
            g, u = u, g
        rec = {'gene': nodes.gene[g], 'partner': nodes.gene[u], 'p_calibrated': float(cand[k]), 'q': float(q[k]),
               'q_cal': float(q_c[k]), 'q_combined': float(q_comb[k]), 'gene_string04_degree': int(sdeg[g]),
               'gene_go_terms': int(go[g]), 'partner_poorly_characterised': bool(poor[u]),
               'partner_2022_degree': int(R.deg[u]), 'gene_fold': int(R.fold[g]), 'partner_fold': int(R.fold[u])}
        for arm in arms:
            rec[f'p_{arm}'] = float(calP[arm][g, u]) if arm in calP else np.nan
        for m, (Xj, Hj) in J.items():
            rec[f'JUMP_{m}_cos'], rec[f'JUMP_{m}_p'] = jump_support_pair(Xj, Hj, g, u)
        rows.append(rec)
    out['pair'] = drivers(pd.DataFrame(rows, columns=[
        'gene', 'partner', 'p_calibrated', 'q', 'q_cal', 'q_combined', 'gene_string04_degree', 'gene_go_terms',
        'partner_poorly_characterised', 'partner_2022_degree', 'gene_fold', 'partner_fold']
        + [f'p_{a}' for a in arms] + [f'JUMP_{m}_{x}' for m in J for x in ('cos', 'p')]))
    out['fdr'] = pd.DataFrame(fdr_rows)
    out['poor'] = {'poorly_characterised_genes': int(poor.sum()),
                   'rule': 'STRING v12>=0.4 genome degree <= Q1 and GO terms <= Q1 (real genes)',
                   'q1_string04': float(np.quantile(sdeg[R.real], float(o['poorly_characterised']['string04_degree_q']))),
                   'q1_go': float(np.quantile(go[R.real], float(o['poorly_characterised']['go_terms_q'])))}
    _write(out['category'], tables / f'predictions_gene_category_{model}_{main_arm}.csv')
    _write(out['pair'], tables / f'predictions_gene_gene_{model}_{main_arm}.csv')
    _write(out['fdr'], tables / f'fdr_summary_{model}_{main_arm}.csv')
    _write_json(out['poor'], tables / 'poorly_characterised_rule.json')
    return out


def time_calibrated(R: Reference, S: np.ndarray, rng: np.random.Generator,
                    frac: float = 0.3) -> Tuple[np.ndarray, Dict[bool, Any]]:
    """Calibrated P(edge added later) of the time-split candidates, cross-fitted on two gene halves.

    :param R: the labels.
    :param S: N x N scores.
    :param rng: the halves and subsample stream.
    :param frac: subsample of each half's pairs.
    :returns: the probabilities per candidate and the two isotonic maps.
    """
    half = rng.random(R.N) < 0.5
    s = S[R.ti, R.tj]
    y = R.tlab['any']
    hi, hj = half[R.ti], half[R.tj]
    isos = {}
    for h in (True, False):
        k = np.flatnonzero((hi == h) & (hj == h))
        k = k[rng.random(len(k)) < frac]
        isos[h] = _isotonic(s[k], y[k])
    inA, inB = hi & hj, ~hi & ~hj
    pa, pb = isos[True].predict(s), isos[False].predict(s)
    return np.where(inA, pb, np.where(inB, pa, 0.5 * (pa + pb))).astype(np.float32), isos


def predict_timesplit(R: Reference, runs_dir: Any, o: Mapping[str, Any], tables: Path) -> Dict[str, Any]:
    """Gene-gene predictions of the time-split model for poorly characterised genes.

    Candidates are pairs in no reference now; ``q_NT`` comes from the
    non-targeting pseudo-gene pairs scored by the same model.

    :param R: the labels.
    :param runs_dir: the runs folder.
    :param o: unprefixed options.
    :param tables: the tables folder.
    :returns: the predictions and the FDR summary, or an empty dict without the time run.
    """
    D = R.D
    model, main_arm, fdr = o['prediction_model'], o['prediction_arm'], float(o['fdr'])
    nodes = R.nodes
    sdeg, go = nodes.string04_degree_genome.to_numpy(), nodes.go_terms.to_numpy()
    poor = poorly_characterised(nodes, R.real, o['poorly_characterised'])
    s04 = np.zeros((R.N, R.N), bool)
    for name in o['known_extra'] or []:
        s04 |= R.matrix(f'E_cur_{known_key(name)}')
    known = R.cur[R.ti, R.tj] | s04[R.ti, R.tj]
    cand = (poor[R.ti] | poor[R.tj]) & ~known
    arms = [main_arm] + [x for x in o['driver_arms'] if x != main_arm]
    Pcal = {}
    dec = raw_s = None
    for arm in arms:
        d = Path(runs_dir) / f'time__{model}__{arm}__none__s0'
        if not (d / 'done.json').exists():
            continue
        S = np.load(d / 'scores.npz')['S'].astype(np.float32)
        p, isos = time_calibrated(R, S, np.random.default_rng(int(o['seed'])))
        Pcal[arm] = p
        if arm == main_arm:
            nts = S[np.ix_(R.nt_idx, R.ridx)].ravel()
            dec = 0.5 * (isos[True].predict(nts) + isos[False].predict(nts)) if len(nts) else np.zeros(0)
            raw_s = S[R.ti, R.tj]
    if main_arm not in Pcal:
        return {}
    pc = Pcal[main_arm][cand]
    o_ = np.argsort(-pc, kind='stable')
    c = pc[o_]
    ds = np.sort(dec)
    d_ge = len(ds) - np.searchsorted(ds, c, side='left')
    n = np.arange(1, len(c) + 1)
    q = np.minimum.accumulate(((d_ge + 1) / max(len(ds), 1) * len(c) / n)[::-1])[::-1]
    q_cal = np.minimum.accumulate((1 - np.cumsum(c) / n)[::-1])[::-1]
    idx = np.flatnonzero(cand)[o_]
    keep = q < fdr
    rows = []
    J = _jump_blocks(D, o)
    for k in np.flatnonzero(keep)[:3000]:
        i, j = R.ti[idx[k]], R.tj[idx[k]]
        g, u = (i, j) if poor[i] else (j, i)
        rec = {'gene': nodes.gene[g], 'partner': nodes.gene[u], 'p_calibrated': float(c[k]), 'q_NT': float(q[k]),
               'q_cal': float(q_cal[k]), 'score': float(raw_s[idx[k]]), 'gene_string04_degree': int(sdeg[g]),
               'gene_go_terms': int(go[g]), 'gene_2022_degree': int(R.deg[g]),
               'partner_poorly_characterised': bool(poor[u]), 'partner_2022_degree': int(R.deg[u])}
        for arm in arms:
            rec[f'p_{arm}'] = float(Pcal[arm][idx[k]]) if arm in Pcal else np.nan
        for m, (Xj, Hj) in J.items():
            rec[f'JUMP_{m}_cos'], rec[f'JUMP_{m}_p'] = jump_support_pair(Xj, Hj, g, u)
        rows.append(rec)
    arm_cols = [f'p_{a}' for a in arms] + [f'JUMP_{m}_{x}' for m in J for x in ('cos', 'p')]
    T = drivers(pd.DataFrame(rows, columns=[
        'gene', 'partner', 'p_calibrated', 'q_NT', 'q_cal', 'score', 'gene_string04_degree', 'gene_go_terms',
        'gene_2022_degree', 'partner_poorly_characterised', 'partner_2022_degree'] + arm_cols))
    summ = {'candidates': int(cand.sum()), 'decoys': int(len(ds)), 'pass_qNT0.1': int(keep.sum()),
            'pass_combined0.1': int((np.maximum(q, q_cal) < fdr).sum()),
            'threshold_p': float(c[keep].min()) if keep.any() else None,
            'NT_decoys_above_threshold': int((ds >= c[keep].min()).sum()) if keep.any() else None,
            'genes': int(T.gene.nunique()) if len(T) else 0}
    _write(T, tables / f'predictions_gene_gene_timesplit_{model}_{main_arm}.csv')
    _write(pd.DataFrame([summ]), tables / f'fdr_summary_timesplit_{model}_{main_arm}.csv')
    return {'pair': T, 'fdr': summ}


def flag_predictions(tables: Path, specificity_table: Any) -> List[str]:
    """Add ``spec_flag_gene`` / ``spec_flag_partner`` to every ``predictions_*.csv``.

    :param tables: the tables folder.
    :param specificity_table: the flags table, or blank (every flag is ``not_in_table``).
    :returns: the tables rewritten.
    """
    gs = _read(specificity_table).set_index('gene') if specificity_table else None
    done = []
    for f in sorted(tables.glob('predictions_*.csv')):
        try:
            t = _read(f, sep=',')
        except pd.errors.EmptyDataError:
            continue
        t['spec_flag_gene'] = pd.Series([specificity_flag(gs, g) for g in t.gene], index=t.index, dtype=object)
        if 'partner' in t:
            t['spec_flag_partner'] = pd.Series([specificity_flag(gs, g) for g in t.partner], index=t.index, dtype=object)
        _write(t, f)
        done.append(str(f))
    return done


def predict_gene_functions(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Write the calibrated, FDR-controlled prediction lists with support, drivers and flags.

    :param settings: the ``gene_graph_*`` settings.
    :returns: ``heldout`` and ``timesplit`` results.
    """
    o = options(settings)
    p = paths(o)
    p['tables'].mkdir(parents=True, exist_ok=True)
    R = Reference(GeneGraphData(p['data']))
    res = {'heldout': predict_heldout(R, p['runs'], o, p['tables']),
           'timesplit': predict_timesplit(R, p['runs'], o, p['tables'])}
    flag_predictions(p['tables'], o.get('specificity_table'))
    return res


def barplot(df: pd.DataFrame, val: str, pts: Mapping[str, Sequence[float]], title: str, ylab: str,
            target: Path, base: Optional[float] = None) -> List[str]:
    """Bars (mean) with jittered points, both at 60 % opacity; controls hatched.

    :param df: one row per bar: ``label``, ``val``, ``color``, ``hatch``.
    :param val: the bar column.
    :param pts: label -> point values.
    :param title: the title.
    :param ylab: the y label.
    :param target: path without suffix; png and svg are written.
    :param base: a base-rate line.
    :returns: the files written.
    """
    import matplotlib

    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .plot import _checked_savefig

    with plt.rc_context({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'spacr', 'svg.fonttype': 'none'}):
        fig = Figure(figsize=(max(4, 0.32 * len(df) + 1.5), 3.4))
        ax = fig.subplots()
        rng = np.random.default_rng(0)
        for i, (_, r) in enumerate(df.iterrows()):
            ax.bar(i, r[val], color=r.color, alpha=0.6, hatch=r.hatch,
                   edgecolor='#1f3a5f' if r.hatch else r.color, linewidth=0.6)
            v = np.asarray(pts.get(r['label'], []), float)
            if len(v):
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=8, color='#0b1f33', alpha=0.6,
                           linewidths=0, zorder=3)
        if base is not None and np.isfinite(base):
            ax.axhline(base, color='#6c7a89', lw=0.8, ls='--')
            ax.text(-0.45, base, 'base rate', va='bottom', ha='left', fontsize=6, color='#6c7a89')
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df.label, rotation=60, ha='right', fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        _register_figure_data(fig, lambda: pd.DataFrame(
            [{'label': k, 'value': float(x)} for k in df.label for x in pts.get(k, [])]),
            x='label', y='value', kind='bar', order=[str(k) for k in df.label], title=title)
        fig.tight_layout()
        target.parent.mkdir(parents=True, exist_ok=True)
        files = [str(target.with_suffix('.png')), str(target.with_suffix('.svg'))]
        _checked_savefig(fig, files[0], dpi=200)
        _checked_savefig(fig, files[1], metadata={'Date': None})
    return files


def bar_style(r: Any) -> Tuple[str, str]:
    """Colour and hatch of a bar: controls grey and hatched, priors dotted.

    :param r: a row with ``model``, ``arm`` and ``control``.
    :returns: ``(colour, hatch)``.
    """
    if r.control != 'none':
        return PAL['control'], '///'
    if r.model in ('prior', 'adamicadar'):
        return PAL['prior'], '...'
    return PAL.get(r.arm, '#3a7ca5'), ''


def _bars(F_: pd.DataFrame, val: str, group: Sequence[str]) -> pd.DataFrame:
    """Bar rows: the mean of ``val`` per group, seed 0, in figure order, styled."""
    g = F_.groupby(list(group))[val].mean().reset_index()
    g = sort_rows(g[g.seed == 0])
    styles = [bar_style(r) for r in g.itertuples()]
    g['color'] = [s[0] for s in styles]
    g['hatch'] = [s[1] for s in styles]
    return g


def gene_graph_figures(settings: Mapping[str, Any]) -> List[str]:
    """Draw the held-out, time-split and calibration figures with their source tables.

    :param settings: the ``gene_graph_*`` settings.
    :returns: the files written.
    """
    o = options(settings)
    p = paths(o)
    t, figs = p['tables'], p['figures']
    files: List[str] = []
    fpath = t / 'heldout_metrics_per_fold.csv'
    if fpath.exists() and fpath.stat().st_size > 1:
        F_ = _read(fpath, sep=',')
        F_['seed'] = seed0(F_.seed)
        F_['label'] = F_.apply(run_label, axis=1)
        for val, title, yl, fn, basecol in (
                ('cat_micro_AUPRC', 'Held-out genes: gene -> category', 'micro AUPRC (folds)', 'fig1_heldout_category', 'cat_base'),
                ('cat_macro_AUPRC', 'Held-out genes: gene -> category, per class', 'macro AUPRC (folds)', 'fig1b_heldout_category_macro', None),
                ('link_AUPRC_2022', 'Held-out genes: hidden gene-gene edges', 'AUPRC (folds)', 'fig2_heldout_link', 'link_base_2022'),
                ('link_AUPRC_current', 'Held-out genes: current reference edges', 'AUPRC (folds)', 'fig2b_heldout_link_current', 'link_base_current')):
            if val not in F_:
                continue
            g = _bars(F_, val, ['label', 'model', 'arm', 'control', 'seed'])
            pts = {k: d[val].tolist() for k, d in F_.groupby('label')}
            _write(g, t / f'{fn}.csv')
            _write(F_[['label', 'task', val]], t / f'{fn}_points.csv')
            files += barplot(g, val, pts, title, yl, figs / fn, F_[basecol].mean() if basecol else None)
    tpath = t / 'time_split_metrics.csv'
    if tpath.exists() and tpath.stat().st_size > 1:
        T = _read(tpath, sep=',')
        T['seed'] = seed0(T.seed)
        T['label'] = T.apply(run_label, axis=1)
        Bt = _read(t / 'time_split_bootstrap.csv', sep=',')
        Bt['label'] = Bt.name.map(dict(zip(T.name, T.label)))
        for val, title, yl, fn, with_pts in (
                ('AUPRC_any', 'Time split: edges added after 2022', 'AUPRC (gene bootstraps)', 'fig3_time_split_auprc', True),
                ('P@1000', 'Time split: top-1,000 confirmed-later rate', 'precision@1,000', 'fig3b_time_split_p1000', False),
                ('enrich@1000_vs_degmatched', 'Time split: top 1,000 vs degree-matched random pairs',
                 'confirmed-later rate / degree-matched', 'fig3c_time_split_vs_degmatched', False)):
            if val not in T:
                continue
            g = _bars(T, val, ['label', 'model', 'arm', 'control', 'seed'])
            pts = {k: d.AUPRC_any.tolist() for k, d in Bt.groupby('label')} if with_pts else {}
            _write(g, t / f'{fn}.csv')
            base = T.base_any.mean() if val == 'AUPRC_any' else (1.0 if 'enrich' in val else None)
            files += barplot(g, val, pts, title, yl, figs / fn, base)
    cpath = t / 'calibration.csv'
    if cpath.exists() and cpath.stat().st_size > 1:
        C = _read(cpath, sep=',')
        C['label'] = C.task + ', ' + C.arm.map(lambda a: ARM_LABEL.get(a, a))
        g = C.groupby(['label', 'arm']).ECE.mean().reset_index()
        g['model'], g['control'], g['seed'] = o['prediction_model'], 'none', 0
        g = _bars(g, 'ECE', ['label', 'model', 'arm', 'control', 'seed'])
        _write(g, t / 'fig5_calibration_ece.csv')
        files += barplot(g, 'ECE', {k: d.ECE.tolist() for k, d in C.groupby('label')},
                         'Calibration of out-of-fold probabilities', 'expected calibration error (folds)',
                         figs / 'fig5_calibration_ece')
    return files


def save_settings_file(settings: Mapping[str, Any], root: Path, name: str) -> List[str]:
    """Write the settings a run used, as the CSV (and JSON) ``spacr-run --settings`` reads back.

    :param settings: the full settings.
    :param root: the module's output folder.
    :param name: the file stem.
    :returns: the files written.
    """
    from .cli import write_settings_file

    return write_settings_file(dict(settings), root / 'settings' / f'{name}.csv')


def run_gene_graph(settings: Mapping[str, Any], over_limit: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Build, train, evaluate and predict in order; what ``spacr-run gene_graph`` calls.

    The settings are written to ``<src>/gene_graph/settings/gene_graph_settings.csv``
    first, so the run can be repeated from that file. A run that yielded the
    card or reached the GPU budget stops after training; run it again to
    resume.

    :param settings: the ``gene_graph_*`` settings.
    :param over_limit: the yield test (see :func:`run_config`).
    :returns: ``summary``, ``train`` and, when training finished, ``tables``
        and ``predictions``.
    """
    full = get_gene_graph_default_settings(settings)
    o = options(full)
    p = paths(o)
    p['root'].mkdir(parents=True, exist_ok=True)
    save_settings_file(full, p['root'], 'gene_graph_settings')
    summary = build_gene_graph(full) if not o.get('data_dir') else json.loads(
        (p['data'] / 'data_summary.json').read_text(encoding='utf-8'))
    result: Dict[str, Any] = {'summary': summary, 'train': train_gene_graph(full, over_limit=over_limit)}
    if result['train']['status'] != 'done':
        return result
    result['tables'] = evaluate_gene_graph(full)
    result['predictions'] = predict_gene_functions(full)
    if o['figures']:
        result['figures'] = gene_graph_figures(full)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run :func:`run_gene_graph` from a settings file.

    ``python -m spacr.gene_graph --settings gene_graph_settings.csv`` runs
    every step; the exit status is 75 when the run yielded the card.

    :param argv: the arguments; None reads :data:`sys.argv`.
    :returns: 0 when finished, 75 when yielded or stopped at the budget.
    """
    from .cli import load_settings_file

    parser = argparse.ArgumentParser(prog='python -m spacr.gene_graph', description=_DESCRIPTION)
    parser.add_argument('--settings', required=True)
    args = parser.parse_args(argv)
    result = run_gene_graph(load_settings_file(args.settings))
    return 0 if result['train']['status'] == 'done' else YIELD_RC


if __name__ == '__main__':
    sys.exit(main())
