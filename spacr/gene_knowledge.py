"""Protein-knowledge feature blocks for gene-level analysis, built so they cannot leak the answer.

What a screen adds to gene-function prediction can only be stated against
what prior protein knowledge alone predicts. This module builds that
knowledge on the gene graph's node order (:mod:`spacr.gene_graph`): one row
per screen gene and one per non-targeting pseudo-gene.

* :func:`fetch_protein_annotations`: UniProt sequences, Pfam / InterPro
  cross-references and function text, current (REST, batches of at most 100
  accessions) and from an archived Swiss-Prot release (``.dat`` streamed out
  of the release tarball); plus a seeded draw of reviewed human proteins
  outside the library, the decoys' knowledge;
* :func:`embed_sequences`: ESM-2 mean residue embeddings, long sequences in
  overlapping windows;
* :func:`embed_text`: function sentences, each with the gene symbols and
  accessions it names, embedded by a local sentence model (mean pooling,
  L2-normalised);
* :func:`predict_3di`: the ProstT5 encoder and its CNN head predict a 3Di
  structure string per sequence, and the 3Di string is embedded;
* :func:`foldseek_similarity`: an all-vs-all Foldseek search gives the
  structural-similarity edges and the paralog pairs;
* :func:`build_knowledge_blocks`: the per-task blocks (text masked per fold,
  ``_time`` blocks from the archive only), decoy blocks and structure blocks;
* :func:`paralog_mask` and :func:`profile_metrics`.

Leakage rules enforced here: in fold ``k`` the fold's genes and decoys get no
text vector and every sentence of any other gene naming a fold-``k`` gene is
dropped before averaging, with the text PCA refit per fold; ``_time`` blocks
read only the archive; sequence, domain and structure blocks are kept for
held-out genes, domain names are never embedded and GO is never an input;
decoys carry an out-of-library protein's knowledge; Foldseek edges are
message-passing inputs only.

No hosted model or API is used. Models run from local Hugging Face weights
(``gene_knowledge_models_dir``); they are downloaded only when
``gene_knowledge_allow_download`` is on. Every step can instead reuse the
stored output of an earlier run (``*_embeddings``, ``*_outputs``,
``foldseek_alignments``), which is how a run is repeated without a GPU.
"""
from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import gene_graph as gg

logger = logging.getLogger(__name__)

APP_KEY = 'gene_knowledge'
PREFIX = 'gene_knowledge_'

#: The UniProt REST fields fetched.
UNIPROT_FIELDS = 'accession,gene_primary,gene_synonym,reviewed,length,xref_pfam,xref_interpro,cc_function,sequence'

#: The 20 3Di states, in the ProstT5 CNN head's output order.
SS = 'ACDEFGHIKLMNPQRSTVWY'

_DEFAULTS: Dict[str, Any] = {
    'gene_knowledge_src': '',
    'gene_knowledge_nodes': '',
    'gene_knowledge_uniprot_function_cache': '',
    'gene_knowledge_uniprot_current_table': '',
    'gene_knowledge_uniprot_archive': '',
    'gene_knowledge_uniprot_archive_table': '',
    'gene_knowledge_uniprot_archive_release': '2022_01',
    'gene_knowledge_allow_network': False,
    'gene_knowledge_decoy_proteins': 63,
    'gene_knowledge_decoy_draw': 80,
    'gene_knowledge_models_dir': '',
    'gene_knowledge_allow_download': False,
    'gene_knowledge_esm_model': 'facebook/esm2_t33_650M_UR50D',
    'gene_knowledge_esm_window': 1022,
    'gene_knowledge_esm_stride': 511,
    'gene_knowledge_esm_tokens_per_batch': 6000,
    'gene_knowledge_esm_embeddings': '',
    'gene_knowledge_esm_pca': 128,
    'gene_knowledge_text_model': 'NeuML/pubmedbert-base-embeddings',
    'gene_knowledge_text_max_tokens': 256,
    'gene_knowledge_text_embeddings': '',
    'gene_knowledge_text_pca': 64,
    'gene_knowledge_text_mask_mentions': True,
    'gene_knowledge_domain_min_genes': 3,
    'gene_knowledge_domain_svd': 64,
    'gene_knowledge_prostt5_model': 'Rostlab/ProstT5_fp16',
    'gene_knowledge_prostt5_cnn': '',
    'gene_knowledge_prostt5_window': 1000,
    'gene_knowledge_prostt5_stride': 750,
    'gene_knowledge_prostt5_residues_per_batch': 2000,
    'gene_knowledge_prostt5_outputs': '',
    'gene_knowledge_s3di_pca': 64,
    'gene_knowledge_kmer': 3,
    'gene_knowledge_kmer_svd': 64,
    'gene_knowledge_struct_spec_dims': 64,
    'gene_knowledge_structure_source': 'prostt5',
    'gene_knowledge_afdb_dir': '',
    'gene_knowledge_foldseek_bin': 'foldseek',
    'gene_knowledge_foldseek_alignments': '',
    'gene_knowledge_foldseek_evalue': 1e-3,
    'gene_knowledge_foldseek_exhaustive': True,
    'gene_knowledge_foldseek_threads': 3,
    'gene_knowledge_paralog_fident': 0.30,
    'gene_knowledge_paralog_shared_pfam': True,
    'gene_knowledge_image_blocks': {},
    'gene_knowledge_image_pca': 64,
    'gene_knowledge_image_nt_prefix': 'NTpseudo',
    'gene_knowledge_n_folds': 5,
    'gene_knowledge_profile_reference_dir': '',
    'gene_knowledge_profile_references': ['CORUM', 'STRING>=0.7', 'BioGRID'],
    'gene_knowledge_profile_sets': {},
    'gene_knowledge_device': 'cuda',
    'gene_knowledge_gpu_cap_gb': 4.5,
    'gene_knowledge_gpu_limit_gb': 19.0,
    'gene_knowledge_admission_hold_s': 180,
    'gene_knowledge_gpu_log': '~/.spacr/gpu/log',
    'gene_knowledge_seed': 20261109,
}

_EXPECTED_TYPES: Dict[str, Any] = {
    key: ((int, float) if isinstance(value, float) else type(value)) for key, value in _DEFAULTS.items()
}
_EXPECTED_TYPES['gene_knowledge_profile_reference_dir'] = (str, list)

_TOOLTIPS: Dict[str, str] = {
    'gene_knowledge_src': "(str) - Output folder; gene_knowledge/ is created under it. Default blank.",
    'gene_knowledge_nodes': "(str) - The gene graph's nodes.parquet: the node order (genes, then non-targeting pseudo-genes). Default blank.",
    'gene_knowledge_uniprot_function_cache': "(str) - Table gene -> UniProt accession and function text (TSV). Default blank.",
    'gene_knowledge_uniprot_current_table': "(str) - Stored current UniProt rows (gene, accession, source, length, pfam, interpro, sequence, function_current) to use instead of fetching. Default blank.",
    'gene_knowledge_uniprot_archive': "(str) - Archived Swiss-Prot release: uniprot_sprot.dat(.gz) or the release tarball. Default blank.",
    'gene_knowledge_uniprot_archive_table': "(str) - Stored rows parsed from the archived release to use instead of parsing. Default blank.",
    'gene_knowledge_uniprot_archive_release': "(str) - Name of the archived release (recorded only). Default '2022_01'.",
    'gene_knowledge_allow_network': "(bool) - Allow the UniProt REST queries. Default False.",
    'gene_knowledge_decoy_proteins': "(int) - Out-of-library proteins used as the decoys' knowledge. Default 63.",
    'gene_knowledge_decoy_draw': "(int) - Proteins drawn (seeded) before the first ones with a sequence are kept. Default 80.",
    'gene_knowledge_models_dir': "(str) - Local Hugging Face cache folder of the models. Default blank.",
    'gene_knowledge_allow_download': "(bool) - Download model weights that are not in the local cache. Default False.",
    'gene_knowledge_esm_model': "(str) - ESM-2 model. Default 'facebook/esm2_t33_650M_UR50D'.",
    'gene_knowledge_esm_window': "(int) - ESM-2 window (residues). Default 1022.",
    'gene_knowledge_esm_stride': "(int) - ESM-2 window stride. Default 511.",
    'gene_knowledge_esm_tokens_per_batch': "(int) - ESM-2 tokens per batch. Default 6000.",
    'gene_knowledge_esm_embeddings': "(str) - Stored esm_by_md5.npz to reuse instead of running ESM-2. Default blank.",
    'gene_knowledge_esm_pca': "(int) - PCA dimensions of the ESM block (fit on library genes, current sequences). Default 128.",
    'gene_knowledge_text_model': "(str) - Local sentence model. Default 'NeuML/pubmedbert-base-embeddings'.",
    'gene_knowledge_text_max_tokens': "(int) - Tokens per sentence. Default 256.",
    'gene_knowledge_text_embeddings': "(str) - Folder with a stored sentences.parquet and sent_emb.npy to reuse. Default blank.",
    'gene_knowledge_text_pca': "(int) - PCA dimensions of the text block, refit per fold. Default 64.",
    'gene_knowledge_text_mask_mentions': "(bool) - Drop every sentence naming a held-out gene (symbol or accession). Default True.",
    'gene_knowledge_domain_min_genes': "(int) - InterPro / Pfam entries kept when in at least this many library genes. Default 3.",
    'gene_knowledge_domain_svd': "(int) - TF-IDF then truncated SVD dimensions of the domain block. Default 64.",
    'gene_knowledge_prostt5_model': "(str) - ProstT5 model. Default 'Rostlab/ProstT5_fp16'.",
    'gene_knowledge_prostt5_cnn': "(str) - ProstT5 CNN head checkpoint (cnn_chkpnt/model.pt). Default blank.",
    'gene_knowledge_prostt5_window': "(int) - ProstT5 window. Default 1000.",
    'gene_knowledge_prostt5_stride': "(int) - ProstT5 window stride. Default 750.",
    'gene_knowledge_prostt5_residues_per_batch': "(int) - ProstT5 residues per batch. Default 2000.",
    'gene_knowledge_prostt5_outputs': "(str) - Folder with a stored 3di.fasta, aa.fasta and emb3di_by_md5.npz to reuse. Default blank.",
    'gene_knowledge_s3di_pca': "(int) - PCA dimensions of the 3Di embedding block. Default 64.",
    'gene_knowledge_kmer': "(int) - 3Di k-mer length of the k-mer TF-IDF block. Default 3.",
    'gene_knowledge_kmer_svd': "(int) - SVD dimensions of the k-mer block. Default 64.",
    'gene_knowledge_struct_spec_dims': "(int) - Laplacian-eigenmap dimensions of the structural graph. Default 64.",
    'gene_knowledge_structure_source': "(str) - 'prostt5' (predicted 3Di) or 'afdb' (AlphaFold DB files, pLDDT-masked). Default 'prostt5'.",
    'gene_knowledge_afdb_dir': "(str) - Folder of AFDB v4 PDB/CIF files (structure_source='afdb'). Default blank.",
    'gene_knowledge_foldseek_bin': "(str) - The foldseek executable. Default 'foldseek'.",
    'gene_knowledge_foldseek_alignments': "(str) - A stored all-vs-all alignment table (query, target, fident, evalue, bits) to reuse. Default blank.",
    'gene_knowledge_foldseek_evalue': "(float) - E-value cut of the search. Default 0.001.",
    'gene_knowledge_foldseek_exhaustive': "(bool) - Exhaustive search. Default True.",
    'gene_knowledge_foldseek_threads': "(int) - Foldseek threads. Default 3.",
    'gene_knowledge_paralog_fident': "(float) - Sequence identity above which a hit is a paralog pair. Default 0.3.",
    'gene_knowledge_paralog_shared_pfam': "(bool) - A shared Pfam family also makes a paralog pair. Default True.",
    'gene_knowledge_image_blocks': "(dict) - Pass-through image embedding blocks, name -> gene table (gene + emb_* columns). Default {}.",
    'gene_knowledge_image_pca': "(int) - PCA dimensions of each image block. Default 64.",
    'gene_knowledge_image_nt_prefix': "(str) - Prefix of the image tables' non-targeting rows given to the decoys. Default 'NTpseudo'.",
    'gene_knowledge_n_folds': "(int) - Folds of the per-fold text blocks. Default 5.",
    'gene_knowledge_profile_reference_dir': "(str or list) - Folder(s) of the reference tables of the profile metrics (spacr.ops_reference). Default blank (no profile metrics).",
    'gene_knowledge_profile_references': "(list) - References whose SELECT and REPORT halves score the profiles. Default CORUM, STRING>=0.7, BioGRID.",
    'gene_knowledge_profile_sets': "(dict) - Profile name -> blocks scored by the profile metrics; {} uses every block and the standard concatenations. Default {}.",
    'gene_knowledge_device': "(str) - 'cuda' or 'cpu' for ESM-2, ProstT5 and the text model; CUDA falls back to the CPU. Default 'cuda'.",
    'gene_knowledge_gpu_cap_gb': "(float) - GPU memory cap of the embedding process (GB). Default 4.5.",
    'gene_knowledge_gpu_limit_gb': "(float) - Card total (GB) the admission rule keeps below. Default 19.",
    'gene_knowledge_admission_hold_s': "(int) - Seconds the card must stay below the limit before a GPU step starts. Default 180.",
    'gene_knowledge_gpu_log': "(str) - File the START / FINISH lines are appended to. Default '~/.spacr/gpu/log'.",
    'gene_knowledge_seed': "(int) - Seed of the decoy draw and every random stream. Default 20261109.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Protein knowledge inputs': [k for k in _DEFAULTS if k.startswith((
        'gene_knowledge_src', 'gene_knowledge_nodes', 'gene_knowledge_uniprot', 'gene_knowledge_allow',
        'gene_knowledge_decoy', 'gene_knowledge_models_dir', 'gene_knowledge_image', 'gene_knowledge_n_folds',
        'gene_knowledge_seed'))],
    'Protein knowledge models': [k for k in _DEFAULTS if k.startswith((
        'gene_knowledge_esm', 'gene_knowledge_text', 'gene_knowledge_domain', 'gene_knowledge_prostt5',
        'gene_knowledge_s3di', 'gene_knowledge_kmer', 'gene_knowledge_struct', 'gene_knowledge_afdb',
        'gene_knowledge_foldseek', 'gene_knowledge_paralog'))],
    'Protein knowledge compute': [k for k in _DEFAULTS if k.startswith((
        'gene_knowledge_device', 'gene_knowledge_gpu', 'gene_knowledge_admission'))],
    'Protein knowledge evaluation': [k for k in _DEFAULTS if k.startswith('gene_knowledge_profile')],
}

_DESCRIPTION = (
    "Protein-knowledge feature blocks for gene-level analysis: ESM-2 sequence "
    "embeddings, InterPro / Pfam domains, fold-masked UniProt function text, "
    "ProstT5 3Di structure and Foldseek similarity, with out-of-library decoys, "
    "archived-release time-split blocks and paralog-excluded profile metrics."
)


def get_gene_knowledge_default_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_gene_knowledge`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every ``gene_knowledge_*`` key.
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
    register_defaults(APP_KEY, get_gene_knowledge_default_settings, expected_types=_EXPECTED_TYPES,
                      tooltips=_TOOLTIPS, categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def options(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings without their prefix, list and dict values parsed.

    :param settings: ``gene_knowledge_*`` settings.
    :returns: ``{unprefixed key: value}``.
    """
    return gg.options(settings, PREFIX, get_gene_knowledge_default_settings)


def out_dir(o: Mapping[str, Any]) -> Path:
    """The module's output folder ``<src>/gene_knowledge``.

    :param o: unprefixed options.
    :returns: the folder.
    :raises ValueError: without ``src``.
    """
    if not o.get('src'):
        raise ValueError('gene_knowledge_src is blank: name the output folder')
    return Path(os.path.expanduser(o['src'])) / 'gene_knowledge'


def md5(seq: str) -> str:
    """The md5 hex digest of a sequence, the key every embedding is stored under.

    :param seq: the sequence.
    :returns: the digest.
    """
    return hashlib.md5(seq.encode()).hexdigest()


def _read_text(path: Any) -> Iterator[str]:
    """Lines of a plain, gzipped or tarred (``uniprot_sprot.dat.gz`` member) Swiss-Prot file."""
    path = os.fspath(path)
    if path.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(path, 'r:*') as tf:
            m = [x for x in tf.getmembers() if x.name.endswith('uniprot_sprot.dat.gz')][0]
            with gzip.open(tf.extractfile(m), 'rt') as fh:
                yield from fh
        return
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as fh:
        yield from fh


def swissprot_entries(lines: Iterable[str]) -> Iterator[List[str]]:
    """Split a Swiss-Prot ``.dat`` stream into entries (lines up to ``//``).

    :param lines: the text lines.
    :returns: an iterator of entries.
    """
    cur: List[str] = []
    for line in lines:
        if line.startswith('//'):
            yield cur
            cur = []
        else:
            cur.append(line.rstrip('\n'))


def parse_swissprot_entry(lines: Sequence[str]) -> Dict[str, Any]:
    """Accessions, gene name, CC FUNCTION text, Pfam / InterPro and sequence of one entry.

    :param lines: the entry's lines.
    :returns: ``acc``, ``gene``, ``function``, ``pfam``, ``interpro``, ``seq`` and ``human``.
    """
    d: Dict[str, Any] = {'acc': [], 'gene': None, 'function': '', 'pfam': [], 'interpro': [], 'seq': '',
                         'human': False}
    in_fn = in_sq = False
    for ln in lines:
        k, v = ln[:2], ln[5:]
        if k == 'AC':
            d['acc'] += [a.strip() for a in v.split(';') if a.strip()]
        elif k == 'OX' and 'NCBI_TaxID=9606' in v:
            d['human'] = True
        elif k == 'GN' and d['gene'] is None:
            m = re.search(r'Name=([^;{]+)', v)
            if m:
                d['gene'] = m.group(1).strip()
        elif k == 'CC':
            if v.startswith('-!- '):
                in_fn = v.startswith('-!- FUNCTION:')
                if in_fn:
                    d['function'] += ' ' + v[len('-!- FUNCTION:'):].strip()
            elif in_fn and v.startswith('    '):
                d['function'] += ' ' + v.strip()
            else:
                in_fn = False
        elif k == 'DR':
            t = [x.strip() for x in v.split(';')]
            if t[0] == 'Pfam':
                d['pfam'].append(t[1])
            elif t[0] == 'InterPro':
                d['interpro'].append(t[1])
        elif k == 'SQ':
            in_sq = True
        elif in_sq and k == '  ':
            d['seq'] += ln.replace(' ', '')
    return d


def parse_archive(path: Any, current: pd.DataFrame) -> pd.DataFrame:
    """Rows of the archived release for the current rows, matched by accession, else by gene name.

    :param path: the archived ``.dat`` (plain, gzipped or the release tarball).
    :param current: the current rows (gene, accession, source).
    :returns: gene, source, match2022, acc2022, acc_all, gene2022, function_2022,
        pfam_2022, interpro_2022 and sequence_2022 per current row.
    """
    want_acc = set(current.accession.dropna())
    want_gene = set(current.gene.dropna())
    rows = []
    for e in swissprot_entries(_read_text(path)):
        if not any(x.startswith('OX   NCBI_TaxID=9606') for x in e):
            continue
        d = parse_swissprot_entry(e)
        if set(d['acc']) & want_acc or d['gene'] in want_gene:
            rows.append({'acc2022': d['acc'][0], 'acc_all': ';'.join(d['acc']), 'gene2022': d['gene'],
                         'function_2022': d['function'].strip(), 'pfam_2022': ';'.join(d['pfam']),
                         'interpro_2022': ';'.join(d['interpro']), 'sequence_2022': d['seq']})
    R = pd.DataFrame(rows, columns=['acc2022', 'acc_all', 'gene2022', 'function_2022', 'pfam_2022',
                                    'interpro_2022', 'sequence_2022'])
    by_acc: Dict[str, int] = {}
    for i, r in R.iterrows():
        for a in r.acc_all.split(';'):
            by_acc.setdefault(a, i)
    by_gene = {g: i for i, g in R.gene2022.items() if g}
    out = []
    for _, r in current.iterrows():
        i = by_acc.get(r.accession)
        how = 'accession'
        if i is None:
            i, how = by_gene.get(r.gene), 'gene'
        if i is None:
            out.append({'gene': r.gene, 'source': r.source, 'match2022': 'none'})
            continue
        out.append({'gene': r.gene, 'source': r.source, 'match2022': how, **R.loc[i].to_dict()})
    return pd.DataFrame(out)


def http_get(url: str, tries: int = 5) -> str:
    """Fetch a URL as text, retrying with a growing pause.

    :param url: the URL.
    :param tries: attempts.
    :returns: the body.
    :raises RuntimeError: when every attempt failed.
    """
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode()
        except OSError as e:
            logger.warning('retry %s %s: %s', t, url[:80], e)
            time.sleep(5 * (t + 1))
    raise RuntimeError(f'could not fetch {url}')


def _tsv(text: str, path: Path) -> pd.DataFrame:
    """Keep a downloaded TSV under downloads/ and read it back through the table reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return gg._read(path, sep='\t')


def fetch_current(genes: Sequence[str], cache: pd.DataFrame, o: Mapping[str, Any], downloads: Path,
                  get: Callable[[str], str] = http_get) -> pd.DataFrame:
    """Current UniProt rows of the library and the seeded out-of-library decoy proteins.

    Accessions come from the function cache in batches of at most 100;
    genes without one are looked up by exact gene name (reviewed human).

    :param genes: library gene names.
    :param cache: the function cache (gene, accession, ...).
    :param o: unprefixed options.
    :param downloads: folder the raw downloads are kept in.
    :param get: fetches a URL as text.
    :returns: gene, accession, source, length, pfam, interpro, sequence, function_current.
    """
    base = 'https://rest.uniprot.org/uniprotkb/'
    acc = cache.dropna(subset=['accession'])
    A = acc.accession.tolist()
    parts = [_tsv(get(f'{base}accessions?format=tsv&size=500&fields={UNIPROT_FIELDS}&accessions=' + ','.join(A[i:i + 100])),
                  downloads / f'accessions_{i:05d}.tsv') for i in range(0, len(A), 100)]
    cur = pd.concat(parts, ignore_index=True)
    cur = cur.merge(acc[['gene', 'accession']], left_on='Entry', right_on='accession', how='right')
    extra = []
    for g in [g for g in genes if g not in set(cur.gene.dropna())]:
        q = urllib.parse.quote(f'(gene_exact:{g}) AND (organism_id:9606) AND (reviewed:true)')
        r = _tsv(get(f'{base}stream?format=tsv&fields={UNIPROT_FIELDS}&query={q}'), downloads / f'gene_{g}.tsv')
        if len(r):
            r = r.iloc[[0]].copy()
            r['gene'], r['accession'] = g, r.Entry
            extra.append(r)
    cur = pd.concat([cur] + extra, ignore_index=True)
    cur['source'] = 'library'
    q = urllib.parse.quote('(organism_id:9606) AND (reviewed:true)')
    allh = _tsv(get(f'{base}stream?format=tsv&fields=accession,gene_primary&query={q}'), downloads / 'human_reviewed_index.tsv')
    lib = set(cur.Entry.dropna()) | set(genes)
    pool = allh[~allh.Entry.isin(lib) & ~allh['Gene Names (primary)'].isin(lib) & allh['Gene Names (primary)'].notna()]
    pick = pool.sample(n=int(o['decoy_draw']), random_state=int(o['seed']))
    dec = _tsv(get(f'{base}accessions?format=tsv&size=500&fields={UNIPROT_FIELDS}&accessions=' + ','.join(pick.Entry)),
               downloads / 'decoys.tsv')
    dec = dec.set_index('Entry').loc[pick.Entry].reset_index()
    dec['gene'], dec['accession'], dec['source'] = dec['Gene Names (primary)'], dec.Entry, 'decoy_pool'
    out = pd.concat([cur, dec], ignore_index=True).rename(columns={
        'Length': 'length', 'Pfam': 'pfam', 'InterPro': 'interpro', 'Sequence': 'sequence',
        'Function [CC]': 'function_current'})
    return out[['gene', 'accession', 'source', 'length', 'pfam', 'interpro', 'sequence', 'function_current']]


def _nodes(o: Mapping[str, Any]) -> pd.DataFrame:
    """The gene graph's node table."""
    if not o.get('nodes'):
        raise ValueError('gene_knowledge_nodes is blank: name the gene graph nodes.parquet')
    return gg._read(o['nodes'])


def fetch_protein_annotations(settings: Mapping[str, Any], get: Callable[[str], str] = http_get) -> Dict[str, Any]:
    """Write the current and the archived UniProt rows of the library and the decoys.

    Stored tables are copied when given; otherwise the current rows are
    fetched (``gene_knowledge_allow_network``) and the archive is parsed.

    :param settings: the ``gene_knowledge_*`` settings.
    :param get: fetches a URL as text.
    :returns: ``current`` and ``archive`` frames.
    :raises RuntimeError: when fetching is needed but the network is not allowed.
    """
    o = options(settings)
    data = out_dir(o) / 'data'
    data.mkdir(parents=True, exist_ok=True)
    if o.get('uniprot_current_table'):
        cur = gg._read(o['uniprot_current_table'])
    else:
        if not o['allow_network']:
            raise RuntimeError('no gene_knowledge_uniprot_current_table and gene_knowledge_allow_network is off')
        nodes = _nodes(o)
        cache = gg._read(o['uniprot_function_cache'], sep='\t')
        cur = fetch_current(nodes.gene[~nodes.is_nt].tolist(), cache, o, out_dir(o) / 'downloads' / 'uniprot_current', get)
    gg._write(cur, data / 'uniprot_current.parquet')
    if o.get('uniprot_archive_table'):
        old = gg._read(o['uniprot_archive_table'])
    elif o.get('uniprot_archive'):
        old = parse_archive(o['uniprot_archive'], cur)
    else:
        old = pd.DataFrame({'gene': cur.gene, 'source': cur.source, 'match2022': 'none'})
    gg._write(old, data / 'uniprot_2022.parquet')
    return {'current': cur, 'archive': old}


def window_spans(n: int, win: int, stride: int) -> List[int]:
    """Start positions of the windows covering a sequence of length ``n``; the last window ends at ``n``.

    :param n: sequence length.
    :param win: window length.
    :param stride: stride.
    :returns: the starts.
    """
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] + win < n:
        starts.append(n - win)
    return starts


def windowed_mean(seqs: Sequence[str], embed_batch: Callable[[List[str]], List[np.ndarray]], win: int, stride: int,
                  tokens_per_batch: int, dim: int) -> np.ndarray:
    """Mean residue embedding per sequence, windows overlapping.

    Every residue's vector is the average over the windows that cover it;
    the sequence vector is the mean over residues. Windows are batched by
    length within a token budget.

    :param seqs: sequences.
    :param embed_batch: per-residue vectors (``len(window) x dim``) for a batch of windows.
    :param win: window length.
    :param stride: stride.
    :param tokens_per_batch: tokens per batch (window length + 2 per window).
    :param dim: embedding dimensions.
    :returns: ``len(seqs) x dim`` float32.
    """
    jobs = [(i, a, s[a:a + win]) for i, s in enumerate(seqs) for a in window_spans(len(s), win, stride)]
    jobs.sort(key=lambda x: len(x[2]))
    cov = [np.zeros(len(s)) for s in seqs]
    for i, a, ch in jobs:
        cov[i][a:a + len(ch)] += 1
    acc = np.zeros((len(seqs), dim), np.float64)
    b = 0
    while b < len(jobs):
        mx = len(jobs[min(b + max(1, tokens_per_batch // (len(jobs[b][2]) + 2)), len(jobs)) - 1][2]) + 2
        batch = jobs[b:b + max(1, tokens_per_batch // mx)]
        hs = embed_batch([x[2] for x in batch])
        for h, (i, a, ch) in zip(hs, batch):
            acc[i] += (np.asarray(h)[:len(ch)] / cov[i][a:a + len(ch), None]).sum(0)
        b += len(batch)
    return (acc / np.maximum(np.array([len(s) for s in seqs]), 1)[:, None]).astype(np.float32)


def _gpu_start(o: Mapping[str, Any], label: str) -> torch.device:
    """The device of an embedding step; on CUDA the card admission and cap are applied."""
    dev = gg._device(o)
    if dev.type == 'cuda':
        cap = float(o['gpu_cap_gb'])
        u = gg.admit(cap + 0.5, float(o['gpu_limit_gb']), float(o['admission_hold_s']))
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / (torch.cuda.mem_get_info()[1] / 2 ** 30)))
        gg.gpu_log(o['gpu_log'], label, f'START single process, cap {cap:.1f} GB (card {u:.1f} GB at admission)')
    return dev


def _hf_kwargs(o: Mapping[str, Any]) -> Dict[str, Any]:
    """Keyword arguments of ``from_pretrained``: the local cache, downloads only when allowed."""
    return {'cache_dir': o.get('models_dir') or None, 'local_files_only': not o['allow_download']}


def esm_embedder(o: Mapping[str, Any], dev: torch.device) -> Callable[[List[str]], List[np.ndarray]]:
    """An ESM-2 per-residue embedder from local weights (fp16 on CUDA, fp32 on the CPU).

    :param o: unprefixed options.
    :param dev: the device.
    :returns: batch of sequences -> per-residue last-layer vectors.
    :raises ImportError: without ``transformers``.
    """
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError as exc:
        raise ImportError('embed_sequences needs transformers: pip install transformers') from exc
    tok = AutoTokenizer.from_pretrained(o['esm_model'], **_hf_kwargs(o))
    model = EsmModel.from_pretrained(o['esm_model'], add_pooling_layer=False,
                                     torch_dtype=torch.float16 if dev.type == 'cuda' else torch.float32,
                                     **_hf_kwargs(o)).to(dev).eval()
    return functools.partial(_esm_batch, tok, model, dev)


def _esm_batch(tok, model, dev, batch: List[str]) -> List[np.ndarray]:
    """Per-residue ESM-2 vectors of a batch (the BOS token dropped)."""
    enc = tok(batch, return_tensors='pt', padding=True).to(dev)
    with torch.no_grad():
        h = model(**enc).last_hidden_state.float().cpu().numpy()
    return [h[k, 1:1 + len(s)] for k, s in enumerate(batch)]


def _sequences(o: Mapping[str, Any]) -> List[str]:
    """Every unique current and archived sequence, sorted by length."""
    data = out_dir(o) / 'data'
    seqs = set()
    for f, c in (('uniprot_current.parquet', 'sequence'), ('uniprot_2022.parquet', 'sequence_2022')):
        d = gg._read(data / f)
        if c in d:
            seqs |= set(d[c].dropna())
    return sorted((s for s in seqs if isinstance(s, str) and s), key=len)


def embed_sequences(settings: Mapping[str, Any],
                    embedder: Optional[Callable[[List[str]], List[np.ndarray]]] = None) -> Path:
    """ESM-2 mean embeddings of every unique sequence, stored by md5 in ``esm/esm_by_md5.npz``.

    A stored file (``gene_knowledge_esm_embeddings``) is copied instead.

    :param settings: the ``gene_knowledge_*`` settings.
    :param embedder: per-residue embedder; ESM-2 from local weights when None.
    :returns: the file written.
    """
    o = options(settings)
    out = out_dir(o) / 'esm'
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'esm_by_md5.npz'
    if o.get('esm_embeddings'):
        shutil.copyfile(os.path.expanduser(o['esm_embeddings']), target)
        return target
    seqs = _sequences(o)
    t0 = time.time()
    dev = torch.device('cpu')
    if embedder is None:
        dev = _gpu_start(o, 'gene-knowledge-esm')
        embedder = esm_embedder(o, dev)
    probe = embedder([seqs[0][:5]])[0] if seqs else np.zeros((1, 1))
    E = windowed_mean(seqs, embedder, int(o['esm_window']), int(o['esm_stride']), int(o['esm_tokens_per_batch']),
                      int(np.asarray(probe).shape[1]))
    np.savez_compressed(target, md5=np.array([md5(s) for s in seqs]), E=E, length=np.array([len(s) for s in seqs]))
    gg._write_json({'name': o['esm_model'], 'device': dev.type, 'seconds': time.time() - t0, 'sequences': len(seqs),
                    'window': int(o['esm_window']), 'stride': int(o['esm_stride'])}, out / 'done.json')
    return target


def clean_function_text(t: Any) -> str:
    """UniProt CC FUNCTION text without evidence tags and citations.

    :param t: the text.
    :returns: the cleaned text; ``""`` for a missing value.
    """
    if not isinstance(t, str):
        return ''
    t = re.sub(r'^FUNCTION:\s*', '', t)
    t = re.sub(r'\{ECO:[^}]*\}', '', t)
    t = re.sub(r'\((?:PubMed|Ref\.)[^)]*\)', '', t)
    t = t.replace('FUNCTION:', ' ')
    return re.sub(r'\s+', ' ', t).strip()


def split_sentences(t: str) -> List[str]:
    """Sentences of a function text, split after ``.`` or ``;`` before a capital or ``(``.

    :param t: cleaned text.
    :returns: sentences longer than three characters.
    """
    return [s.strip() for s in re.split(r'(?<=[.;])\s+(?=[A-Z(])', t) if len(s.strip()) > 3]


_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z0-9]')


def sentence_mentions(sentence: str, symbols: set, accession_gene: Mapping[str, str]) -> str:
    """The genes a sentence names, as whole-word symbols or accessions, ``;``-joined and sorted.

    :param sentence: the sentence.
    :param symbols: library symbols.
    :param accession_gene: accession -> gene.
    :returns: the mentions.
    """
    toks = set(_TOKEN.findall(sentence))
    m = {t for t in toks if t in symbols} | {accession_gene[t] for t in toks if t in accession_gene}
    return ';'.join(sorted(m))


def sentence_table(cache: pd.DataFrame, current: pd.DataFrame, archive: pd.DataFrame,
                   symbols: Iterable[str]) -> pd.DataFrame:
    """Every function sentence of every version with its sentence id and mentions.

    ``current`` sentences come from the function cache (library) and the
    current REST text (decoys); ``2022`` sentences from the archive only.

    :param cache: the function cache (gene, function).
    :param current: current rows.
    :param archive: archived rows.
    :param symbols: library symbols.
    :returns: version, gene, sentence, sid, mentions.
    """
    cur = current.drop_duplicates('gene')
    old = archive.drop_duplicates('gene')
    rows = []
    for g, f in zip(cache.gene, cache.function):
        rows += [('current', g, s) for s in split_sentences(clean_function_text(f))]
    for g, f in zip(cur.gene[cur.source == 'decoy_pool'], cur.function_current[cur.source == 'decoy_pool']):
        rows += [('current', g, s) for s in split_sentences(clean_function_text(f))]
    for g, f in zip(old.gene, old.get('function_2022', pd.Series([None] * len(old)))):
        rows += [('2022', g, s) for s in split_sentences(clean_function_text(f))]
    S = pd.DataFrame(rows, columns=['version', 'gene', 'sentence'])
    uniq = pd.Index(S.sentence.unique())
    S['sid'] = uniq.get_indexer(S.sentence)
    syms = set(symbols) | set(cache.gene)
    acc = dict(zip(cur.accession, cur.gene))
    ment = [sentence_mentions(s, syms, acc) for s in uniq]
    S['mentions'] = [ment[i] for i in S.sid]
    return S


def text_embedder(o: Mapping[str, Any], dev: torch.device) -> Callable[[List[str]], np.ndarray]:
    """A local sentence embedder: mean pooling over the attention mask, L2-normalised.

    :param o: unprefixed options.
    :param dev: the device.
    :returns: batch of sentences -> vectors.
    :raises ImportError: without ``transformers``.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError('embed_text needs transformers: pip install transformers') from exc
    tok = AutoTokenizer.from_pretrained(o['text_model'], **_hf_kwargs(o))
    mdl = AutoModel.from_pretrained(o['text_model'], **_hf_kwargs(o)).to(dev).eval()
    return functools.partial(_text_batch, tok, mdl, dev, int(o['text_max_tokens']))


def _text_batch(tok, mdl, dev, max_tokens: int, batch: List[str]) -> np.ndarray:
    """Mean-pooled, L2-normalised sentence vectors of a batch."""
    enc = tok(batch, padding=True, truncation=True, max_length=max_tokens, return_tensors='pt').to(dev)
    with torch.no_grad():
        h = mdl(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(v, dim=1).float().cpu().numpy()


def embed_text(settings: Mapping[str, Any], embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
               batch_size: int = 64) -> Dict[str, Path]:
    """Sentence table with mentions (``text/sentences.parquet``) and sentence vectors (``text/sent_emb.npy``).

    A stored folder (``gene_knowledge_text_embeddings``) is copied instead.

    :param settings: the ``gene_knowledge_*`` settings.
    :param embedder: sentence embedder; the local text model when None.
    :param batch_size: sentences per batch.
    :returns: ``sentences`` and ``embeddings`` paths.
    """
    o = options(settings)
    out = out_dir(o) / 'text'
    out.mkdir(parents=True, exist_ok=True)
    targets = {'sentences': out / 'sentences.parquet', 'embeddings': out / 'sent_emb.npy'}
    if o.get('text_embeddings'):
        src = Path(os.path.expanduser(o['text_embeddings']))
        for p in targets.values():
            shutil.copyfile(src / p.name, p)
        return targets
    data = out_dir(o) / 'data'
    nodes = _nodes(o)
    S = sentence_table(gg._read(o['uniprot_function_cache'], sep='\t'), gg._read(data / 'uniprot_current.parquet'),
                       gg._read(data / 'uniprot_2022.parquet'), nodes.sym[~nodes.is_nt])
    uniq = S.drop_duplicates('sid').sort_values('sid').sentence.tolist()
    if embedder is None:
        embedder = text_embedder(o, gg._device(o))
    order = np.argsort([len(s) for s in uniq], kind='stable')
    E = None
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        v = np.asarray(embedder([uniq[j] for j in idx]), np.float32)
        E = np.zeros((len(uniq), v.shape[1]), np.float32) if E is None else E
        E[idx] = v
    np.save(targets['embeddings'], E if E is not None else np.zeros((0, 1), np.float32))
    gg._write(S, targets['sentences'])
    return targets


class CNN(nn.Module):
    """ProstT5's 3Di prediction head (two convolutions over the residue axis), as published."""

    def __init__(self):
        """Create the two convolutions."""
        super().__init__()
        self.classifier = nn.Sequential(nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)), nn.ReLU(),
                                        nn.Dropout(0.0), nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)))

    def forward(self, x):
        """3Di logits (batch x 20 x residues) from encoder states (batch x residues x 1024).

        :param x: the encoder states (batch x residues x 1024).
        :returns: the 3Di logits (batch x 20 x residues).
        """
        return self.classifier(x.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)


def residue_batches(items: Sequence[Tuple[int, int, str]], max_residues: int) -> Iterator[List[Tuple[int, int, str]]]:
    """Consecutive batches of windows within a residue budget.

    :param items: ``(sequence index, start, window)``.
    :param max_residues: residues per batch.
    :returns: the batches.
    """
    b: List[Tuple[int, int, str]] = []
    n = 0
    for it in items:
        if b and n + len(it[2]) > max_residues:
            yield b
            b, n = [], 0
        b.append(it)
        n += len(it[2])
    if b:
        yield b


def three_di(seqs: Sequence[str], encode: Callable[[List[str], List[int]], List[torch.Tensor]],
             head: Callable[[torch.Tensor], torch.Tensor], win: int, stride: int, max_residues: int
             ) -> Tuple[List[str], np.ndarray]:
    """Predicted 3Di strings and their mean encoder embedding.

    Pass 1: each window ``<AA2fold>`` is encoded and the head's argmax gives
    the 3Di state; a residue covered by two windows takes the first one's.
    Pass 2: the 3Di string (``<fold2AA>``, lowercase) is encoded and the
    states are averaged over residues.

    :param seqs: amino-acid sequences.
    :param encode: texts and lengths -> per-residue encoder states.
    :param head: encoder states (1 x L x d) -> logits (1 x 20 x L).
    :param win: window length.
    :param stride: stride.
    :param max_residues: residues per batch.
    :returns: the 3Di strings and ``len(seqs) x d`` embeddings.
    """
    items = [(i, a, s[a:a + win]) for i, s in enumerate(seqs) for a in window_spans(len(s), win, stride)]
    tdi: List[List[Optional[str]]] = [[None] * len(s) for s in seqs]
    for b in residue_batches(items, max_residues):
        texts = ['<AA2fold> ' + ' '.join(''.join(c if c in SS else 'X' for c in w)) for _, _, w in b]
        for (i, a, w), h in zip(b, encode(texts, [len(w) for _, _, w in b])):
            with torch.no_grad():
                y = head(h[None]).argmax(1)[0].cpu().numpy()
            for k, c in enumerate(y):
                if tdi[i][a + k] is None:
                    tdi[i][a + k] = SS[int(c)]
    strings = [''.join(x) for x in tdi]
    items = [(i, a, s[a:a + win]) for i, s in enumerate(strings) for a in window_spans(len(s), win, stride)]
    acc: List[Optional[np.ndarray]] = [None] * len(seqs)
    cnt = np.zeros(len(seqs))
    for b in residue_batches(items, max_residues):
        texts = ['<fold2AA> ' + ' '.join(w.lower()) for _, _, w in b]
        for (i, a, w), h in zip(b, encode(texts, [len(w) for _, _, w in b])):
            v = h.float().sum(0).cpu().numpy()
            acc[i] = v if acc[i] is None else acc[i] + v
            cnt[i] += len(w)
    dim = next((len(a) for a in acc if a is not None), 1)
    E = np.stack([(a / c) if a is not None else np.zeros(dim) for a, c in zip(acc, cnt)]) if seqs else np.zeros((0, dim))
    return strings, E.astype(np.float32)


def _prostt5(o: Mapping[str, Any], dev: torch.device):
    """The ProstT5 encoder (as a batch encoder) and its CNN head from local weights."""
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError('predict_3di needs transformers and sentencepiece') from exc
    half = dev.type == 'cuda'
    tok = T5Tokenizer.from_pretrained(o['prostt5_model'], do_lower_case=False, **_hf_kwargs(o))
    model = T5EncoderModel.from_pretrained(o['prostt5_model'], torch_dtype=torch.float16 if half else torch.float32,
                                           **_hf_kwargs(o)).to(dev).eval()
    cnn = CNN()
    cnn.load_state_dict(torch.load(os.path.expanduser(o['prostt5_cnn']), map_location='cpu')['state_dict'])
    cnn = (cnn.half() if half else cnn.float()).to(dev).eval()
    return functools.partial(_t5_encode, tok, model, dev), functools.partial(_cnn_head, cnn)


def _t5_encode(tok, model, dev, texts: List[str], lens: List[int]) -> List[torch.Tensor]:
    """Per-residue encoder states of a batch of prefixed texts."""
    enc = tok.batch_encode_plus(texts, add_special_tokens=True, padding='longest', return_tensors='pt').to(dev)
    with torch.no_grad():
        h = model(enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
    return [h[i, 1:1 + L] for i, L in enumerate(lens)]


def _cnn_head(cnn, h: torch.Tensor) -> torch.Tensor:
    """The CNN head in the encoder's dtype."""
    return cnn(h.to(next(cnn.parameters()).dtype))


def read_fasta(path: Any) -> Dict[str, str]:
    """Records of a FASTA file, name -> sequence.

    :param path: the file.
    :returns: the records.
    """
    out: Dict[str, str] = {}
    name = None
    with open(os.path.expanduser(os.fspath(path)), encoding='utf-8') as fh:
        for ln in fh:
            if ln.startswith('>'):
                name = ln[1:].strip()
                out[name] = ''
            elif name is not None:
                out[name] += ln.strip()
    return out


def predict_3di(settings: Mapping[str, Any], encode: Optional[Callable] = None,
                head: Optional[Callable] = None) -> Dict[str, Path]:
    """ProstT5 3Di strings and 3Di embeddings of every current sequence.

    Writes ``prostt5/3di.fasta``, ``prostt5/aa.fasta`` (records named by md5)
    and ``prostt5/emb3di_by_md5.npz``. A stored folder
    (``gene_knowledge_prostt5_outputs``) is copied instead.

    :param settings: the ``gene_knowledge_*`` settings.
    :param encode: batch encoder; ProstT5 from local weights when None.
    :param head: the 3Di head; ProstT5's CNN when None.
    :returns: the paths.
    """
    o = options(settings)
    out = out_dir(o) / 'prostt5'
    out.mkdir(parents=True, exist_ok=True)
    targets = {k: out / k for k in ('3di.fasta', 'aa.fasta', 'emb3di_by_md5.npz')}
    if o.get('prostt5_outputs'):
        src = Path(os.path.expanduser(o['prostt5_outputs']))
        for p in targets.values():
            shutil.copyfile(src / p.name, p)
        return targets
    cur = gg._read(out_dir(o) / 'data' / 'uniprot_current.parquet')
    seqs = sorted({s for s in cur.sequence.dropna() if s}, key=len)
    if encode is None or head is None:
        encode, head = _prostt5(o, _gpu_start(o, 'gene-knowledge-prostt5'))
    tdi, E = three_di(seqs, encode, head, int(o['prostt5_window']), int(o['prostt5_stride']),
                      int(o['prostt5_residues_per_batch']))
    keys = [md5(s) for s in seqs]
    np.savez_compressed(targets['emb3di_by_md5.npz'], md5=np.array(keys), E=E)
    targets['3di.fasta'].write_text(''.join(f'>{m}\n{t}\n' for m, t in zip(keys, tdi)), encoding='utf-8')
    targets['aa.fasta'].write_text(''.join(f'>{m}\n{s}\n' for m, s in zip(keys, seqs)), encoding='utf-8')
    return targets


def run_foldseek(o: Mapping[str, Any], folder: Path, prostt5: Path) -> Path:
    """Run the all-vs-all Foldseek search and return its alignment table.

    ``prostt5`` builds both databases from sequence with ``base:createdb``
    (amino acids, and the predicted 3Di as ``<db>_ss``, unshuffled);
    ``afdb`` builds one from the structure files with pLDDT masking.

    :param o: unprefixed options.
    :param folder: the structure folder.
    :param prostt5: the ProstT5 output folder.
    :returns: ``aln.m8``.
    """
    fs = [os.path.expanduser(o['foldseek_bin'])]
    db = folder / 'db'
    if o['structure_source'] == 'afdb':
        subprocess.run(fs + ['createdb', os.path.expanduser(o['afdb_dir']), str(db), '--mask-bfactor-threshold', '70'],
                       check=True)
    else:
        subprocess.run(fs + ['base:createdb', str(prostt5 / 'aa.fasta'), str(db), '--shuffle', '0'], check=True)
        subprocess.run(fs + ['base:createdb', str(prostt5 / '3di.fasta'), f'{db}_ss', '--shuffle', '0'], check=True)
    subprocess.run(fs + ['search', str(db), str(db), str(folder / 'aln'), str(folder / 'tmp'),
                         '--exhaustive-search', str(int(bool(o['foldseek_exhaustive']))),
                         '-e', str(o['foldseek_evalue']), '--threads', str(o['foldseek_threads'])], check=True)
    subprocess.run(fs + ['convertalis', str(db), str(db), str(folder / 'aln'), str(folder / 'aln.m8'),
                         '--format-output', 'query,target,fident,evalue,bits'], check=True)
    return folder / 'aln.m8'


def read_alignments(path: Any) -> pd.DataFrame:
    """A Foldseek alignment table (query, target, fident, evalue, bits), self hits removed.

    :param path: the tab-separated ``.m8`` file.
    :returns: q, t, fident, evalue, bits.
    """
    rows = []
    with open(os.path.expanduser(os.fspath(path)), encoding='utf-8') as fh:
        for line in fh:
            t = line.rstrip('\n').split('\t')
            if len(t) >= 5:
                rows.append((t[0], t[1], float(t[2]), float(t[3]), float(t[4])))
    m8 = pd.DataFrame(rows, columns=['q', 't', 'fident', 'evalue', 'bits'])
    return m8[m8.q != m8.t].reset_index(drop=True)


def structure_pairs(m8: pd.DataFrame, md5_of: Sequence[Optional[str]], lib: np.ndarray,
                    fident: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    """Structural-similarity edges and paralog pairs on the node order.

    Every hit (either direction) between two nodes is an edge; a hit above
    ``fident`` between two library genes, or two library nodes with an
    identical sequence, is a paralog pair (and an edge).

    :param m8: alignments.
    :param md5_of: each node's sequence md5, or None.
    :param lib: library-gene mask.
    :param fident: the paralog identity cut.
    :returns: ``(E_struct, E_paralog)``, sorted unique pairs ``i < j``.
    """
    nodes_of: Dict[str, List[int]] = {}
    for i, m in enumerate(md5_of):
        if m:
            nodes_of.setdefault(m, []).append(i)
    pairs, para = set(), set()
    for q, t, fi in zip(m8.q, m8.t, m8.fident):
        for a in nodes_of.get(q, []):
            for b in nodes_of.get(t, []):
                if a != b:
                    e = (min(a, b), max(a, b))
                    pairs.add(e)
                    if fi > fident and lib[a] and lib[b]:
                        para.add(e)
    for ii in nodes_of.values():
        for a in ii:
            for b in ii:
                if a < b and lib[a] and lib[b]:
                    para.add((a, b))
                    pairs.add((a, b))
    return (np.array(sorted(pairs), np.int32).reshape(-1, 2), np.array(sorted(para), np.int32).reshape(-1, 2))


def node_proteins(nodes: pd.DataFrame, current: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """The protein of every node: the library gene itself, and an out-of-library decoy per pseudo-gene.

    :param nodes: the node table.
    :param current: current rows.
    :returns: the node -> protein names and the decoy rows used.
    :raises ValueError: when there are fewer decoy proteins than pseudo-genes.
    """
    lib = ~nodes.is_nt.to_numpy()
    n_nt = int((~lib).sum())
    dec = current[(current.source == 'decoy_pool') & current.sequence.notna()].drop_duplicates('gene').iloc[:n_nt]
    if len(dec) < n_nt:
        raise ValueError(f'{len(dec)} decoy proteins with a sequence for {n_nt} pseudo-genes')
    return list(nodes.gene[lib]) + list(dec.gene), dec


def foldseek_similarity(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Structural-similarity edges and paralog pairs from an all-vs-all Foldseek search.

    A stored alignment table (``gene_knowledge_foldseek_alignments``) is read
    instead of searching.

    :param settings: the ``gene_knowledge_*`` settings.
    :returns: ``E_struct``, ``E_paralog``, ``hits`` and ``md5_of``.
    """
    o = options(settings)
    folder = out_dir(o) / 'structure'
    folder.mkdir(parents=True, exist_ok=True)
    path = (Path(os.path.expanduser(o['foldseek_alignments'])) if o.get('foldseek_alignments')
            else run_foldseek(o, folder, out_dir(o) / 'prostt5'))
    m8 = read_alignments(path)
    nodes = _nodes(o)
    cur = gg._read(out_dir(o) / 'data' / 'uniprot_current.parquet')
    proteins, _ = node_proteins(nodes, cur)
    seq = cur.drop_duplicates('gene').set_index('gene').sequence
    md5_of = [md5(seq[g]) if g in seq.index and isinstance(seq[g], str) else None for g in proteins]
    E, EP = structure_pairs(m8, md5_of, ~nodes.is_nt.to_numpy(), float(o['paralog_fident']))
    return {'E_struct': E, 'E_paralog': EP, 'hits': int(len(m8)), 'md5_of': md5_of}


def rms(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``X`` divided by the root mean square of its ``ref`` rows, float32.

    :param X: the block.
    :param ref: rows the scale is measured on.
    :returns: the scaled block.
    """
    return (X / np.sqrt((X[ref] ** 2).mean())).astype(np.float32)


def pca_block(M: np.ndarray, have: np.ndarray, lib: np.ndarray, k: int, seed: int = 0) -> Tuple[np.ndarray, float]:
    """A PCA block fit on the present library rows only, unit RMS on those rows.

    :param M: the raw rows.
    :param have: rows present.
    :param lib: library rows.
    :param k: components (at most fit rows - 1).
    :param seed: PCA seed.
    :returns: the block (absent rows zero) and the explained variance.
    """
    from sklearn.decomposition import PCA

    fit = have & lib
    mu = M[fit].mean(0)
    p = PCA(min(k, int(fit.sum()) - 1, M.shape[1]), random_state=seed).fit(M[fit] - mu)
    X = np.zeros((len(M), p.n_components_), np.float32)
    X[have] = p.transform(M[have] - mu)
    return rms(X, fit), float(p.explained_variance_ratio_.sum())


def _by_md5(path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """Stored embeddings and their md5 index."""
    e = np.load(path)
    return e['E'], {m: i for i, m in enumerate(e['md5'])}


def _md5_matrix(seqs: Sequence[Any], E: np.ndarray, index: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of stored embeddings for each node's sequence."""
    M = np.zeros((len(seqs), E.shape[1]), np.float32)
    h = np.zeros(len(seqs), bool)
    for i, s in enumerate(seqs):
        if isinstance(s, str) and s:
            j = index.get(md5(s))
            if j is not None:
                M[i] = E[j]
                h[i] = True
    return M, h


def domain_block(entries: Sequence[Optional[str]], lib: np.ndarray, min_genes: int, k: int):
    """InterPro + Pfam membership, entries in at least ``min_genes`` library genes, TF-IDF then SVD.

    :param entries: each node's ``;``-joined entries, None without a row.
    :param lib: library rows.
    :param min_genes: vocabulary cut.
    :param k: SVD dimensions.
    :returns: the block, its presence, the vocabulary, the membership matrix and a summary.
    """
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    N = len(entries)
    h = np.array([x is not None for x in entries])
    sets = [set(t for t in (x or '').split(';') if t) for x in entries]
    cnt = pd.Series([t for i, st in enumerate(sets) if lib[i] for t in st], dtype=object).value_counts()
    vocab = sorted(cnt.index[cnt >= min_genes])
    vi = {t: j for j, t in enumerate(vocab)}
    r, c = [], []
    for i, st in enumerate(sets):
        for t in st:
            if t in vi:
                r.append(i)
                c.append(vi[t])
    B = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(N, len(vocab)))
    T = TfidfTransformer().fit(B[lib & h]).transform(B)
    svd = TruncatedSVD(min(k, max(len(vocab) - 1, 1)), random_state=0).fit(T[lib & h])
    X = svd.transform(T).astype(np.float32)
    X[~h] = 0
    summ = {'dims': int(X.shape[1]), 'vocab': len(vocab), 'coverage_lib': float(h[lib].mean()),
            'lib_with_no_entry': int(((np.asarray(B.sum(1)).ravel() == 0) & lib & h).sum()),
            'svd_var': float(svd.explained_variance_ratio_.sum())}
    return rms(X, lib & h), h, vocab, B, summ


def text_block(S: pd.DataFrame, SE: np.ndarray, version: str, node_of: Mapping[str, int], names: Sequence[str],
               genes: Sequence[str], test: np.ndarray, lib: np.ndarray, k: int,
               mask_mentions: bool = True) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """One text block: mean kept-sentence vector per node, PCA fit on the present library rows.

    Test nodes get no vector, and a sentence naming a test node (by symbol
    or by node name) is dropped from every other node.

    :param S: the sentence table.
    :param SE: sentence vectors.
    :param version: ``current`` or ``2022``.
    :param node_of: protein -> node.
    :param names: each node's symbol (decoys: the protein).
    :param genes: each node's name.
    :param test: held-out nodes.
    :param lib: library rows.
    :param k: PCA dimensions.
    :param mask_mentions: drop sentences naming held-out nodes.
    :returns: the block, its presence, the sentences dropped and the explained variance.
    """
    d = S[S.version == version]
    hidden = (set(np.asarray(names)[test]) | set(np.asarray(genes)[test])) if mask_mentions else set()
    M = np.zeros((len(names), SE.shape[1]), np.float32)
    h = np.zeros(len(names), bool)
    dropped = 0
    for g, dd in d.groupby('gene'):
        i = node_of.get(g)
        if i is None or test[i]:
            continue
        keep = [sid for sid, m in zip(dd.sid, dd.mentions) if not (set(str(m).split(';')) - {''}) & hidden]
        dropped += len(dd) - len(keep)
        if keep:
            M[i] = SE[keep].mean(0)
            h[i] = True
    X, var = pca_block(M, h, lib, k)
    return X, h, dropped, var


def spectral_embedding(E: np.ndarray, N: int, k: int, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Laplacian eigenmaps of an edge list, unit RMS on ``ref`` rows; isolated nodes zero.

    :param E: edges.
    :param N: nodes.
    :param k: dimensions.
    :param ref: rows of the RMS scale (with an edge).
    :returns: ``(X, present)``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    A = sp.csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(N, N))
    A.data[:] = 1
    d = np.asarray(A.sum(1)).ravel()
    dinv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0)
    k = min(k, N - 2)
    vals, vecs = eigsh(sp.diags(dinv) @ A @ sp.diags(dinv), k=k + 1, which='LA', v0=np.random.default_rng(0).random(N))
    V = vecs[:, np.argsort(-vals)[1:]].astype(np.float32)
    hs = d > 0
    V[~hs] = 0
    return rms(V, hs & ref), hs


def build_knowledge_blocks(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Assemble every knowledge block on the gene graph's node order.

    Writes ``blocks.npz`` (``X_esm``, ``X_esm_time``, ``X_dom``,
    ``X_dom_time``, ``X_txt_f<k>``, ``X_txt_time``, ``X_txt_all`` (every
    gene's own text: a circular profile reference only), ``X_<image block>``,
    each with ``H_*``), ``structure.npz`` (``X_s3di``, ``X_s3di_kmer``,
    ``X_struct_spec``, ``E_struct``, ``E_paralog_fident30``), the domain
    membership and vocabulary, ``node_protein.parquet``,
    ``blocks_summary.json`` and ``structure_summary.json``.

    :param settings: the ``gene_knowledge_*`` settings.
    :returns: the two summaries.
    """
    import scipy.sparse as sp

    o = options(settings)
    root = out_dir(o)
    data = root / 'data'
    nodes = _nodes(o)
    N = len(nodes)
    lib = ~nodes.is_nt.to_numpy()
    fold = nodes.fold.to_numpy()
    cur = gg._read(data / 'uniprot_current.parquet')
    old = gg._read(data / 'uniprot_2022.parquet')
    proteins, dec = node_proteins(nodes, cur)
    libc = cur[cur.source == 'library'].drop_duplicates('gene').set_index('gene')
    libo = old[old.source == 'library'].drop_duplicates('gene').set_index('gene')
    deco = old[old.source == 'decoy_pool'].drop_duplicates('gene').set_index('gene')
    rowc = [libc.loc[g] if g in libc.index else None for g in nodes.gene[lib]] + [r for _, r in dec.iterrows()]
    rowo = ([libo.loc[g] if g in libo.index else None for g in nodes.gene[lib]]
            + [deco.loc[g] if g in deco.index else None for g in dec.gene])
    z: Dict[str, np.ndarray] = {}
    summ: Dict[str, Any] = {'decoy_proteins': list(dec.gene), 'blocks': {}}
    E, em = _by_md5(root / 'esm' / 'esm_by_md5.npz')
    Mc, hc = _md5_matrix([r.sequence if r is not None else None for r in rowc], E, em)
    Mo, ho = _md5_matrix([r.get('sequence_2022') if r is not None else None for r in rowo], E, em)
    from sklearn.decomposition import PCA

    fit = hc & lib
    mu = Mc[fit].mean(0)
    k = min(int(o['esm_pca']), int(fit.sum()) - 1, Mc.shape[1])
    p = PCA(k, random_state=0).fit(Mc[fit] - mu)
    s = np.sqrt((p.transform(Mc[fit] - mu) ** 2).mean())
    for name, M, h in (('esm', Mc, hc), ('esm_time', Mo, ho)):
        X = np.zeros((N, k), np.float32)
        if h.any():
            X[h] = p.transform(M[h] - mu) / s
        z[f'X_{name}'], z[f'H_{name}'] = X, h
        summ['blocks'][name] = {'dims': k, 'coverage_lib': float(h[lib].mean()), 'coverage_nt': float(h[~lib].mean()),
                                'pca_var': float(p.explained_variance_ratio_.sum())}

    def entries(rows, a, b):
        """Each node's ``;``-joined Pfam and InterPro entries."""
        return [None if r is None else ';'.join(str(x) for x in (r.get(a), r.get(b)) if isinstance(x, str))
                for r in rows]

    X, h, vocab_c, Bc, ds = domain_block(entries(rowc, 'pfam', 'interpro'), lib, int(o['domain_min_genes']),
                                         int(o['domain_svd']))
    z['X_dom'], z['H_dom'], summ['blocks']['dom'] = X, h, ds
    X, h, _, _, ds = domain_block(entries(rowo, 'pfam_2022', 'interpro_2022'), lib, int(o['domain_min_genes']),
                                  int(o['domain_svd']))
    z['X_dom_time'], z['H_dom_time'], summ['blocks']['dom_time'] = X, h, ds
    sp.save_npz(data / 'dom_membership_current.npz', Bc.tocsr())
    gg._write_json(vocab_c, data / 'dom_vocab_current.json')
    S = gg._read(root / 'text' / 'sentences.parquet')
    SE = np.load(root / 'text' / 'sent_emb.npy')
    node_of = {g: i for i, g in enumerate(proteins)}
    names = list(nodes.sym[lib]) + list(dec.gene)
    genes = list(nodes.gene)
    kt, mask = int(o['text_pca']), bool(o['text_mask_mentions'])
    for f in range(int(o['n_folds'])):
        X, h, dr, var = text_block(S, SE, 'current', node_of, names, genes, fold == f, lib, kt, mask)
        z[f'X_txt_f{f}'], z[f'H_txt_f{f}'] = X, h
        summ['blocks'][f'txt_f{f}'] = {'dims': int(X.shape[1]), 'coverage_lib': float(h[lib].mean()),
                                       'coverage_test': float(h[fold == f].mean()),
                                       'sentences_dropped_naming_fold_genes': int(dr), 'pca_var': var}
    none = np.zeros(N, bool)
    X, h, _, var = text_block(S, SE, '2022', node_of, names, genes, none, lib, kt, mask)
    z['X_txt_time'], z['H_txt_time'] = X, h
    summ['blocks']['txt_time'] = {'dims': int(X.shape[1]), 'coverage_lib': float(h[lib].mean()), 'pca_var': var}
    z['X_txt_all'], z['H_txt_all'], _, _ = text_block(S, SE, 'current', node_of, names, genes, none, lib, kt, mask)
    for b, f in (o['image_blocks'] or {}).items():
        d = gg._read(f)
        cols = [c for c in d.columns if str(c).startswith('emb_')]
        d = d.set_index('gene')
        ntn = sorted(g for g in d.index if str(g).startswith(o['image_nt_prefix']))
        rows = list(nodes.gene[lib]) + ntn[:int((~lib).sum())]
        M = np.zeros((N, len(cols)), np.float32)
        h = np.zeros(N, bool)
        for i, g in enumerate(rows):
            if g in d.index:
                M[i] = d.loc[g, cols].to_numpy(float)
                h[i] = True
        X, var = pca_block(M, h, lib, int(o['image_pca']))
        z[f'X_{b}'], z[f'H_{b}'] = X, h
        summ['blocks'][b] = {'dims': int(X.shape[1]), 'coverage_lib': float(h[lib].mean()),
                             'coverage_nt': float(h[~lib].mean()), 'pca_var': var}
    np.savez_compressed(root / 'blocks.npz', **z)
    gg._write(pd.DataFrame({'node': np.arange(N), 'gene': nodes.gene, 'protein': proteins}), root / 'node_protein.parquet')
    gg._write_json(summ, root / 'blocks_summary.json')
    sz = structure_blocks(o, nodes, cur, proteins)
    return {'blocks': summ, 'structure': sz}


def structure_blocks(o: Mapping[str, Any], nodes: pd.DataFrame, cur: pd.DataFrame,
                     proteins: Sequence[str]) -> Dict[str, Any]:
    """The 3Di embedding, 3Di k-mer and structural-graph blocks, the edges and the paralog pairs.

    :param o: unprefixed options.
    :param nodes: the node table.
    :param cur: current rows.
    :param proteins: node -> protein.
    :returns: the structure summary (also written to ``structure_summary.json``).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    root = out_dir(o)
    pt = root / 'prostt5'
    N = len(nodes)
    lib = ~nodes.is_nt.to_numpy()
    fs = foldseek_similarity({PREFIX + k: v for k, v in o.items()})
    E, EP, md5_of = fs['E_struct'], fs['E_paralog'], fs['md5_of']
    z: Dict[str, np.ndarray] = {'E_struct': E, 'E_paralog_fident30': EP}
    emb, ei = _by_md5(pt / 'emb3di_by_md5.npz')
    M = np.zeros((N, emb.shape[1]), np.float32)
    h = np.zeros(N, bool)
    for i, m in enumerate(md5_of):
        if m in ei:
            M[i] = emb[ei[m]]
            h[i] = True
    z['X_s3di'], var = pca_block(M, h, lib, int(o['s3di_pca']))
    z['H_s3di'] = h
    tdi = read_fasta(pt / '3di.fasta')
    docs = [tdi.get(m, '') if m else '' for m in md5_of]
    kk = int(o['kmer'])
    C = CountVectorizer(analyzer='char', ngram_range=(kk, kk), lowercase=False).fit_transform(docs)
    T = TfidfTransformer(sublinear_tf=True).fit(C[lib & h]).transform(C)
    X = TruncatedSVD(min(int(o['kmer_svd']), C.shape[1] - 1), random_state=0).fit(T[lib & h]).transform(T).astype(np.float32)
    X[~h] = 0
    z['X_s3di_kmer'], z['H_s3di_kmer'] = rms(X, lib & h), h
    z['X_struct_spec'], z['H_struct_spec'] = spectral_embedding(E, N, int(o['struct_spec_dims']), lib)
    hs = z['H_struct_spec']
    np.savez_compressed(root / 'structure.npz', **z)
    summ = {'source': ('ProstT5-predicted 3Di' if o['structure_source'] == 'prostt5' else 'AFDB v4 structures'),
            'proteins_with_3di': int(h.sum()), 'lib_coverage': float(h[lib].mean()),
            'foldseek_hits_E1e-3': fs['hits'], 'struct_edges': int(len(E)),
            'struct_edges_lib_lib': int((lib[E[:, 0]] & lib[E[:, 1]]).sum()) if len(E) else 0,
            'nodes_with_struct_edge': int(hs.sum()), 'paralog_pairs_fident30': int(len(EP)), 's3di_pca_var': var,
            'plddt_mask': ('not applicable (no coordinates for predicted 3Di)' if o['structure_source'] == 'prostt5'
                           else 'pLDDT < 70 masked'),
            'tm_score': ('not available (no coordinates)' if o['structure_source'] == 'prostt5' else 'available')}
    gg._write_json(summ, root / 'structure_summary.json')
    return summ


def paralog_mask(N: int, paralog_pairs: Optional[np.ndarray] = None, membership: Any = None,
                 vocab: Optional[Sequence[str]] = None, shared_pfam: bool = True) -> np.ndarray:
    """Paralog pairs: Foldseek identity above the cut, or a shared Pfam family.

    :param N: nodes.
    :param paralog_pairs: the identity pairs (``E_paralog_fident30``).
    :param membership: node x entry membership (sparse) of the current domains.
    :param vocab: the entries of ``membership``; ``PF*`` entries are Pfam.
    :param shared_pfam: add the shared-Pfam pairs.
    :returns: a symmetric N x N boolean matrix with a false diagonal.
    """
    M = np.zeros((N, N), bool)
    if paralog_pairs is not None and len(paralog_pairs):
        M[paralog_pairs[:, 0], paralog_pairs[:, 1]] = True
    if shared_pfam and membership is not None and vocab is not None:
        pf = [i for i, v in enumerate(vocab) if str(v).startswith('PF')]
        if pf:
            Bp = membership.tocsc()[:, pf].astype(np.float32)
            M |= (Bp @ Bp.T).toarray() > 0
    M |= M.T
    np.fill_diagonal(M, False)
    return M


def load_paralog_mask(knowledge_dir: Any, N: int, shared_pfam: bool = True) -> np.ndarray:
    """:func:`paralog_mask` from a knowledge output folder.

    :param knowledge_dir: ``<src>/gene_knowledge``.
    :param N: nodes.
    :param shared_pfam: add the shared-Pfam pairs.
    :returns: the mask.
    """
    import scipy.sparse as sp

    root = Path(knowledge_dir)
    pairs = np.load(root / 'structure.npz')['E_paralog_fident30'] if (root / 'structure.npz').exists() else None
    B = sp.load_npz(root / 'data' / 'dom_membership_current.npz')
    vocab = json.loads((root / 'data' / 'dom_vocab_current.json').read_text(encoding='utf-8'))
    return paralog_mask(N, pairs, B, vocab, shared_pfam)


_PROFILE_SETS = {
    'ops': ['ops'], 'img_dino': ['img_dino'], 'img_rn50': ['img_rn50'], 'esm': ['esm'], 'dom': ['dom'],
    'txt (circular)': ['txt_all'], 's3di': ['s3di'], 's3di_kmer': ['s3di_kmer'], 'struct_spec': ['struct_spec'],
    'ops+img': ['ops', 'img_dino', 'img_rn50'], 'esm+dom': ['esm', 'dom'], 'esm+ops': ['esm', 'ops'],
    'esm+dom+ops': ['esm', 'dom', 'ops'], 'esm+dom+s3di': ['esm', 'dom', 's3di'],
    'esm+dom+s3di+ops+img': ['esm', 'dom', 's3di', 'ops', 'img_dino', 'img_rn50']}


def profile_metrics(settings: Mapping[str, Any], screen_blocks: Optional[Any] = None) -> pd.DataFrame:
    """SELECT / REPORT retrieval mAP of every block and concatenation, plus paralog-excluded SELECT.

    Each profile is the concatenation of its blocks (absent rows zero) on
    the library genes; the score is the mean per-gene average precision
    (:func:`spacr.sp_stats.pair_map`) of retrieving each reference half's
    partners. The paralog-excluded variant ranks paralog pairs last and
    removes them from the positives. A control row shuffles the gene labels
    of the full concatenation.

    :param settings: the ``gene_knowledge_*`` settings.
    :param screen_blocks: a gene graph ``graph.npz`` (or its dict) with the screen blocks (``X_ops``).
    :returns: the table (also written to ``profile_metrics.csv``).
    """
    from .ops_reference import positive_matrix, reference_pairs
    from .sp_stats import pair_map, profile_cosine

    o = options(settings)
    root = out_dir(o)
    z: Dict[str, np.ndarray] = {}
    if screen_blocks is not None:
        z.update(dict(np.load(screen_blocks)) if isinstance(screen_blocks, (str, os.PathLike)) else screen_blocks)
    z.update(dict(np.load(root / 'blocks.npz')))
    if (root / 'structure.npz').exists():
        z.update(dict(np.load(root / 'structure.npz')))
    nodes = _nodes(o)
    lib = ~nodes.is_nt.to_numpy()
    genes = nodes.gene.to_numpy()[lib]
    refs = [f'{r}@{h}' for r in o['profile_references'] for h in ('SELECT', 'REPORT')]
    POS = {r: positive_matrix(genes, reference_pairs(r, o['profile_reference_dir'])) for r in refs}
    PARA = load_paralog_mask(root, len(nodes), bool(o['paralog_shared_pfam']))[np.ix_(lib, lib)]
    sets = o['profile_sets'] or _PROFILE_SETS

    def concat(bl, perm=None):
        """The concatenated profile of blocks on library rows."""
        X = np.concatenate([z[f'X_{b}'][lib] * z[f'H_{b}'][lib][:, None] for b in bl], 1)
        return X if perm is None else X[perm]

    def scored(S, POSd, name, bl):
        """One row of mAPs."""
        rec = {'profile': name, 'blocks': '+'.join(bl), 'dims': None}
        for r in refs:
            rec[f'mAP_{r}'] = float(pair_map(S, genes, POSd[r], pvalues=False).ap.mean())
        rec['SELECT_score'] = float(np.mean([rec[f'mAP_{r}'] for r in refs if r.endswith('SELECT')]))
        rec['REPORT_score'] = float(np.mean([rec[f'mAP_{r}'] for r in refs if r.endswith('REPORT')]))
        return rec

    rows = []
    for name, bl in sets.items():
        if any(f'X_{b}' not in z for b in bl):
            continue
        X = concat(bl)
        S = profile_cosine(X)
        rec = scored(S, POS, name, bl)
        rec['dims'] = int(X.shape[1])
        S2 = S.copy()
        S2[PARA] = -2.0
        for r in refs:
            if r.endswith('SELECT'):
                rec[f'mAP_{r}_noparalog'] = float(pair_map(S2, genes, POS[r] & ~PARA, pvalues=False).ap.mean())
        rec['SELECT_score_noparalog'] = float(np.mean([rec[f'mAP_{r}_noparalog'] for r in refs if r.endswith('SELECT')]))
        rows.append(rec)
    full = [b for b in _PROFILE_SETS['esm+dom+s3di+ops+img' if 'X_s3di' in z else 'esm+dom+ops'] if f'X_{b}' in z]
    if full:
        X = concat(full, np.random.default_rng(0).permutation(int(lib.sum())))
        rec = scored(profile_cosine(X), POS, 'control: concatenation, gene labels shuffled', full)
        rec['dims'] = int(X.shape[1])
        rows.append(rec)
    T = pd.DataFrame(rows)
    gg._write(T, root / 'profile_metrics.csv')
    return T


def run_gene_knowledge(settings: Mapping[str, Any], screen_blocks: Optional[Any] = None) -> Dict[str, Any]:
    """Fetch, embed, predict structure, search and assemble the blocks in order.

    The settings are written to ``<src>/gene_knowledge/settings/gene_knowledge_settings.csv``.

    :param settings: the ``gene_knowledge_*`` settings.
    :param screen_blocks: the gene graph ``graph.npz`` for the profile metrics.
    :returns: the block and structure summaries, and the profile metrics when references are set.
    """
    full = get_gene_knowledge_default_settings(settings)
    o = options(full)
    root = out_dir(o)
    root.mkdir(parents=True, exist_ok=True)
    gg.save_settings_file(full, root, 'gene_knowledge_settings')
    fetch_protein_annotations(full)
    embed_sequences(full)
    embed_text(full)
    predict_3di(full)
    result = build_knowledge_blocks(full)
    if o.get('profile_reference_dir'):
        result['profile_metrics'] = profile_metrics(full, screen_blocks)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run :func:`run_gene_knowledge` from a settings file.

    :param argv: the arguments; None reads :data:`sys.argv`.
    :returns: 0.
    """
    from .cli import load_settings_file

    parser = argparse.ArgumentParser(prog='python -m spacr.gene_knowledge', description=_DESCRIPTION)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--screen-blocks', default=None)
    args = parser.parse_args(argv)
    run_gene_knowledge(load_settings_file(args.settings), args.screen_blocks)
    return 0


if __name__ == '__main__':
    sys.exit(main())
