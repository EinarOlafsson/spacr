"""Verify an unchanged Methods export; do not approve or rewrite its science.

The recorded tutorial deliberately teaches review of an imperfect draft.
Independently check the limited facts its narration states, preserve the raw
export, and distinguish a declared generator seed from an inferred global one.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


def check_gene_family(rows):
    """Check the complete recorded BH family, not just a displayed top row."""
    if not rows or len({r['gene'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate gene identities')
    size = len(rows)
    ordered = sorted(enumerate(rows), key=lambda pair: float(pair[1]['p_value']))
    expected = [0.0] * size
    running = 1.0
    for rank in range(size, 0, -1):
        index, row = ordered[rank - 1]
        p = float(row['p_value'])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid probability')
        running = min(running, p * size / rank)
        expected[index] = running
    for row, q in zip(rows, expected):
        if (row['multiple_testing_method'] != 'fdr_bh'
                or int(row['tested_genes_in_family']) != size
                or not math.isclose(float(row['q_value']), q, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(float(row['adjusted_p_value']), q, rel_tol=1e-10, abs_tol=1e-12)
                or (row['significant'].lower() == 'true') != (q < float(row['alpha']))):
            raise ValueError('Saved BH family or significance disagrees with the independent check')
    alphas = {float(r['alpha']) for r in rows}
    if len(alphas) != 1:
        raise ValueError('Mixed significance thresholds')
    alpha = alphas.pop()
    return dict(n_genes=size, alpha=alpha, n_significant=sum(q < alpha for q in expected),
                min_q=min(expected), all_bh_values_checked=True)


def check_export(text, methods, results, digest):
    """Require both unchanged prose and the exact structured appendix."""
    prefix = methods.strip() + '\n\n' + results.strip() + '\n\n## Appendix: run digest\n\n```json\n'
    if not text.startswith(prefix) or not text.endswith('\n```\n'):
        raise ValueError('Export does not preserve the displayed sections and appendix')
    if json.loads(text[len(prefix):-5]) != digest:
        raise ValueError('Exported digest differs from the actual GUI digest')


def review_warnings(digest, manifest, family, results):
    """Name observed limitations without guessing undocumented historical state."""
    declared = manifest['seeds']['declared']
    global_seed_recorded = any(key in declared for key in ('random_seed', 'seed'))
    issues = []
    if not global_seed_recorded:
        issues.append('The journal declares the permutation seed, not a global run seed; the generic seed sentence needs review.')
    if not family['n_significant'] and digest['hits'] and 'strongest hits' in results:
        issues.append('The displayed ranked candidates are not significant at the stated threshold; calling them hits needs review.')
    if digest['spacr_version'] != manifest['env']['spacr']:
        issues.append('The exporter version differs from the recorded analysis version; do not report it as the analysis version.')
    return issues


def inspect_review(proof, export_path):
    """Verify this example's limited teaching facts without certifying its draft."""
    work = Path(proof['private_folder'])
    manifest_path = work / 'run_dir/manifest.json'
    gene_path = work / 'project/results/guide_permutation/results_gene.csv'
    manifest = json.loads(manifest_path.read_text())
    declared = manifest['seeds']['declared']
    digest = proof['digest']
    if digest['run']['seed_declared'] != declared:
        raise ValueError('Declared seeds were lost from the digest')
    with gene_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    family = check_gene_family(rows)
    stats = digest['statistics']
    if any(stats[key] != family[other] for key, other in
           [('n_genes_tested', 'n_genes'), ('alpha', 'alpha'), ('n_significant', 'n_significant')]):
        raise ValueError('Displayed gene summary disagrees with its complete saved family')
    by_gene = {r['gene']: r for r in rows}
    for hit in digest['hits']:
        row = by_gene[hit['gene']]
        for key, source in [('effect', 'coefficient'), ('p_value', 'p_value'), ('q_value', 'q_value')]:
            if not math.isclose(float(hit[key]), float(row[source]), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('A displayed candidate value differs from the saved result')
    export_path = Path(export_path)
    check_export(export_path.read_text(), proof['methods'], proof['results'], digest)
    global_seed_recorded = any(key in declared for key in ('random_seed', 'seed'))
    seed = declared['guide_permutation_seed']
    # Missing provenance is not evidence that a different seed was never used.
    # Do not replace every 42 with 0 or claim the two generators are identical.
    issues = review_warnings(digest, manifest, family, proof['results'])
    return dict(scope='Native export identity plus seed provenance and gene-family numerical checks only',
                family=family, declared_permutation_seed=seed,
                displayed_generic_seed=digest['run'].get('seed'),
                recorded_analysis_version=manifest['env']['spacr'],
                displayed_exporter_version=digest['spacr_version'],
                global_seed_recorded=global_seed_recorded,
                global_seed_actual_use_verified=False, issues=issues,
                original_draft_scientifically_approved=False,
                automated_numeric_check_is_semantic_approval=False,
                export_path=str(export_path), export_sha256=hashlib.sha256(export_path.read_bytes()).hexdigest(),
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [manifest_path, gene_path]},
                published=False)
