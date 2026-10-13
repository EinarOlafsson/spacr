"""Builds receipt.json from the 2026-10-11 re-score folder (read-only inputs).
Run: python -I build_receipt.py   (inputs live outside the repository)."""
import csv, hashlib, json, collections
R = '/media/carruthers/mnt3/claude/gt_rescore_20261011'
OUT = __file__.rsplit('/', 1)[0] + '/receipt.json'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
S = json.load(open(f'{R}/scores/summary.json'))
M = json.load(open(f'{R}/manifest.json'))
by = {(x['source'], x['stem']): x for x in M['fields']}
rows = list(csv.DictReader(open(f'{R}/scores/per_field.csv')))
sources = collections.OrderedDict()
for r in rows:
    k = (r['family'], r['source'])
    s = sources.setdefault(k, dict(fields=set(), in_pool=set(), reviewed_empty=set(), paged=set()))
    s['fields'].add(r['field'])
    if r['in_pool'] == 'True': s['in_pool'].add(r['identity_stem'])
    if r['reviewed_empty'] == 'True': s['reviewed_empty'].add(r['field'])
    if r['review_basis'] == 'paged_no_ledger_row': s['paged'].add(r['field'])
counts = [dict(family=f, source=s, scored_fields=len(v['fields']), pooled_fields=len(v['in_pool']),
               reviewed_empty=len(v['reviewed_empty']), paged_no_ledger_row=len(v['paged']))
          for (f, s), v in sources.items()]
hashes = {}
for (fam, src), v in sources.items():
    lst = []
    for fld in sorted(v['fields']):
        m = by.get((src, fld))
        lst.append(dict(field=fld, **({k: m.get(k) for k in ('image_sha256', 'image_pixels_sha256', 'mask_sha256', 'mask_pixels_sha256')} if m else dict(manifest_record=False))))
    hashes[src] = lst
tabs = {}
for fam in ('pv', 'plaque'):
    d = S[fam]
    tabs[fam] = {k: (v if isinstance(v, list) else v) for k, v in d.items()}
receipt = dict(
    generated_by='features/data/model_rescore_curated_gt_2026-10-11/build_receipt.py',
    date='2026-10-11',
    request='Re-score the published PV (toxoplasma_pv_v1..v5) and plaque (toxoplasma_plaque_v1..v4) models on the curated ground truth; draft model cards and zoo notes only.',
    published=dict(hugging_face_upload=False, git_push=False, note='Model card drafts are in gt_rescore_20261011/cards/<repo>/README.draft.md'),
    inputs={n: sha(f'{R}/{n}') for n in ('manifest.json', 'scores/summary.md', 'scores/summary.json', 'scores/per_field.csv', 'scores/weights.json')},
    manifest=dict(curated_records=len(M['fields']), excluded_records=len(M['excluded']), counts_per_source=M['counts']),
    scored=dict(counts_per_source=counts, pool=dict(pv=dict(records=771, distinct_fields=655, reviewed_empty=113), plaque=dict(records=810, distinct_fields=562, reviewed_empty=67))),
    decisions=[
        'The 98 round-3 PV fields with no ledger row are INCLUDED, labelled review_basis = paged_no_ledger_row; 39 of them are empty masks; tables are given with and without them.',
        'The 3 plaque literature fields marked done and Discard (with frozen copies) are EXCLUDED.',
        'The 11 Gel Doc set-aside wells (6 GRA8 Plate1 duplicates of ReimagedPlate1, 5 Dia_Plate18) are EXCLUDED.',
        'PV v5 is MARKED AS LEAKED: it trained or validated on 29 of 39 curated PV literature hold-out fields and 3 of 11 anchor wells; it is scored only on fields it never saw.',
        'Defaults do not change: PV v4 and plaque v4.',
        'Round-3 PV: the later toxoplasma_pv_r7 mask is used for 335 of 373 ledger-done fields; 57 stale frozen copies in the 263-set and 1 discarded copy in the 50-set are dropped.',
        'Pool has one record per physical field; a pooled field is seen by a model if any copy was seen.'],
    seen_counts_263_set=dict(plaque_v1=54, plaque_v2=178, plaque_v3=137, plaque_v4=0,
                             note='GRA8 Plate1 (hold-out of plaque v3) is the same physical plate as ReimagedPlate1 (training data of plaque v4).'),
    tables=tabs,
    verdicts=S['verdicts'],
    caveats=[
        'Small literature n: plaque 20 fields; PV 39 fields, 10 unseen by v5; PV common subset without paged fields 65; plaque common without v3-seeded truth 81.',
        'Gel Doc model_review truth was seeded with plaque v3 (57 fields accepted unchanged), which favours v3; pmc_staged drafts came from the plaque round-3 lineage.',
        'PV v1 and plaque v1 splits were reconstructed from code and seeds, not saved lists.',
        'Strict hold-out removes exact field matches only; no_group columns show plate, acquisition and paper sharing.',
        'Pooled strict hold-out has different n per model and does not rank models; use the common subsets.',
        'Defaults settings only (diameter 30, flow 0.4, cell probability 0).',
        'Source rows are subsets of the earlier 263- and 50-field sets and are not comparable with the 2026-10-09 and 2026-10-10 numbers.',
        'pmc_staged curation may still be in progress (ledger last written 2026-10-06).'],
    file_hashes_from_manifest=hashes)
json.dump(receipt, open(OUT, 'w'), indent=1, default=str)
print(OUT, len(json.dumps(receipt)))
