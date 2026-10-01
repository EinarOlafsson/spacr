"""Triage every English lesson against the user-walkthrough principles.

Moved into the repo on 2026-09-29 from the 09-22 authoring workspace. Its
"live" input used to be a saved 09-23 snapshot of the catalog that MAIN's
Pages serves (sha256 c395980a...), so every lesson rewritten after that day
stayed "published_rewrite: false" and lessons 71 and 82-85 (rewritten or
authored after the hard-coded list) stayed "pending". The live input is now
fetched from the deployed nightly Pages catalog, which is what the published
media candidates are built for; main's catalog is compared separately and
only reported, since it changes at the release merge.

    python tools/tutorials/audit_user_walkthroughs.py
"""
import argparse
import datetime
import urllib.request
import hashlib
import json
import re
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
NIGHTLY = 'https://einarolafsson.github.io/spacr/nightly/tutorials/catalog/lessons_en.json'
MAIN = 'https://einarolafsson.github.io/spacr/tutorials/catalog/lessons_en.json'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--live-catalog', default=NIGHTLY)
parser.add_argument('--main-catalog', default=MAIN)
args = parser.parse_args()


def fetch(url):
    return urllib.request.urlopen(url, timeout=60).read()

root = repo / 'tools/tutorials/lessons'
revised = {'33_plate_viewer', '78_spacr_screens', '79_module_inputs_outputs', '80_image_analysis_pathways', '81_sequencing_pathways', '70_explain_cv', '72_volcano_explorer', '73_parameter_sweep', '74_import_images', '75_regression_diagnostics', '77_embeddings', '66_outliers', '68_power_design', '48_hit_list', '56_lineage', '64_gate_editor', '65_feature_explorer', '67_experiment_design', '69_dose_response', '59_anndata_export', '60_pca', '61_tabulate', '63_small_multiples', '57_layer_viewer', '62_feature_dictionary', '49_methods_results', '58_graph_builder', '01_pypi_github', '02_conda_install', '03_pip_install', '04_platform_installers', '05_home', '06_api', '07_mask', '08_measure', '09_annotate', '10_classify_cv', '11_classify_ml', '15_image_umap', '12_map_barcodes', '13_regression', '14_make_masks', '16_activation', '17_timelapse', '18_motility', '19_train_cellpose', '20_cellpose_masks', '21_model_compare', '22_model_zoo', '23_agreement', '24_plaque', '25_recruitment', '29_report', '30_plate_queue', '31_external_masks', '32_align_stitch', '35_converter', '34_database', '36_import', '37_batch', '38_distributed_jobs', '39_classifier_evaluation', '40_run_history', '42_curate', '43_illumination', '44_data_manager', '45_project_browser', '46_napari_bridge', '47_barcode_qc', '50_run_compare', '51_control_charts', '52_pipeline_graph', '53_prediction_profiler', '54_qc_dashboard', '55_image_scatter', '26_invasion', '27_replication', '28_training_runs', '41_classify', '76_ops',
           # Rewritten 2026-09-25 on the real screen example (e1e6a23d7).
           '71_investigate_hit'}
patterns = {
    'authoring_or_test_details': r'\b(?:we (?:checked|verified|independently)|independently (?:checked|verified|reconciled)|checksums?|source hashes|acceptance|neutral (?:recording )?path|this recording (?:uses|reuses|verifies|shows)|no (?:AI|service|provider) (?:request|response)|not (?:executed|run|tested) (?:here|on)|legacy engine|validated (?:workflow|pipeline))\b',
    'negative_framing_to_review': r'\b(?:not (?:a|an|the|proof)|does not (?:prove|establish|certify)|do not (?:assume|treat|imply))\b',
    'retired_menu': r'\b(?:Help.{0,12}Demos|Demos menu)\b',
}
new_lessons = {'82_toxoplasma', '83_plasmodium', '84_candida', '85_host_pathogen'}
rows = []
live_bytes = fetch(args.live_catalog)
live_lessons = {x['id']: x for x in json.loads(live_bytes)['lessons']}
main_bytes = fetch(args.main_catalog)
main_lessons = {x['id']: x for x in json.loads(main_bytes)['lessons']}
prepared_lessons = {x['id']: x for x in json.loads((repo / 'docs/source/_extra/tutorials/catalog/lessons_en.json').read_text())['lessons']}

def matches(lesson, catalog):
    other = catalog.get(lesson['id'], {})
    return all(other.get(key) == value for key, value in lesson.items())

sources = {lesson['id']: lesson for lesson in json.loads((
    repo / 'docs/source/_extra/tutorials/catalog/lessons_en.json').read_text())['lessons']}
for path in sorted(root.glob('[0-9][0-9]_*.json')):
    lesson = json.loads(path.read_text())
    if 'id' in lesson:
        sources[lesson['id']] = lesson
for identity, lesson in sorted(sources.items()):
    identity = lesson['id']
    current = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    flags = []
    for field, text in [('prerequisite', lesson.get('prerequisite', ''))] + [
            (f'scenes[{i}].narration', scene['narration'])
            for i, scene in enumerate(lesson['scenes'])]:
        for reason, pattern in patterns.items():
            if re.search(pattern, text, re.I):
                flags.append(dict(field=field, reason=reason))
    stale = []
    for review_path in sorted((root / 'reviews').glob(identity + '.*.json')):
        # The all-languages Map Barcodes review record (read by
        # apply_map_reviews.py) is not a per-language translation review.
        if review_path.name in {'12_map_barcodes.reviewed.json'}:
            continue
        review = json.loads(review_path.read_text())
        if review.get('english_sha256') != current:
            stale.append(review.get('language', review_path.stem))
    rows.append(dict(lesson=identity, english_sha256=current,
                     scene_count=len(lesson['scenes']),
                     source_revised=identity in revised,
                     newly_authored=identity in new_lessons,
                     manual_review=(('newly authored' if identity in new_lessons else 'source rewritten')
                                    + ('; live narration aligned' if matches(lesson, live_lessons) else '; media publication pending'))
                                   if identity in revised | new_lessons else 'pending',
                     automated_review_flags=flags,
                     incompatible_translation_reviews=stale,
                     prepared_rewrite=identity in revised | new_lessons and matches(lesson, prepared_lessons),
                     published_rewrite=identity in revised | new_lessons and matches(lesson, live_lessons),
                     main_catalog_matches=matches(lesson, main_lessons)))
for row in rows:
    if row['lesson']=='59_anndata_export':
        row['current_workflow_followup']='Normal exporter and six native GUI exports verified; current recording replaces the older helper route.'
report = dict(schema=1, date=datetime.date.today().isoformat(),
              scope='Authoring triage against user walkthrough principles; automated flags are review prompts, not proof of a bad tutorial or publication blockers.',
              translation_policy='Register stale reviews; ready English may publish without translated audio.',
              required_sequence=['task and inputs', 'current module route', 'test data when available', 'key settings and effects', 'run', 'outputs and next step'],
              source_lessons=len(rows), manually_rewritten=sorted(revised),
              new_capture_required=['02_conda_install', '03_pip_install', '04_platform_installers', '17_timelapse', '19_train_cellpose', '76_ops'],
              fresh_capture_completed=['33_plate_viewer', '59_anndata_export', '77_embeddings', '18_motility', '22_model_zoo', *sorted(new_lessons)],
              newly_authored=sorted(new_lessons),
              remaining_existing_scripts=sorted(set(sources) - revised - new_lessons),
              live_catalog_url=args.live_catalog,
              live_catalog_sha256=hashlib.sha256(live_bytes).hexdigest(),
              main_catalog_url=args.main_catalog,
              main_catalog_sha256=hashlib.sha256(main_bytes).hexdigest(),
              main_catalog_differs_until_release_merge=sorted(row['lesson'] for row in rows if not row['main_catalog_matches']),
              manual_review_pending=sorted(row['lesson'] for row in rows if row['manual_review'] == 'pending'),
              narration_and_caption_refresh_required=sorted(row['lesson'] for row in rows if (row['source_revised'] or row['newly_authored']) and not row['published_rewrite']),
              rows=rows)
output = repo / 'tools/tutorials/evidence/2026-09-23-user-walkthrough-review.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(source_lessons=len(rows), manually_rewritten=len(revised),
                     unpublished=report['narration_and_caption_refresh_required'],
                     pending=report['manual_review_pending'],
                     lessons_flagged=sum(bool(row['automated_review_flags']) for row in rows),
                     incompatible_reviews=sum(len(row['incompatible_translation_reviews']) for row in rows))))
