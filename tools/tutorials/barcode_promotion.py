"""A Map Barcodes placeholder needs real recordings and all final media to close."""
import hashlib
import json
from pathlib import Path

from check_completed_matrix import check
from map_barcodes_data import REFERENCES, verify_counts
from stage_lesson import read

IDENTITY = '12_map_barcodes'


def require_recorded_map(stage, language, lesson):
    if stage is None:
        raise ValueError('Map Barcodes promotion needs actual recording and media evidence')
    stage = Path(stage).resolve()
    prefix = 'captions' if language in {'da', 'de', 'is', 'ko', 'nb', 'sv'} else 'lessons'
    records = [row for row in read(stage / 'catalog' / f'{prefix}_{language}.json')['lessons']
               if row['id'] == IDENTITY]
    if records != [lesson] or lesson.get('status') == 'coming_soon':
        raise ValueError('Promotion must match the exact recorded language catalog')
    english = read(stage / 'production' / IDENTITY / 'lesson.en.json')
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    matrix = check(stage, IDENTITY, stage.parent / 'tools/render_all_voices.py')
    if (matrix.get('passed') is not True or matrix.get('unique_final_tracks') != 50
            or len(matrix.get('browser_reports', [])) != 14
            or matrix.get('canonical_english_sha256') != canonical):
        raise ValueError('Map Barcodes requires all fifty voices and fourteen browser cases')
    capture = read(stage / 'captures/map_verified/scientific_acceptance.json')
    search = read(stage / 'captures/map_search_v2/barcode_search.json')
    if (capture.get('accepted') is not True or capture.get('app_source_modified') is not False
            or search.get('no_automatic_settings_change') is not True
            or search.get('applied_through_visible_button') is not True
            or search.get('application_functions_replaced') is not False
            or capture['api'].get('inputs_unchanged') is not True
            or capture['api'].get('gui_barcode_set_control_claimed') is not False
            or capture['api'].get('barcode_set') != ['column', 'grna']):
        raise ValueError('The actual GUI search and separately scoped API recording are required')
    references = stage / 'map_references/map_search_v2'
    paths = {role: references / (Path(name).stem + '_RC.csv' if role != 'column' else name)
             for role, name in REFERENCES.items()}
    source = stage / 'map_dataset'
    if not source.exists():
        source = stage / 'example_data/sequencing'
    gui = verify_counts(source / 'SRR33531217_paired', paths, 10000)
    api = verify_counts(stage / 'map_api_recorded_output',
                        {role: paths[role] for role in ('column', 'grna')}, 1000)
    if gui != capture['gui'] or api != capture['api']['api_two_barcodes']:
        raise ValueError('The recorded saved counts or their reference assignments changed')
    if ((gui['extracted_rows'], gui['mapped_reads'], gui['count_rows']) != (8611, 7657, 4099)
            or (api['extracted_rows'], api['mapped_reads'], api['count_rows']) != (869, 793, 535)):
        raise ValueError('The saved results do not support the narrated example')
    return matrix
