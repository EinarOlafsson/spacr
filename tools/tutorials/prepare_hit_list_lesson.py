"""Package and verify the disclosed companion demonstration, not a route fix."""
import json
from pathlib import Path
import zipfile
from build_evaluation_example import sha
from hit_list_evidence import expected_hits, filtered, check_rows, read_rows
from hit_list_exports import check_markdown, check_html
from stage_lesson import DEFAULT_STAGE, REPO, read, write


def prepare():
    capture = DEFAULT_STAGE / 'captures/hit_list_readable_companion_v2'
    proof = read(capture / 'hit-list-workflow.json')
    companion = proof.get('companion', {})
    if not (proof['accepted'] and proof['all_originals_unchanged']
            and proof['all_private_inputs_unchanged'] and proof['active_jobs_after_close'] == 0
            and companion.get('explicit_external_launcher') is True
            and companion.get('native_shortcut_fixed') is False
            and companion.get('app_widget_modified') is False
            and proof['native_fold_state']['hits_visible'] is False):
        raise ValueError('Require the successful explicitly disclosed companion, not a repaired native route')
    root = Path(__file__).parent
    if sha(root / 'hit_list_companion.py') != companion['launcher_sha256']:
        raise ValueError('Recorded companion differs from the downloadable launcher')
    for name, digest in proof['source_files'].items():
        if sha(name) != digest: raise ValueError('Original Regression output changed')
    data = Path(proof['private_folder']) / 'results'
    contents = {}
    for name, digest in proof['private_input_hashes'].items():
        if sha(data / name) != digest: raise ValueError('Private recorded input changed')
        contents['results/' + name] = (data / name).read_bytes()
    rows = expected_hits(data)
    if len(rows) != 325 or min(r['q_value'] for r in rows) <= .05:
        raise ValueError('Expected the recorded nonsignificant gene family')
    chosen = filtered(rows, min_guides=2, min_agreement=1)
    if len(chosen) != 57: raise ValueError('Recorded candidate subset changed')
    for record in proof['exports'].values():
        if sha(record['path']) != record['sha256']: raise ValueError('A native export changed')
    values = dict(csv=check_rows(read_rows(proof['exports']['csv']['path']), chosen),
        markdown=check_markdown(Path(proof['exports']['md']['path']).read_text(), chosen),
        html=check_html(Path(proof['exports']['html']['path']).read_text(), chosen))
    for name in ('hit_list_companion.py', 'hit_list_README.txt'):
        contents['README.txt' if name.endswith('.txt') else name] = (root / name).read_bytes()
    contents['source_manifest.json'] = (json.dumps(dict(original_sha256=proof['private_input_hashes'],
        original_results=True, companion_launcher_sha256=companion['launcher_sha256'],
        recorded_export_filters=proof['export_filters'], native_shortcut_fixed=False), indent=2)+'\n').encode()
    target = REPO / 'docs/source/_extra/tutorials/examples/Hit_List_real_results_companion.zip'
    if target.exists(): raise FileExistsError('Preserve existing tutorial bundle')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data_bytes in contents.items():
            info = zipfile.ZipInfo(name, date_time=(2026,9,11,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info, data_bytes)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any(archive.read(name) != data_bytes for name, data_bytes in contents.items()):
            raise ValueError('Download differs from verified source files')
    result = dict(accepted=True, scope='Unmodified Hit List widget in an explicit EXTERNAL companion; native shortcut remains broken',
        workflow=proof, export_values_checked=values, download_sha256=sha(target),
        native_shortcut_fixed=False, application_files_modified=False, published=False)
    write(capture / 'scientific_acceptance.json', result)
    write(REPO / 'tools/tutorials/evidence/2026-09-11_hit_list_companion_verification.json', result)
    print(target, sha(target)); print(values)


if __name__ == '__main__': prepare()
