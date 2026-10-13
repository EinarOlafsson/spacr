from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / 'subcell-docs-37478293839'
read = lambda path: json.loads(path.read_text())
sha = lambda payload: hashlib.sha256(payload).hexdigest()
complete = read(root / 'publication-completion.json')
deployed = read(scratch / 'subcell-current-deployment-r1.json')
api_root = scratch / 'subcell-deployed-API-browser-r1'
guide_root = scratch / 'subcell-deployed-puncta-guides-r1'
api = read(api_root / 'acceptance.json')
guides = read(guide_root / 'acceptance.json')
assert complete['passed'] and deployed['passed'] and api['passed'] and guides['passed']
expected = 'e259c2d4ecfa34b8228a8e8eab4520ff06edcf04'
assert complete['actual_resolved_nightly_source'] == api['actual_resolved_nightly_source'] == guides['actual_resolved_nightly_source'] == expected
assert deployed['actual_channels']['channels']['nightly']['commit'] == expected
assert len(api['languages']) == len(guides['all_108_reviewed_message_readbacks']) == 9
assert sum(sum(panel['exact_reviewed_blocks_visible'] for panel in row['reviewed_panels']) for row in api['languages']) == 135
assert all(row['all_twelve_exact_reviewed_translations_visible'] for row in guides['all_108_reviewed_message_readbacks'])
visual = read(scratch / 'subcell-deployed-screenshot-visual-review-r1.json')
assert visual['actual_browser_screenshots_visually_reviewed'] and len(visual['screenshots']) == 6
out = Path('features/data/615_subcell_actual_publication_2026-10-06')
out.mkdir(exist_ok=False)
def archive(source, relative):
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix in ('.log', '.html', '.js'):
        target = target.with_name(target.name + '.gz')
        target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    else:
        shutil.copyfile(source, target)
    return target
for source in sorted(api_root.iterdir()): archive(source, Path('actual_API_browser') / source.name)
for source in sorted(guide_root.iterdir()): archive(source, Path('actual_puncta_guides') / source.name)
for source in sorted(root.glob('workflow-observation-*.json')): archive(source, Path('workflow') / source.name)
for name in ('publication-completion.json', 'publication-stage-journal.json', 'branch-artifact-inventory-r1.json'):
    archive(root / name, Path('workflow') / name)
for relative in ('channels.json', 'nightly/translation-compatibility.json', 'nightly/guide-translations.json', 'nightly/tutorials/index.html'):
    archive(root / 'assembled' / relative, Path('normal_assembled_reference') / relative)
for name in ('finish_subcell_publication_r1.py', 'verify_subcell_current_deployment_r1.py', 'verify_subcell_deployed_API_browser_r1.py', 'verify_subcell_deployed_puncta_guides_r1.py', 'subcell-publication-worker-r1.log', 'subcell-current-deployment-r1.json', 'subcell-publication-expectations-r1.json', 'subcell-hosted-source-contract-proof-r1.json', 'subcell-deployed-screenshot-visual-review-r1.json', 'subcell-docs-complete-workflow-r1.log'):
    archive(scratch / name, Path(name))
archive(Path(__file__), Path(Path(__file__).name))
receipt = {'item':615, 'passed':True, 'completed_documentation_workflow':37478293839,
    'actual_resolved_and_deployed_nightly_source':expected,
    'actual_normal_artifacts_assembled_with_standard_publisher':True,
    'all_ten_deployed_API_catalogs_and_fourteen_tutorial_catalogs_exact':True,
    'API_symbols':13182, 'all_135_reviewed_API_blocks_visible_in_nine_languages':True,
    'actual_screen_mapping_callback_on_its_own_API_page':True,
    'all_108_new_puncta_reference_messages_visible_in_nine_guides':True,
    'actual_six_browser_screenshots_visually_reviewed':True,
    'immutable_accepted_tutorial_media_commit':'0889ee14f5a7368a791df27c7331862a6c7741b4',
    'all_three_live_tutorial_media_roots_exact_to_prior_full_playback_and_decoded_visual_acceptance':True,
    'earlier_full_tutorial_acceptance_receipt':'features/data/615_installation_actual_nightly_2026-10-06.json',
    'no_application_catalog_or_tutorial_source_change_for_this_publication_receipt':True,
    'remaining_goal_scopes_not_claimed':['Official obtained-checkpoint Cell-DINO integration and GPU benchmark', 'Expert-labelled four-stain accuracy', 'Remaining Home source/CPU/CI/Qt acceptance', 'Missing calibrated low-light/manual masks, original OPS inputs and labelled QC/event data', 'Native-speaker or native Windows/macOS capture'],
    'artifacts':{str(path):{'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())} for path in sorted(out.rglob('*')) if path.is_file()}}
out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS actual e259 publication acceptance archived without changing application/catalog/tutorial source.',flush=True)
