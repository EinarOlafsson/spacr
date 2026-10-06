from pathlib import Path
import ast
import gzip
import hashlib
import json

import requests

commit = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
out = Path('features/data/560_Cell_DINO_upstream_availability_2026-10-06_r2')
out.mkdir(exist_ok=False)
files = ('README.md', 'dinov2/hub/cell_dino/backbones.py', 'LICENSE_CELL_DINO_CODE', 'LICENSE_CELL_DINO_MODELS')
records = {}
texts = {}
for name in files:
    url = f'https://raw.githubusercontent.com/facebookresearch/dinov2/{commit}/{name}'
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    payload = response.content
    target = out / (name.replace('/', '--') + '.gz')
    target.write_bytes(gzip.compress(payload, mtime=0))
    assert gzip.decompress(target.read_bytes()) == payload
    texts[name] = payload.decode()
    records[name] = {'primary_URL': url, 'SHA256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
assert 'https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/' in texts['README.md']
assert 'pretrained_path' in texts['dinov2/hub/cell_dino/backbones.py']
module = ast.parse(texts['dinov2/hub/cell_dino/backbones.py'])
models = {}
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name.startswith('cell_dino_'):
        defaults = {arg.arg: ast.literal_eval(value) for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults) if value is not None and isinstance(value, ast.Constant)}
        calls = [call for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == '_make_cell_dino_model']
        constants = {keyword.arg: ast.literal_eval(keyword.value) for call in calls for keyword in call.keywords if keyword.arg and isinstance(keyword.value, ast.Constant)}
        models[node.name] = {'actual_upstream_keyword_defaults': defaults, 'actual_upstream_architecture_constant_arguments': constants}
assert models
form_url = 'https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/'
response = requests.get(form_url, timeout=90)
form_record = {'primary_URL': form_url, 'status': response.status_code, 'final_URL': response.url, 'SHA256': hashlib.sha256(response.content).hexdigest(), 'bytes': len(response.content)}
(out / 'official-download-request-form.html.gz').write_bytes(gzip.compress(response.content, mtime=0))
source = Path('spacr/embeddings.py')
current = source.read_text()
assert "Cell-DINO's weights are not published yet" in current
record = {'item': 560, 'upstream_primary_availability_and_architecture_audit_complete': True, 'official_upstream_commit': commit, 'source_files': records, 'upstream_model_factories': models, 'official_download_request_form': form_record, 'upstream_now_documents_Cell_DINO_checkpoint_download_request_and_local_path_loading': True, 'current_spacr_unsupported_reason_claiming_unpublished_weights_is_stale': True, 'application_source_SHA256': hashlib.sha256(source.read_bytes()).hexdigest(), 'no_form_submission_personal_information_license_acceptance_or_checkpoint_download': True, 'no_Cell_DINO_model_load_GPU_or_feature_support_claim': True, 'Human_Protein_Atlas_and_Cell_Painting_channels_must_follow_actual_model_not_arbitrary_RGB_stain_assignment': True, 'remaining_feature_source_implementation_owned_by_Home_GPU_and_docs_by_workstation': True}
copy = out / Path(__file__).name
copy.write_bytes(Path(__file__).read_bytes())
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
for filename in ('verify_cell_dino_upstream_availability_r1.py', 'cell-dino-upstream-availability-r1.log'):
    source = scratch / filename
    target = out / source.name
    target.write_bytes(source.read_bytes())
record['first_private_audit_failed_at_upstream_README_typo_license_filename_preserved'] = True
record['actual_weights_license_filename_verified_against_upstream_tree'] = 'LICENSE_CELL_DINO_MODELS'
all_artifacts = list(out.iterdir()) + list(Path('features/data/560_Cell_DINO_upstream_availability_2026-10-06').iterdir())
record['artifacts'] = {str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in sorted(all_artifacts)}
out.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
note = f'''\n2026-10-06 workstation Cell-DINO primary-source blocker correction: the current official facebookresearch/dinov2 checkpoint {commit} documents Cell-DINO model factories, pretrained_path/pretrained_url loading, and the Meta checkpoint request page https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/. The prior claim that Meta has not released weights is superseded: acquisition now depends on an obtained official checkpoint, not waiting for publication. Current spaCR _foundation_encoder still refuses with the stale unpublished-weights reason and references a README_CELL_DINO.md that is not the current documented route. Receipt 560_Cell_DINO_upstream_availability_2026-10-06.json preserves the pinned official README, model factories, licences and form response plus exact current application hash. No form is submitted, identity provided, licence accepted or checkpoint loaded by this read-only audit. The optional user checkpoint/path request is pending. Home: please reconcile Cell-DINO support/unsupported wording and explicit checkpoint/model channel configuration in your remaining source lane; do not assign RGB planes invented biological stain identities. Workstation owns the consequent normal API/runtime/docs/tutorial refresh and all model/GPU verification after a source checkpoint. The existing current application-source freeze remains unchanged for the active FAISS validation. No full foundation-backbone human-label comparison or new Cell-DINO support is claimed.\n'''
for path in (Path('features/future/560_foundation_model_embeddings.txt'), Path('features/325_two_sessions_one_repo_working_protocol.temp'), Path('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt')):
    assert path.is_file(), path
    with path.open('a') as stream:
        stream.write(note)
print('PASS pinned official Cell-DINO availability/factory evidence preserved; stale unpublished-weights source claim recorded for Home; no checkpoint obtained or feature support claimed.', flush=True)
