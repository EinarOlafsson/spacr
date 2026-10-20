"""One authorized free-only creation and finite source upload; no inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import requests

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import HfApi, configure_http_backend, get_token
from huggingface_hub.hf_api import RepoFile

BASE = Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015')
TARGET = 'einarolafsson/spacr-mask-generator'
SOURCE = '35aeaa9ddc1f09cbd85fe75617a68df81adade0f'
MANIFEST = BASE / 'source-staging-35aeaa9ddc-no-weights.json'
AUTH = BASE / 'root-free-deployment-authorization.json'
STAGE = BASE / 'upload-source-35aeaa-no-weights'
EVENTS = BASE / 'deployment-retry-events.jsonl'
RESULT = BASE / 'deployment-retry-result.json'
ATTEMPT = BASE / 'deployment-upload-retry-attempt.lock'

class BoundedSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', (300, 90))
        return super().request(*args, **kwargs)

configure_http_backend(backend_factory=BoundedSession)

def note(step, **values):
    row = {'time_utc': datetime.now(timezone.utc).isoformat(), 'step': step, **values}
    with EVENTS.open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n'); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()

state = {'target': TARGET, 'source_commit': SOURCE, 'mutation_attempts': [], 'status': 'UPLOAD_ONLY_RETRY_STARTED', 'authorization': 'Parent /root explicit one justified upload-only retry after retained timeout; same payload/parent, no create/settings/fallback; 300s socket-write and90s read budget.'}
try:
    descriptor = os.open(ATTEMPT, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, (datetime.now(timezone.utc).isoformat() + '\n').encode()); os.close(descriptor)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['source_commit'] == SOURCE and manifest['file_count'] == 1182
    assert manifest['archive_sha256'] == '15c7456fa460716cfcf4e38d9271ff391fbfcf23abfef12ce798fb61a2f26309'
    assert sha(Path(manifest['archive'])) == manifest['archive_sha256']
    expected = {row['destination_path']: row for row in manifest['files']}
    assert len(expected) == 1182
    assert not STAGE.is_symlink() and STAGE.is_dir()
    assert {str(p.relative_to(STAGE)) for p in STAGE.rglob('*') if p.is_file()} == set(expected)
    assert all(not p.is_symlink() for p in STAGE.rglob('*'))
    assert all(sha(STAGE / path) == row['sha256'] for path, row in expected.items())
    assert 'spacr/resources/models/toxo_plaque_cyto_e25000_X1120_Y1120.CP_model' not in expected
    note('same_payload_verified', files=1182, source=SOURCE, archive_sha256=manifest['archive_sha256'])
    token = get_token(); assert token
    session = BoundedSession(); session.headers['Authorization'] = 'Bearer ' + token
    api = HfApi(token=token)
    response = session.get('https://huggingface.co/api/whoami-v2', timeout=(10, 30), allow_redirects=False)
    response.raise_for_status(); who = response.json()
    assert who.get('name') == 'einarolafsson' and who.get('type') == 'user'
    assert who.get('isPro') is False and who.get('canPay') is False
    info = api.space_info(TARGET)
    assert info.id == TARGET and info.private is False and info.sdk == 'gradio'
    assert info.sha == 'a020b17c09b277739e56f0be0ee4de63c5e454f9'
    initial = {item.path: item for item in api.list_repo_tree(TARGET, repo_type='space', revision=info.sha, recursive=True) if isinstance(item, RepoFile)}
    assert set(initial) == {'.gitattributes', 'README.md'}
    assert initial['README.md'].blob_id == '54e40d5683ec2f6b94341e8c401e1953c0774183' and initial['README.md'].size == 266
    assert initial['.gitattributes'].blob_id == 'a6344aac8c09253b3b630fb776ae94478aa0275b' and initial['.gitattributes'].size == 1519
    runtime = api.get_space_runtime(TARGET)
    assert runtime.requested_hardware == 'zero-a10g' and runtime.hardware in (None, 'zero-a10g')
    variables = {'SPACR_WEB_ZERO_GPU': '1', 'SPACR_WEB_PROVISION_MODELS': '1'}
    actual_variables = api.get_space_variables(TARGET)
    assert {key: value.value for key, value in actual_variables.items()} == variables
    state['fresh_guard'] = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'effective_user': 'einarolafsson', 'parent_commit': info.sha, 'current': runtime.hardware, 'requested': runtime.requested_hardware, 'stage': runtime.stage, 'initial_two_files_unchanged': True, 'public_variables_exact': True}
    note('fresh_upload_only_guards_passed', **state['fresh_guard'])
    state['mutation_attempts'].append({'operation': 'upload_source_once', 'files': 1182, 'parent_commit': info.sha})
    note('upload_started', files=1182, parent_commit=info.sha)
    uploaded = api.upload_folder(repo_id=TARGET, repo_type='space', folder_path=STAGE, parent_commit=info.sha, commit_message='Deploy reviewed spaCR mask generator source ' + SOURCE[:12], commit_description='Source ' + SOURCE + '; verified payload inventory SHA256 ' + sha(MANIFEST) + '; exact legacy weight excluded. No user data, checkpoints or Git history uploaded.')
    state['uploaded_commit'] = uploaded.oid; note('upload_committed', commit=uploaded.oid)
    remote = {item.path: item for item in api.list_repo_tree(TARGET, repo_type='space', revision=uploaded.oid, recursive=True) if isinstance(item, RepoFile)}
    assert set(expected).issubset(remote)
    assert set(remote) - set(expected) <= {'.gitattributes'}
    readback = []
    for path, row in expected.items():
        item = remote[path]; assert item.size == row['bytes']
        if item.lfs:
            assert item.lfs.sha256 == row['sha256'] and item.lfs.size == row['bytes']; method = 'Hub LFS SHA256 and byte count'
        else:
            assert item.blob_id == row['git_blob']; method = 'Hub Git blob and byte count'
        readback.append({'path': path, 'bytes': item.size, 'blob': item.blob_id, 'lfs_sha256': item.lfs.sha256 if item.lfs else None, 'method': method})
    state['remote_files'] = len(remote); state['verified_payload_files'] = len(readback); state['provider_added_files'] = sorted(set(remote) - set(expected))
    (BASE / 'deployment-retry-all-payload-readback.json').write_text(json.dumps({'space_commit': uploaded.oid, 'source_commit': SOURCE, 'files': readback}, indent=2) + '\n')
    controls = ['README.md', 'requirements.txt', 'tools/web_mask_generator/space_app.py', 'tools/web_mask_generator/provision.py', 'tools/web_mask_generator/space/models.json', 'tools/web_mask_generator/app.py', 'tools/web_mask_generator/service.py', 'spacr/settings.py']
    checked = []
    for path in controls:
        # Read actual public bytes at immutable uploaded revision, not a local cache.
        response = requests.get('https://huggingface.co/spaces/' + TARGET + '/resolve/' + uploaded.oid + '/' + path, timeout=(10, 60))
        response.raise_for_status(); assert len(response.content) == expected[path]['bytes']
        assert hashlib.sha256(response.content).hexdigest() == expected[path]['sha256']; checked.append(path)
    runtime = api.get_space_runtime(TARGET)
    assert runtime.requested_hardware == 'zero-a10g' and runtime.hardware in (None, 'zero-a10g')
    final = api.space_info(TARGET); assert final.sha == uploaded.oid
    state.update({'status': 'SOURCE_UPLOADED_AND_VERIFIED_BUILD_ACCEPTANCE_PENDING', 'final_runtime': {'stage': runtime.stage, 'current': runtime.hardware, 'requested': runtime.requested_hardware}, 'actual_control_byte_readback': checked, 'inference_attempted': False, 'other_spaces_mutated': False})
    note('payload_readback_verified', files=len(readback), control_files=len(checked), commit=uploaded.oid, runtime=state['final_runtime'])
except BaseException as exc:
    state['status'] = 'STOPPED_NO_RETRY'; state['error_type'] = type(exc).__name__
    response = getattr(exc, 'response', None)
    if response is not None: state['error_http_status'] = response.status_code
    note('stopped', error_type=state['error_type'], http_status=state.get('error_http_status'), mutation_attempts=len(state['mutation_attempts']))
    raise
finally:
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    RESULT.write_text(json.dumps(state, indent=2) + '\n')
