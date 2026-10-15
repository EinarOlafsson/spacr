"""Apply one reviewed four-file packaging update; no other service mutation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import requests

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import CommitOperationAdd, HfApi, configure_http_backend, get_token
from huggingface_hub.hf_api import RepoFile

BASE = Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015')
TARGET = 'einarolafsson/spacr-mask-generator'
PARENT = '85b41cf7084471679f8c1076df8ee4f0b18b09b6'
PATCH = '2c2b6df05bd8f90930d32451d75b3015359836b4'
PUBLIC = 'b5b04e0772b58a7ee9dd401e53bfca971c4ff3b4'
EVENTS = BASE / 'source-archive-stage-b-update-events.jsonl'
RESULT = BASE / 'source-archive-stage-b-update-result.json'
ATTEMPT = BASE / 'source-archive-stage-b-update-attempt.lock'
class BoundedSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', (300, 90))
        return super().request(*args, **kwargs)
configure_http_backend(backend_factory=BoundedSession)

def git(path, revision):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + path])

def note(step, **fields):
    value = {'time_utc': datetime.now(timezone.utc).isoformat(), 'step': step, **fields}
    with EVENTS.open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True) + '\n'); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(value, sort_keys=True), flush=True)

def verify_tree(api, revision, expected):
    actual = {x.path: x for x in api.list_repo_tree(TARGET, repo_type='space', revision=revision, recursive=True) if isinstance(x, RepoFile)}
    assert set(actual) == set(expected) | {'.gitattributes'}
    assert actual['.gitattributes'].blob_id == 'f3f49149ace668fa22e292e81918bf0e2671206e'
    verified = []
    for path, row in expected.items():
        remote = actual[path]; assert remote.size == row['bytes']
        if remote.lfs:
            assert remote.lfs.size == row['bytes'] and remote.lfs.sha256 == row['sha256']
        else:
            assert remote.blob_id == row['git_blob']
        verified.append({'path': path, 'bytes': remote.size, 'blob': remote.blob_id, 'lfs_sha256': remote.lfs.sha256 if remote.lfs else None})
    return verified

state = {'target': TARGET, 'parent_commit': PARENT, 'patch_source_commit': PATCH, 'published_source_commit': PUBLIC, 'mutation_attempts': [], 'status': 'STARTED', 'authorization': 'Parent /root explicitly authorized ONE StageB four-config update from published2c2/b5 bytes atop exact85b41; preserve source archive/allother source/freeflags, no inference/settings/otheroperations.'}
try:
    descriptor = os.open(ATTEMPT, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, (datetime.now(timezone.utc).isoformat() + '\n').encode()); os.close(descriptor)
    manifest = json.loads((BASE / 'source-staging-35aeaa9ddc-no-weights.json').read_text())
    expected = {row['destination_path']: row.copy() for row in manifest['files']}; assert len(expected) == 1182
    payload = {}
    for source, destination in [('tools/web_mask_generator/space/README.md', 'README.md'), ('tools/web_mask_generator/space/requirements.txt', 'requirements.txt')]:
        raw = git(source, PATCH); assert raw == git(source, PUBLIC)
        payload[source] = raw; payload[destination] = raw
    assert len(payload) == 4
    # Preimage source remains the last published Git-config snapshot; add archive.
    for source, destination in [('tools/web_mask_generator/space/README.md', 'README.md'), ('tools/web_mask_generator/space/requirements.txt', 'requirements.txt')]:
        raw = git(source, '84615c61607759607c01429fe7d349cfb8242e9a')
        for path in (source, destination):
            expected[path].update({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()})
    artifact = json.loads((BASE / 'source-archive-stage-a-result.json').read_text())['artifact']
    expected[artifact['remote_path']] = {'bytes': artifact['bytes'], 'sha256': artifact['sha256'], 'git_blob': None}
    assert hashlib.sha256(payload['requirements.txt']).hexdigest() == '232252571abeceb9d49b7c01e4372959db78fdddd1464527358a7efeb316a2b0'
    assert hashlib.sha256(payload['README.md']).hexdigest() == '24040f48624071632ecf4f9447e2afba96def03d26080be686c06255ad0c3212'
    assert b'/resolve/85b41cf7084471679f8c1076df8ee4f0b18b09b6/' in payload['requirements.txt']
    assert payload['requirements.txt'].splitlines()[0].endswith(b'#sha256=a2068b97a8973a64fa13febbd81b87e933befb7684779be50d356fe969ce457f')
    state['four_files'] = {path: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)} for path, raw in payload.items()}
    note('published_four_file_bytes_verified', source=PUBLIC, files=state['four_files'])
    token = get_token(); assert token
    session = BoundedSession(); session.headers['Authorization'] = 'Bearer ' + token
    who = session.get('https://huggingface.co/api/whoami-v2', timeout=(10, 30), allow_redirects=False); who.raise_for_status(); who = who.json()
    assert who.get('name') == 'einarolafsson' and who.get('type') == 'user'
    assert who.get('isPro') is False and who.get('canPay') is False
    api = HfApi(token=token)
    info = api.space_info(TARGET); assert info.id == TARGET and info.private is False and info.sdk == 'gradio' and info.sha == PARENT
    runtime = api.get_space_runtime(TARGET); assert runtime.requested_hardware == 'zero-a10g' and runtime.hardware in (None, 'zero-a10g')
    variables = api.get_space_variables(TARGET)
    assert {key: val.value for key, val in variables.items()} == {'SPACR_WEB_ZERO_GPU': '1', 'SPACR_WEB_PROVISION_MODELS': '1'}
    verify_tree(api, PARENT, expected)
    state['fresh_guards'] = {'time_utc': datetime.now(timezone.utc).isoformat(), 'user': 'einarolafsson', 'parent': info.sha, 'stage': runtime.stage, 'requested': runtime.requested_hardware, 'current': runtime.hardware, 'flags_exact': True, 'all1182_sources_plus_archive_verified': True}
    note('fresh_guards_verified', **state['fresh_guards'])
    state['mutation_attempts'].append({'operation': 'atomic_four_file_update', 'parent_commit': PARENT, 'paths': sorted(payload)})
    note('atomic_update_requested', parent=PARENT, paths=sorted(payload))
    commit = api.create_commit(repo_id=TARGET, repo_type='space', parent_commit=PARENT, num_threads=1, commit_message='Install managed spaCR from verified immutable HTTP source archive', commit_description='Published spaCR source ' + PUBLIC + '; exact private patch ' + PATCH + '. Four packaging files only; all other source and free hardware/settings unchanged.', operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=raw) for path, raw in sorted(payload.items())])
    state['uploaded_commit'] = commit.oid; note('atomic_update_committed', commit=commit.oid)
    for path, raw in payload.items():
        expected[path].update({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()})
    verified = verify_tree(api, commit.oid, expected)
    (BASE / 'source-archive-stage-b-update-all-payload-readback.json').write_text(json.dumps({'space_commit': commit.oid, 'files': verified}, indent=2) + '\n')
    for path, raw in payload.items():
        response = requests.get('https://huggingface.co/spaces/' + TARGET + '/resolve/' + commit.oid + '/' + path, timeout=(10, 60)); response.raise_for_status(); assert response.content == raw
    final = api.space_info(TARGET); assert final.sha == commit.oid
    runtime = api.get_space_runtime(TARGET); assert runtime.requested_hardware == 'zero-a10g' and runtime.hardware in (None, 'zero-a10g')
    state.update({'status': 'FOUR_FILE_UPDATE_AND_FULL_TREE_VERIFIED_REBUILD_PENDING', 'verified_files': len(verified), 'changed_files': sorted(payload), 'unchanged_source_files': len(verified) - len(payload) - 1, 'archive_unchanged': True, 'actual_four_file_byte_readback': True, 'provider_gitattributes_unchanged': True, 'runtime': {'stage': runtime.stage, 'requested': runtime.requested_hardware, 'current': runtime.hardware}, 'other_mutations': False, 'inference_attempted': False})
    note('full_tree_and_four_file_bytes_verified', commit=commit.oid, files=len(verified), runtime=state['runtime'])
except BaseException as error:
    state['status'] = 'STOPPED_NO_RETRY'; state['error_type'] = type(error).__name__
    response = getattr(error, 'response', None)
    if response is not None: state['error_http_status'] = response.status_code
    note('stopped', error_type=type(error).__name__, mutation_attempts=len(state['mutation_attempts']))
    raise
finally:
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    RESULT.write_text(json.dumps(state, indent=2) + '\n')
