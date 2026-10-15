"""Build only the reviewed tracked source selection; no HF client or network."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015')
OUT = Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
SOURCE = '35aeaa9ddc1f09cbd85fe75617a68df81adade0f'
OLD = Path('/mnt/wd4tb/scratch/n698-zerogpu-deployment-preparation-20261015/source-staging-manifest.json')

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])

paths = json.loads(OLD.read_text())['archive_paths']
entries = []
for record in git('ls-tree', '-r', '-l', '-z', SOURCE, '--', *paths).split(b'\0'):
    if not record:
        continue
    metadata, path = record.split(b'\t', 1)
    if path.decode() == 'spacr/resources/models/toxo_plaque_cyto_e25000_X1120_Y1120.CP_model':
        continue
    mode, kind, blob, size = metadata.split()
    assert kind == b'blob' and mode in (b'100644', b'100755')
    entries.append({'source_path': path.decode(), 'destination_path': path.decode(), 'git_blob': blob.decode(), 'bytes': int(size), 'mode': mode.decode()})
for src, dest in [('tools/web_mask_generator/space/README.md', 'README.md'), ('tools/web_mask_generator/space/requirements.txt', 'requirements.txt')]:
    row = next(x.copy() for x in entries if x['source_path'] == src)
    row['destination_path'] = dest
    entries.append(row)
entries.sort(key=lambda x: x['destination_path'])
assert len({x['destination_path'] for x in entries}) == len(entries)
mtime = int(git('show', '-s', '--format=%ct', SOURCE))
archive = OUT / 'spacr-mask-generator-source-35aeaa9ddc-no-weights.tar'
assert not archive.exists(), 'Never overwrite an earlier bundle'
proc = subprocess.Popen(['git', '-C', str(ROOT), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
try:
    with tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as tar:
        for row in entries:
            path = PurePosixPath(row['destination_path'])
            assert not path.is_absolute() and '..' not in path.parts
            proc.stdin.write((row['git_blob'] + '\n').encode()); proc.stdin.flush()
            header = proc.stdout.readline().split()
            assert header == [row['git_blob'].encode(), b'blob', str(row['bytes']).encode()]
            entry = tarfile.TarInfo(str(path)); entry.size = row['bytes']; entry.mode = int(row['mode'], 8) & 0o777
            entry.mtime = mtime; entry.uid = entry.gid = 0; entry.uname = entry.gname = ''
            tar.addfile(entry, proc.stdout)
            assert proc.stdout.read(1) == b'\n'
finally:
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0
with tarfile.open(archive, 'r') as tar:
    members = tar.getmembers()
    assert [(x.name, x.size) for x in members] == [(x['destination_path'], x['bytes']) for x in entries]
    assert all(x.isfile() for x in members)
    for member, row in zip(members, entries):
        h = hashlib.sha256()
        blob = hashlib.sha1(('blob ' + str(row['bytes']) + '\0').encode())
        with tar.extractfile(member) as handle:
            while chunk := handle.read(1024 * 1024):
                h.update(chunk); blob.update(chunk)
        assert blob.hexdigest() == row['git_blob']
        row['sha256'] = h.hexdigest()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()
manifest = {'source_commit': SOURCE, 'source_tree': git('rev-parse', SOURCE + '^{tree}').decode().strip(), 'files': entries, 'file_count': len(entries), 'bytes': sum(x['bytes'] for x in entries), 'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive), 'archive_paths': paths, 'exact_excluded_tracked_files': ['spacr/resources/models/toxo_plaque_cyto_e25000_X1120_Y1120.CP_model'], 'validation': 'Every destination is regular; exact archive inventory, Git blob and SHA256 verified by streaming original tracked blobs. No extraction or network.', 'excluded': 'Repository history, credentials, features evidence, tests, built docs, tutorials, actual weight files and untracked data.'}
(OUT / 'source-staging-35aeaa9ddc-no-weights.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({k: v for k, v in manifest.items() if k not in ('files', 'archive_paths')}, indent=2))
