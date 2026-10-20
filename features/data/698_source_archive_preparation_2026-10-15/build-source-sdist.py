"""Build ordinary setuptools source distribution from verified public66 source."""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from importlib.metadata import version

S = Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
manifest = json.loads((S / 'source-staging-35aeaa9ddc-no-weights.json').read_text())
source = S / 'sdist-source-public66ac'
out = S / 'ordinary-sdist'
assert not source.exists() and not out.exists()
source.mkdir(mode=0o700); out.mkdir()
expected = {r['destination_path']: r for r in manifest['files']}
with tarfile.open(manifest['archive'], 'r') as archive:
    for member in archive:
        name = PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts
        row = expected[member.name]; assert member.size == row['bytes']
        path = source / member.name; path.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as incoming, path.open('xb') as outgoing:
            while chunk := incoming.read(1024 * 1024): outgoing.write(chunk)
        path.chmod(int(row['mode'], 8) & 0o777)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
assert len([p for p in source.rglob('*') if p.is_file()]) == 1182
command = [os.environ['SPACR_SDIST_PYTHON'], '-c', 'from setuptools.build_meta import build_sdist; import sys; print(build_sdist(sys.argv[1]))', str(out)]
completed = subprocess.run(command, cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
(S / 'ordinary-sdist-builder.log').write_bytes(completed.stdout)
assert completed.returncode == 0
archives = list(out.glob('*.tar.gz')); assert len(archives) == 1
artifact = archives[0]; rows = []; generated = []; original = set(); total = 0
with tarfile.open(artifact, 'r:gz') as archive:
    for member in archive:
        assert member.isdir() or member.isfile()
        name = PurePosixPath(member.name); assert not name.is_absolute() and '..' not in name.parts
        if member.isdir(): continue
        path = str(PurePosixPath(*name.parts[1:])); raw = archive.extractfile(member).read()
        digest = hashlib.sha256(raw).hexdigest(); total += len(raw)
        assert not path.lower().endswith(('.cp_model', '.pt', '.pth', '.onnx', '.safetensors'))
        assert '.git' not in name.parts
        if path in expected:
            assert digest == expected[path]['sha256'], path
            original.add(path); rows.append({'archive_path': member.name, 'source_path': path, 'bytes': len(raw), 'sha256': digest, 'original_git_blob': expected[path]['git_blob']})
        else:
            assert path == 'PKG-INFO' or path.startswith('spacr.egg-info/') or path == 'setup.cfg', path
            generated.append({'archive_path': member.name, 'bytes': len(raw), 'sha256': digest})
missing_python = [p for p in expected if p.startswith('spacr/') and p.endswith('.py') and p not in original]
assert not missing_python, missing_python
required = ['setup.py', 'pyproject.toml', 'README.rst', 'MANIFEST.in', 'LICENSE', 'packaging/linux/io.github.olafssonlab.spacr.desktop'] + [f'packaging/linux/icons/hicolor/{x}x{x}/apps/io.github.olafssonlab.spacr.png' for x in (16,32,48,64,128,256,512)]
assert all(p in original for p in required)
for path, row in expected.items():
    assert hashlib.sha256((source / path).read_bytes()).hexdigest() == row['sha256']
proof = {'source_commit':'35aeaa9ddc1f09cbd85fe75617a68df81adade0f', 'public_space_source_commit':'66ac681d3040aab1823aed0bf16b4ed5c89589f7', 'method':'Normal setuptools.build_meta.build_sdist with existing setuptools/wheel, private verified source copy; no package install/network/checkpoints/runtime model imports.', 'builder_versions':{name:version(name)for name in ('setuptools','wheel','pip')}, 'builder_exit':completed.returncode, 'archive':str(artifact), 'archive_bytes':artifact.stat().st_size, 'archive_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(), 'archive_uncompressed_file_bytes':total, 'original_files':rows, 'generated_metadata_files':generated, 'all_spacr_python_files_preserved':True, 'all_original_inputs_unchanged_after_build':True, 'excluded_by_normal_distribution_manifest':[p for p in expected if p not in original], 'required_setup_and_linux_data_files_present':required, 'weights_or_history_in_archive':False}
(S/'ordinary-sdist-inventory.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items()if k not in ('original_files','generated_metadata_files','excluded_by_normal_distribution_manifest','required_setup_and_linux_data_files_present')},indent=2))
