"""Prepare the N704 documentation-only owner packet from immutable Git objects."""
import ast
import collections
import datetime
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
import urllib.parse

REPO = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-graph-api-20261015')
SCRATCH = Path('/mnt/wd4tb/scratch/home-b-n704-current-promotion-20261015')
OLD = Path('/mnt/wd4tb/scratch/home-b-n704-20261014/promotion')
OUT = REPO / 'features/data/704_current_main_owner_packet_2026-10-15'
BASE = 'e0bedd1534d71708f65b60a2d9e24b6328d4ce9d'
SOURCE = '9263977fc9b977be880ca783d6716d2c6c4aeb6c'
COORDINATOR = '40c379e307e66b6855dabb5b157aac2bb1866fb1'


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def blobs(refs):
    raw = subprocess.run(['git', 'cat-file', '--batch'], cwd=REPO,
                         input=('\n'.join(refs) + '\n').encode(),
                         stdout=subprocess.PIPE, check=True).stdout
    result, offset = {}, 0
    for ref in refs:
        end = raw.index(b'\n', offset)
        fields = raw[offset:end].split()
        assert len(fields) == 3 and fields[1] == b'blob', (ref, fields)
        size = int(fields[2]); start = end + 1
        result[ref] = raw[start:start + size]
        assert raw[start + size:start + size + 1] == b'\n'
        offset = start + size + 1
    assert offset == len(raw)
    return result


def tree(rev):
    result = {}
    for line in git('ls-tree', '-r', rev).decode().splitlines():
        meta, path = line.split('\t', 1)
        mode, kind, oid = meta.split()
        result[path] = {'mode': mode, 'kind': kind, 'oid': oid}
    return result


def image_and_link_paths(path, data):
    text = data.decode()
    images = re.findall(r'(?:src|srcset)="([^"]+)"', text)
    images += re.findall(r'^\s*\.\. (?:\|[^|]+\| )?(?:image|figure)::\s+(\S+)', text, re.M)
    links = re.findall(r'href="([^"]+)"', text)
    links += re.findall(r'^\s*:target:\s+(\S+)', text, re.M)
    links += re.findall(r'<([^<>\s]+)>(?=`_)', text)
    result = {'images': set(), 'links': set()}
    for kind, urls in [('images', images), ('links', links)]:
        for url in urls:
            parsed = urllib.parse.urlsplit(html.unescape(url))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            dest = posixpath.normpath(posixpath.join(posixpath.dirname(path), urllib.parse.unquote(parsed.path)))
            assert not dest.startswith('../'), (path, dest)
            result[kind].add(dest)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_files = (OLD / 'files.txt').read_bytes()
    old_paths = original_files.decode().splitlines()
    old_manifest_bytes = (OLD / 'manifest.json').read_bytes()
    old = json.loads(old_manifest_bytes)
    assert len(old_paths) == len(set(old_paths)) == 678
    assert BASE == old['base_main_commit']
    assert sha((OLD / 'readme-docs-only.patch').read_bytes()) == old['patch_sha256']
    main_tree, source_tree, coordinator_tree = tree(BASE), tree(SOURCE), tree(COORDINATOR)
    readmes = [p for p in old_paths if p.endswith('.rst')]
    assert len(readmes) == 10
    readme_blobs = blobs([SOURCE + ':' + p for p in readmes])
    assets, local_links, by_readme = set(), set(), {}
    for path in readmes:
        refs = image_and_link_paths(path, readme_blobs[SOURCE + ':' + path])
        assets |= refs['images']; local_links |= refs['links']
        by_readme[path] = {kind: sorted(values) for kind, values in refs.items()}
    dependencies = sorted((assets | local_links) - set(old_paths) - set(main_tree))
    assert len(dependencies) == 60
    assert sum(p.startswith('docs/readme_text/') for p in dependencies) == 20
    assert sum(p.startswith('docs/readme_images/') and p.endswith('.svg') for p in dependencies) == 40
    assert len(assets) == 1747
    assert set(old['asset_sha256_equal_in_main_and_source']) <= assets
    paths = sorted(set(old_paths) | set(dependencies))
    assert len(paths) == 738
    assert all(source_tree[p] == coordinator_tree[p] for p in paths)
    assert not any(p.startswith('docs/readme_text/') for p in main_tree)
    for path in paths:
        assert source_tree[path]['kind'] == 'blob'
        assert source_tree[path]['mode'] == '100644'
    for path in assets - set(dependencies):
        assert source_tree[path] == main_tree[path], path
    assert (assets | local_links) <= set(paths) | set(main_tree)
    readable_destinations = {p for p in local_links if p.startswith('docs/readme_text/')}
    assert len(readable_destinations) == 670
    assert set(p for p in source_tree if p.startswith('docs/readme_text/')) == readable_destinations
    per_language = collections.Counter(p.split('/')[2] for p in readable_destinations)
    assert len(per_language) == 10 and set(per_language.values()) == {67}

    code_paths = ['tools/build_readme_images.py', 'packaging/generate_readme_visuals.py',
                  'tools/build_model_cards.py', 'tools/build_score_cards.py', 'setup.py',
                  'tests/test_pypi_readme_images.py']
    wanted = sorted(set(paths) | assets | set(code_paths))
    current = blobs([SOURCE + ':' + p for p in wanted])
    old_output = blobs([old['source_commit'] + ':' + p for p in old_paths])
    base_paths = sorted(set(paths) & set(main_tree))
    base_blobs = blobs([BASE + ':' + p for p in base_paths])
    changed_since_old = [p for p in old_paths if old_output[old['source_commit'] + ':' + p] != current[SOURCE + ':' + p]]
    assert len(changed_since_old) == 19
    historical_assets = old['asset_sha256_equal_in_main_and_source']
    for path, digest in historical_assets.items():
        assert sha(current[SOURCE + ':' + path]) == digest, path

    patch = git('diff', '--binary', '--full-index', BASE, SOURCE, '--', *paths)
    raw_patch = SCRATCH / 'readme-docs-only.patch'
    raw_patch.write_bytes(patch)
    env = dict(os.environ, GIT_INDEX_FILE=str(SCRATCH / 'candidate-current.index'))
    git('read-tree', BASE, env=env)
    git('apply', '--cached', '--check', str(raw_patch), env=env)
    git('apply', '--cached', str(raw_patch), env=env)
    candidate = git('write-tree', env=env).decode().strip()
    applied_paths = git('diff', '--name-only', BASE, candidate).decode().splitlines()
    assert applied_paths == paths
    promoted_tree = tree(candidate)
    assert all(promoted_tree[p] == source_tree[p] for p in paths)
    assert all(promoted_tree[p] == meta for p, meta in main_tree.items() if p not in paths)
    assert set(promoted_tree) - set(main_tree) <= set(paths)
    assert all(promoted_tree[p] == source_tree[p] for p in assets)

    historical_setup = blobs([old['source_commit'] + ':setup.py'])[old['source_commit'] + ':setup.py']
    def pypi_body(data):
        return ast.dump(next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == 'pypi_readme'), include_attributes=False)
    assert pypi_body(historical_setup) == pypi_body(current[SOURCE + ':setup.py'])
    assert b'https://github.com/EinarOlafsson/spacr/blob/main/' in current[SOURCE + ':setup.py']

    manifest = {
        'base_main_commit': BASE, 'source_commit': SOURCE, 'coordinator_commit': COORDINATOR,
        'coordinator_selected_paths_identical': True, 'promotion_tree': candidate,
        'promotion_file_count': len(paths),
        'categories': {'readmes': 10, 'readable_pages': 670, 'model_pages_including_index': 18, 'new_svg_assets': 40},
        'source_and_promoted_files': {p: {'sha256': sha(current[SOURCE + ':' + p]), **source_tree[p],
                                        'main_sha256': sha(base_blobs[BASE + ':' + p]) if p in main_tree else None,
                                        'main_git': main_tree.get(p)} for p in paths},
        'referenced_image_asset_count': len(assets),
        'referenced_assets': {p: {'sha256': sha(current[SOURCE + ':' + p]), **source_tree[p],
                                 'already_identical_on_main': p in main_tree and main_tree[p] == source_tree[p]} for p in sorted(assets)},
        'readme_references': by_readme,
        'added_dependencies': dependencies,
        'changed_historical_outputs': {p: {'old_sha256': sha(old_output[old['source_commit'] + ':' + p]),
                                         'current_sha256': sha(current[SOURCE + ':' + p])} for p in changed_since_old},
        'source_tool_and_test_pins': {p: {'sha256': sha(current[SOURCE + ':' + p]), **source_tree[p]} for p in code_paths},
        'patch_sha256': sha(patch), 'patch_bytes': len(patch),
        'historical': {'manifest_sha256': sha(old_manifest_bytes), 'files_sha256': sha(original_files),
                       'patch_sha256': old['patch_sha256'], 'source_commit': old['source_commit'],
                       'promotion_tree': old['promotion_tree']},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
    (OUT / 'readme-docs-only.patch.gz').write_bytes(gzip.compress(patch, mtime=0))
    (OUT / 'manifest.json.gz').write_bytes(gzip.compress(manifest_bytes, mtime=0))
    (OUT / 'files.txt').write_text('\n'.join(paths) + '\n')
    (OUT / 'historical-678-files.txt').write_bytes(original_files)
    (OUT / 'historical-678-manifest.json.gz').write_bytes(gzip.compress(old_manifest_bytes, mtime=0))
    for name in ['704_pypi_sanitizer_acceptance_2026-10-14', '704_readable_panel_implementation_2026-10-14']:
        data = (REPO / 'features/data' / name / 'receipt.json').read_bytes()
        (OUT / ('historical-' + name + '-receipt.json.gz')).write_bytes(gzip.compress(data, mtime=0))
    dump(OUT / 'added-dependencies.json', {
        'reason': 'The current generated READMEs add Hardware support and Models panels in every language. The historical 678-file list omitted their 20 readable pages and 40 phone/desktop SVGs, all absent from main.',
        'source_commit': SOURCE, 'base_main_commit': BASE,
        'files': {p: {'sha256': sha(current[SOURCE + ':' + p]), 'referenced_by': [r for r, refs in by_readme.items() if p in refs['links'] or p in refs['images']]} for p in dependencies},
    })
    receipt = {
        'item': 'N704', 'date_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Portable documentation-only promotion preparation; no branch/main/service publication and no current hosted or whole-item acceptance.',
        'base_main_commit': BASE, 'source_commit': SOURCE, 'coordinator_commit': COORDINATOR,
        'promotion_tree': candidate, 'promotion_file_count': 738,
        'historical_scope_preserved': 678, 'necessary_added_text_pages': 20, 'necessary_added_svg_assets': 40,
        'current_readable_panel_routes': dict(sorted(per_language.items())),
        'main_readable_text_page_count': 0,
        'referenced_assets': {'total': 1747, 'already_main_equal': 1707, 'new_source_equal_in_patch': 40},
        'checks': {
            'alternate_index_apply_check': 'PASS', 'alternate_index_apply': 'PASS',
            'exact_changed_paths': 'PASS: 738', 'all_promoted_blob_ids_modes_source_equal': 'PASS: 738',
            'all_unselected_main_blob_ids_modes_unchanged': 'PASS',
            'all_current_direct_local_readme_links_and_images_resolve_in_candidate': 'PASS (path existence only; no fragment/render/HTTP claim)',
            'all_current_readable_panel_destination_paths_present': 'PASS: 670, 67 per language',
            'all_referenced_image_bytes_match_source': 'PASS: 1747',
            'historical_1707_image_sha256_unchanged': 'PASS',
            'coordinator_selected_outputs_match_public_source': 'PASS: 738',
            'pypi_readme_function_ast_equals_historical_renderer_tested_function': 'PASS',
        },
        'historical_test_scope': 'Preserved prior 118 owning checks and both four-case real PyPI sanitizer receipts. Not rerun and not represented as current 67-panel renderer/hosted acceptance.',
        'not_performed': ['README/model generation', 'image rendering', 'pytest', 'GitHub/PyPI hosted readback', 'PyPI upload', 'main/remote/config mutation'],
        'remaining_owner_work': 'Review/apply the exact packet in the established private documentation-promotion lane, publish through that owner procedure, and verify actual GitHub/PyPI rendered links plus destination leading content in all ten locales. PyPI release remains deferred to the maintainer release lane.',
        'files': {},
    }
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            receipt['files'][path.name] = {'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size}
    dump(OUT / 'receipt.json', receipt)
    print(json.dumps({'output': str(OUT), 'files': len(paths), 'patch_bytes': len(patch),
                      'patch_gzip_bytes': (OUT / 'readme-docs-only.patch.gz').stat().st_size,
                      'patch_sha256': sha(patch), 'promotion_tree': candidate, 'checks': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
