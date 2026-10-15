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

REPO = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015')
SCRATCH = Path('/mnt/wd4tb/scratch/home-b-n704-after-website-packet-20261015')
OLD = Path('/mnt/wd4tb/scratch/home-b-n704-20261014/promotion')
OUT = REPO / 'features/data/704_after_website_main_owner_packet_2026-10-15'
BASE = 'e0bedd1534d71708f65b60a2d9e24b6328d4ce9d'
SOURCE = '2d5dd8788d542e9d98929d90a59c69235bfaee65'
COORDINATOR = '94341133a288771230b67570f945ebfeb1ea6945'


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
    prior_prefix = 'features/data/704_current_main_owner_packet_2026-10-15/'
    historical_paths = git('ls-tree', '-r', '--name-only', SOURCE, '--', prior_prefix).decode().splitlines()
    history = blobs([SOURCE + ':' + p for p in historical_paths])
    for p in historical_paths:
        target = OUT / 'history' / 'prior-738' / p.removeprefix(prior_prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(history[SOURCE + ':' + p])
    prior = OUT / 'history' / 'prior-738'
    old_manifest_bytes = gzip.decompress((prior / 'manifest.json.gz').read_bytes())
    old = json.loads(old_manifest_bytes)
    prior_receipt = json.loads((prior / 'receipt.json').read_text())
    for name, expected in prior_receipt['files'].items():
        data = (prior / name).read_bytes()
        assert sha(data) == expected['sha256'] and len(data) == expected['bytes'], name
    assert sha(gzip.decompress((prior / 'readme-docs-only.patch.gz').read_bytes())) == old['patch_sha256']
    paths = (prior / 'files.txt').read_text().splitlines()
    assert len(paths) == len(set(paths)) == 738
    assert paths == sorted(old['source_and_promoted_files'])
    assert git('rev-parse', 'origin/main').decode().strip() == BASE
    assert git('rev-parse', 'origin/nightly').decode().strip() == SOURCE
    main_tree, source_tree, coordinator_tree = tree(BASE), tree(SOURCE), tree(COORDINATOR)
    readmes = [p for p in paths if p.endswith('.rst')]
    assert len(readmes) == 10
    current_readmes = blobs([SOURCE + ':' + p for p in readmes])
    assets, local_links, by_readme = set(), set(), {}
    for p in readmes:
        refs = image_and_link_paths(p, current_readmes[SOURCE + ':' + p])
        assets |= refs['images']; local_links |= refs['links']
        by_readme[p] = {kind: sorted(values) for kind, values in refs.items()}
    assert len(assets) == 1747 and assets == set(old['referenced_assets'])
    readable = {p for p in local_links if p.startswith('docs/readme_text/')}
    assert len(readable) == 670
    assert set(p for p in source_tree if p.startswith('docs/readme_text/')) == readable
    by_language = collections.Counter(p.split('/')[2] for p in readable)
    assert len(by_language) == 10 and set(by_language.values()) == {67}
    assert (assets | local_links) <= set(paths) | set(main_tree)
    assert all(source_tree[p] == coordinator_tree[p] for p in paths)
    code_paths = list(old['source_tool_and_test_pins'])
    wanted = sorted(set(paths) | assets | set(code_paths))
    current = blobs([SOURCE + ':' + p for p in wanted])
    previous = blobs([old['source_commit'] + ':' + p for p in paths])
    changed = [p for p in paths if previous[old['source_commit'] + ':' + p] != current[SOURCE + ':' + p]]
    assert changed == readmes, changed
    for p in readmes:
        before, after = previous[old['source_commit'] + ':' + p], current[SOURCE + ':' + p]
        line = after.decode().splitlines()[2]
        assert 'https://einarolafsson.github.io/projects/spacr/' in line
        assert after.replace((line + '\n\n').encode(), b'', 1) == before, p
    for p in assets:
        assert sha(current[SOURCE + ':' + p]) == old['referenced_assets'][p]['sha256'], p
    base_paths = sorted(set(paths) & set(main_tree))
    base_blobs = blobs([BASE + ':' + p for p in base_paths])
    patch = git('diff', '--binary', '--full-index', BASE, SOURCE, '--', *paths)
    raw_patch = SCRATCH / 'readme-docs-only.patch'
    raw_patch.write_bytes(patch)
    env = dict(os.environ, GIT_INDEX_FILE=str(SCRATCH / 'candidate-after-website.index'))
    git('read-tree', BASE, env=env)
    git('apply', '--cached', '--check', str(raw_patch), env=env)
    git('apply', '--cached', str(raw_patch), env=env)
    candidate = git('write-tree', env=env).decode().strip()
    assert git('diff', '--name-only', BASE, candidate).decode().splitlines() == paths
    candidate_tree = tree(candidate)
    assert all(candidate_tree[p] == source_tree[p] for p in paths)
    assert all(candidate_tree[p] == meta for p, meta in main_tree.items() if p not in paths)
    assert set(candidate_tree) - set(main_tree) <= set(paths)
    assert all(candidate_tree[p] == source_tree[p] for p in assets)
    assert all(candidate_tree[p]['kind'] == 'blob' for p in local_links | assets)
    prior_setup = blobs([old['source_commit'] + ':setup.py'])[old['source_commit'] + ':setup.py']
    def pypi_body(data):
        return ast.dump(next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == 'pypi_readme'), include_attributes=False)
    assert pypi_body(prior_setup) == pypi_body(current[SOURCE + ':setup.py'])
    manifest = {
        'base_main_commit': BASE, 'source_commit': SOURCE, 'packet_preparation_head': COORDINATOR,
        'promotion_tree': candidate, 'promotion_file_count': len(paths),
        'categories': old['categories'],
        'source_and_promoted_files': {p: {'sha256': sha(current[SOURCE + ':' + p]), **source_tree[p], 'main_sha256': sha(base_blobs[BASE + ':' + p]) if p in main_tree else None, 'main_git': main_tree.get(p)} for p in paths},
        'referenced_image_asset_count': len(assets),
        'referenced_assets': {p: {'sha256': sha(current[SOURCE + ':' + p]), **source_tree[p], 'already_identical_on_main': p in main_tree and main_tree[p] == source_tree[p]} for p in sorted(assets)},
        'readme_references': by_readme,
        'changed_since_prior_738_packet': {p: {'old_sha256': sha(previous[old['source_commit'] + ':' + p]), 'current_sha256': sha(current[SOURCE + ':' + p]), 'only_change': 'One previously reviewed plain Website link plus blank line after existing top badge row'} for p in changed},
        'source_tool_and_test_pins': {p: {'sha256': sha(current[SOURCE + ':' + p]), **source_tree[p]} for p in code_paths},
        'patch_sha256': sha(patch), 'patch_bytes': len(patch),
        'historical': {'source_commit': old['source_commit'], 'manifest_sha256': sha(old_manifest_bytes), 'receipt_sha256': sha((prior / 'receipt.json').read_bytes()), 'patch_sha256': old['patch_sha256'], 'promotion_tree': old['promotion_tree'], 'preserved_files': {p.removeprefix(prior_prefix): sha(history[SOURCE + ':' + p]) for p in historical_paths}},
    }
    (OUT / 'manifest.json.gz').write_bytes(gzip.compress((json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode(), mtime=0))
    (OUT / 'readme-docs-only.patch.gz').write_bytes(gzip.compress(patch, mtime=0))
    (OUT / 'files.txt').write_bytes((prior / 'files.txt').read_bytes())
    (OUT / 'historical-678-files.txt').write_bytes((prior / 'historical-678-files.txt').read_bytes())
    (OUT / 'verify_packet.py').write_bytes((prior / 'verify_packet.py').read_bytes())
    (OUT / 'prepare_packet.py').write_bytes(Path(__file__).read_bytes())
    instructions = f'''N704 README MAIN OWNER PACKET — AFTER WEBSITE LINKS
==================================================
Unapplied documentation-only refresh. No main, remote or service changed.
Target main (freshly fetched): {BASE}
Published source nightly: {SOURCE}
Verified isolated candidate tree: {candidate}
Uncompressed patch SHA256: {sha(patch)}

Exactly the same738 paths as the retained prior packet: ten READMEs,
670 readable pages (67 per language),18 model pages and40 SVG assets.
Only the ten README postimages changed since that packet: the reviewed plain
Website link on line3. All1747 referenced image assets remain byte-identical.
All738 postimages/modes equal current published source; every unselected main
blob/mode stays unchanged. All670 readable routes and direct local references
exist in the alternate-index candidate (no browser/fragment/HTTP claim).
Complete prior packet and receipt are retained unchanged in history/prior-738.

Owner application: use only your own clean private main-promotion worktree at
the exact target main pin. If main advanced, reconcile a new packet; do not
blindly apply, hand-merge generated output or broadly cherry-pick source.
Decompress readme-docs-only.patch.gz into WD scratch, verify its SHA256 above,
then git apply --check and git apply --index there. Use the unchanged portable
verifier before committing:

  python /absolute/packet/verify_packet.py --repo /your/private/worktree --candidate $(git write-tree)

The owner retains actual main publication and GitHub/PyPI rendering/readback.
No generator, renderer, pytest, model, main update, upload or deployment ran
for this refresh. The four real PyPI renderer checks on current ten READMEs
are already recorded separately in687_readme_homepage_2026-10-15; no replay.
This packet does not promote pyproject metadata, application source, tools,
workflows, tutorials, runtime/API catalogs, or any unrelated generated files.
'''
    (OUT / 'OWNER_INSTRUCTIONS.txt').write_text(instructions)
    receipt = {
        'item':'N704', 'status':'PASS_UNAPPLIED_OWNER_PACKET', 'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Exact738 documentation-only main candidate refresh; no whole-item/hosted/publication claim',
        'base_main_commit':BASE, 'fresh_remote_tracking_main':git('rev-parse','origin/main').decode().strip(),
        'source_commit':SOURCE, 'promotion_tree':candidate, 'promotion_file_count':738,
        'changed_since_prior_738_packet':changed, 'prior_packet_file_count':len(historical_paths),
        'referenced_assets':{'total':1747,'unchanged_since_prior':1747,'already_main_equal':sum(candidate_tree[p] == main_tree.get(p) for p in assets)},
        'current_readable_panel_routes':dict(sorted(by_language.items())),
        'checks':{'alternate_index_apply_check':'PASS','alternate_index_apply':'PASS','all738_postimages_modes_source_equal':'PASS','all_unselected_main_blobs_modes_unchanged':'PASS','all1747_current_asset_hashes_preserved':'PASS','all670_readable_routes_and_direct_local_references_exist':'PASS, existence only','source_matches_preparation_head_for_selected_paths':'PASS','current_pypi_renderer_AST_equals_prior_packet':'PASS','prior_packet_bytes_and_receipt_preserved':'PASS'},
        'current_test_evidence':'features/data/687_readme_homepage_2026-10-15: ten normal READMEs,2690 unchanged assets,normal currency,four actual PyPI checks; not replayed here',
        'not_performed':['main/remote/config mutation','generator/render/test execution','GitHub/PyPI hosted readback','model/data/tutorial changes','PyPI upload'],
        'remaining_owner_work':'Review/apply through established private main-promotion lane, preserve newer source, then actual publication/rendered-link acceptance; release remains separate',
        'files':{},
    }
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): receipt['files'][str(p.relative_to(OUT))] = {'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
    dump(OUT/'receipt.json', receipt)
    print(json.dumps({'status':'PASS','candidate_tree':candidate,'files':len(paths),'assets':len(assets),'readable_routes':len(readable),'changed_prior_files':len(changed),'patch_sha256':sha(patch),'patch_bytes':len(patch),'compressed_patch_bytes':(OUT/'readme-docs-only.patch.gz').stat().st_size},indent=2))

if __name__ == '__main__':
    main()
