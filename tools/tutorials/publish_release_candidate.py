#!/usr/bin/env python3
"""Publish a verified tutorial candidate: a new media revision, then the Pages tree.

Two separate, re-runnable steps. Neither touches ``main`` on either host.

``upload``
    Pushes the candidate's ``media_host/`` (all narration voices, their timing
    sidecars and the 4K masters) to a NEW branch of the media dataset with the
    publisher's own ``hf upload`` command. It never uploads to ``main``, because
    the live site reads ``resolve/main`` until Pages changes, and new narration
    under the old live catalogs would desynchronise it. Every uploaded file is
    then READ BACK from the resulting commit and hashed against the candidate
    manifest. The commit is tagged only after every file matches. An existing
    branch or tag name is refused, so a published revision is never overwritten.

``pages``
    Writes the candidate's ``web/`` into ``docs/source/_extra/tutorials``,
    byte-checked against the manifest, with the media roots pinned to the
    uploaded COMMIT (only a commit id cannot be moved). Pushing ``nightly``
    publishes its preview; pushing ``main`` publishes the main site through
    ``.github/workflows/docs.yml``. Each channel retains its own media revision.
    Archived lessons (``tools/tutorials/archived.py``) stay hidden: their
    catalog entries, navigation places and posters are left out of Pages
    exactly as ``archive_lessons.py`` removes them; the candidate keeps them.

Usage (tutorial toolchain python, ``cd tools/tutorials``)::

    publish_release_candidate.py upload <candidate> --branch B --tag T
    publish_release_candidate.py readback <candidate> --commit C
    publish_release_candidate.py pages <candidate>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from check_completed_matrix import digest
from build_release_candidate import copy_checked
from append_staged_lessons import parse_javascript
from stage_lesson import REPO, read, write
from validate_candidate import validate
import archive_lessons
from archived import ARCHIVED_IDS

sys.path.insert(0, str(REPO / 'tools'))
import build_tutorial_index

sys.path.insert(0, str(Path(__file__).resolve().parent / 'authoring' / 'tools'))
from publish_tutorials import HF_DATASET, hf_upload_ready  # noqa: E402


def publication_view(root):
    """The exact normal visible projection; the candidate stays complete."""
    files = archive_lessons.load_tree(Path(root) / 'web')
    return archive_lessons.archived_view(files, hidden=ARCHIVED_IDS) if ARCHIVED_IDS else files


def publication_evidence_matches(checks, pages, catalog, manifest):
    """Require one entire known domain, never a count-only passing subset."""
    listed = {row['id'] for row in catalog['lessons']}
    ready = {row['id'] for row in catalog['lessons'] if row.get('status') != 'coming_soon'}
    cases = checks.get('ready_playback_cases', [])
    return (len(cases) == len(ready) and {case.get('lesson') for case in cases} == ready
            and all(case.get('passed') is True for case in cases)
            and ('routes' not in checks or checks['routes'] == len(catalog['lessons']))
            and ('ready_lessons' not in checks or checks['ready_lessons'] == len(ready))
            and pages.get('ready') == len(ready) and pages.get('routes') == len(catalog['lessons'])
            and pages.get('hosted_web_copies') == len(set(manifest.get('hosted_web_lessons', [])) & listed))

RECEIPT = 'publication-receipt.json'
PAGES = REPO / 'docs/source/_extra/tutorials'
HOST = f'https://huggingface.co/datasets/{HF_DATASET}/resolve/'
VERSIONED = ('app_v2.js', 'styles.css', 'lesson_catalog.js', 'module_navigation.js', 'voice_catalog.js')


def media_records(manifest):
    prefix = 'media_host/'
    return {r['path'][len(prefix):]: r for r in manifest['files'] if r['path'].startswith(prefix)}


def require_verified_candidate(root):
    """The candidate must be held, unmodified and browser-verified as it stands."""
    manifest_path = root / 'release-manifest.json'
    manifest = read(manifest_path)
    if manifest.get('release_hold') is not True or manifest.get('uploaded') is not False:
        raise SystemExit('Expected a held candidate that has not been uploaded')
    validate(root, include_hosted_media=True)
    report = read(root / 'checks/candidate-browser-checks.json')
    ready = manifest['ready_lessons']
    if (report.get('passed') is not True
            or report.get('manifest_sha256') != digest(manifest_path)
            or len(report['ready_playback_cases']) != ready
            or not all(case.get('passed') is True for case in report['ready_playback_cases'])):
        raise SystemExit('Browser evidence does not describe this exact candidate')
    return manifest


def git_blob_id(path):
    """Git's object id for a plain (non-LFS) file."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def readback(root, commit, *, workers=12):
    """Hash every candidate media file as the host serves it at ``commit``."""
    import requests
    from huggingface_hub import HfApi, get_token

    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise SystemExit('Read back a full commit id, not a movable name')
    manifest = read(root / 'release-manifest.json')
    token = get_token()
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    expected = media_records(manifest)
    tree = {}
    for entry in HfApi().list_repo_tree(HF_DATASET, repo_type='dataset', revision=commit,
                                        recursive=True):
        if entry.__class__.__name__ == 'RepoFile':
            tree[entry.path] = entry
    metadata_failures = []
    for relative, record in expected.items():
        entry = tree.get(relative)
        if entry is None or entry.size != record['bytes']:
            metadata_failures.append(relative)
        elif entry.lfs is not None:
            if entry.lfs.sha256 != record['sha256']:
                metadata_failures.append(relative)
        elif entry.blob_id != git_blob_id(root / 'media_host' / relative):
            metadata_failures.append(relative)

    def fetch(relative):
        url = HOST + commit + '/' + relative
        for attempt in range(4):
            try:
                value, size = hashlib.sha256(), 0
                with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    for block in response.iter_content(1024 * 1024):
                        value.update(block)
                        size += len(block)
                return relative, value.hexdigest(), size
            except requests.RequestException as error:
                if attempt == 3:
                    raise
                delay = 5 * (attempt + 1)
                response = error.response
                if response is not None and response.status_code == 429:
                    reset = re.search(r'(?:^|;)\s*t=(\d+)', response.headers.get('RateLimit', ''))
                    retry = response.headers.get('Retry-After', '')
                    delay = max(delay, int(reset[1]) + 1 if reset else 0,
                                int(retry) + 1 if retry.isdigit() else 0)
                    print(f'  media host rate limit; retrying after {delay}s', flush=True)
                time.sleep(delay)

    downloaded, byte_failures = 0, []
    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, relative) for relative in sorted(expected)]
        for done, future in enumerate(as_completed(futures), 1):
            relative, sha256, size = future.result()
            record = expected[relative]
            if sha256 != record['sha256'] or size != record['bytes']:
                byte_failures.append(relative)
            downloaded += size
            if done % 500 == 0 or done == len(futures):
                print(f'  read back {done}/{len(futures)} files, {downloaded / 1e9:.2f} GB', flush=True)
    extra = sorted(set(tree) - set(expected))
    result = {'commit': commit, 'files_expected': len(expected),
              'metadata_matched': len(expected) - len(metadata_failures),
              'metadata_failures': metadata_failures,
              'bytes_read_back': downloaded,
              'downloaded_sha256_matched': len(expected) - len(byte_failures),
              'download_failures': byte_failures,
              'files_on_revision_not_in_candidate': extra,
              'seconds': round(time.time() - started, 1),
              'passed': not metadata_failures and not byte_failures}
    print(json.dumps({k: v for k, v in result.items() if k != 'files_on_revision_not_in_candidate'}), flush=True)
    return result


def upload(root, branch, tag):
    from huggingface_hub import HfApi

    manifest = require_verified_candidate(root)
    if not hf_upload_ready():
        raise SystemExit(1)
    api = HfApi()
    refs = api.list_repo_refs(HF_DATASET, repo_type='dataset')
    taken = {ref.name for ref in (*refs.branches, *refs.tags)}
    if branch in taken or tag in taken or branch == tag:
        raise SystemExit(f'Refusing to reuse an existing revision name: {branch} / {tag}')
    base = next(ref.target_commit for ref in refs.branches if ref.name == 'main')
    api.create_branch(HF_DATASET, repo_type='dataset', branch=branch, revision=base, exist_ok=False)
    command = ['hf', 'upload', HF_DATASET, str(root / 'media_host'), '.', '--repo-type', 'dataset',
               '--revision', branch,
               '--commit-message', f'Tutorial media for release candidate {root.name}',
               '--commit-description', f'release-manifest.json sha256 {digest(root / "release-manifest.json")}']
    print('  ' + ' '.join(command), flush=True)
    if subprocess.run(command).returncode != 0:
        raise SystemExit('hf upload failed; the branch exists and holds no verified revision')
    commits = api.list_repo_commits(HF_DATASET, repo_type='dataset', revision=branch)
    commit = commits[0].commit_id
    if len(commits) < 2 or commits[1].commit_id != base:
        raise SystemExit(f'Unexpected history on {branch}: expected one commit on {base}')
    checked = readback(root, commit)
    receipt = {'repository': f'datasets/{HF_DATASET}', 'branch': branch, 'base_commit': base,
               'commit': commit, 'tag': None, 'media_root': HOST + commit,
               'manifest_sha256': digest(root / 'release-manifest.json'),
               'media_files': len(media_records(manifest)),
               'media_bytes': manifest['media_host_bytes'], 'readback': checked}
    if not checked['passed']:
        write(root / RECEIPT, receipt)
        raise SystemExit('Read-back failed; the revision was NOT tagged')
    api.create_tag(HF_DATASET, repo_type='dataset', tag=tag, revision=commit, exist_ok=False,
                   tag_message=f'Tutorial media for {root.name}')
    # An annotated tag's ref names the TAG OBJECT, not the commit (on 2026-09-15
    # list_repo_refs gave 8b402f85 for a tag resolving to e35b2c12), so compare
    # what the tag RESOLVES to.
    if api.dataset_info(HF_DATASET, revision=tag).sha != commit:
        raise SystemExit('Tag does not resolve to the verified commit')
    receipt['tag'] = tag
    write(root / RECEIPT, receipt)
    print('PUBLISHED MEDIA', HOST + commit, flush=True)


def resume_receipt(root, branch, tag, checked):
    """Write the receipt for a run whose read-back passed but whose receipt was not written.

    Read-only against the host: it creates no branch or tag, and it refuses
    unless the tag already resolves to the read-back commit and that commit is
    the branch's single commit on main.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    manifest = read(root / 'release-manifest.json')
    expected = media_records(manifest)
    commit = checked['commit']
    history = api.list_repo_commits(HF_DATASET, repo_type='dataset', revision=branch)
    base = history[1].commit_id if len(history) > 1 else None
    main = api.dataset_info(HF_DATASET).sha
    if history[0].commit_id != commit or base != main:
        raise SystemExit(f'{branch} is not exactly one commit ({commit}) on main ({main})')
    if (checked.get('passed') is not True or checked['files_expected'] != len(expected)
            or checked['downloaded_sha256_matched'] != len(expected)
            or checked['metadata_matched'] != len(expected)
            or checked['metadata_failures'] or checked['download_failures']):
        raise SystemExit('Read-back evidence does not cover every file of this commit')
    if api.dataset_info(HF_DATASET, revision=tag).sha != commit:
        raise SystemExit('Tag is missing or does not resolve to the read-back commit')
    tree = {entry.path for entry in api.list_repo_tree(HF_DATASET, repo_type='dataset', revision=commit,
                                                       recursive=True)
            if entry.__class__.__name__ == 'RepoFile'}
    if set(expected) - tree:
        raise SystemExit('Commit lacks candidate media')
    checked = {**checked, 'files_on_revision_not_in_candidate': sorted(tree - set(expected))}
    write(root / RECEIPT, {'repository': f'datasets/{HF_DATASET}', 'branch': branch, 'base_commit': base,
                           'commit': commit, 'tag': tag, 'media_root': HOST + commit,
                           'manifest_sha256': digest(root / 'release-manifest.json'),
                           'media_files': len(expected), 'media_bytes': manifest['media_host_bytes'],
                           'readback': checked,
                           'readback_source': 'completed immutable-commit readback resumed after upload'})
    print('RECEIPT', root / RECEIPT, HOST + commit, flush=True)


def reuse_media_receipt(root, baseline):
    """Reuse a completed immutable publication's media, not an interrupted upload.

    Current main may advance. The original branch/tag/commit, every host file
    identity and the original full readback must still describe these exact media.
    No hosted bytes are downloaded, uploaded or relabelled as newly checked.
    """
    from copy import deepcopy
    from huggingface_hub import HfApi

    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    if root == baseline or (root / RECEIPT).exists():
        raise SystemExit('Require a distinct candidate without an existing receipt')
    manifest = require_verified_candidate(root)
    validate(baseline, require_browser=True)
    previous = read(baseline / 'release-manifest.json')
    saved = read(baseline / RECEIPT)
    expected = media_records(manifest)
    if not expected or expected != media_records(previous):
        raise SystemExit('Media reuse requires exactly the original media records')
    count = len(expected)
    size = sum(record['bytes'] for record in expected.values())
    commit, branch, tag = (saved.get(key) for key in ('commit', 'branch', 'tag'))
    checked, pages = saved.get('readback', {}), saved.get('pages', {})
    published = read(baseline / 'checks/published-media-browser-checks.json')
    full = read(baseline / 'web/catalog/lessons_en.json')
    shown = archive_lessons.archived_view({'catalog/lessons_en.json': full},
                                         hidden=ARCHIVED_IDS)['catalog/lessons_en.json']
    complete_publication = any(publication_evidence_matches(published, pages, catalog, previous)
                               for catalog in (full, shown))
    cases = published.get('ready_playback_cases', [])
    if (not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit)
            or not isinstance(saved.get('base_commit'), str)
            or not re.fullmatch(r'[0-9a-f]{40}', saved['base_commit'])
            or not isinstance(branch, str) or not branch or branch == 'main'
            or not isinstance(tag, str) or not tag or branch == tag
            or saved.get('repository') != f'datasets/{HF_DATASET}'
            or saved.get('media_root') != HOST + commit
            or saved.get('manifest_sha256') != digest(baseline / 'release-manifest.json')
            or saved.get('media_files') != count or saved.get('media_bytes') != size
            or manifest.get('media_host_bytes') != size or previous.get('media_host_bytes') != size
            or checked.get('passed') is not True or checked.get('commit') != commit
            or any(checked.get(key) != count for key in
                   ('files_expected', 'metadata_matched', 'downloaded_sha256_matched'))
            or checked.get('bytes_read_back') != size
            or checked.get('metadata_failures') != [] or checked.get('download_failures') != []
            or not complete_publication
            or published.get('passed') is not True or published.get('published') is not False
            or published.get('manifest_sha256') != saved['manifest_sha256']
            or published.get('media_root') != saved['media_root']
            or not pages.get('index_sha256') or published.get('index_sha256') != pages['index_sha256']
            or published.get('placeholder_cases') != []
            or not all(case.get('passed') is True for case in cases)):
        raise SystemExit('Original verified publication evidence is incomplete or stale')
    api = HfApi()
    if any(api.dataset_info(HF_DATASET, revision=ref).sha != commit for ref in (branch, tag)):
        raise SystemExit('Original publication branch/tag no longer resolves to its verified commit')
    history = api.list_repo_commits(HF_DATASET, repo_type='dataset', revision=commit)
    if len(history) < 2 or history[0].commit_id != commit or history[1].commit_id != saved['base_commit']:
        raise SystemExit('Immutable publication history differs from its recorded historical base')
    tree = {entry.path: entry for entry in api.list_repo_tree(
        HF_DATASET, repo_type='dataset', revision=commit, recursive=True)
        if entry.__class__.__name__ == 'RepoFile'}
    for relative, record in expected.items():
        entry = tree.get(relative)
        if (entry is None or entry.size != record['bytes']
                or (entry.lfs is not None and entry.lfs.sha256 != record['sha256'])
                or (entry.lfs is None and entry.blob_id != git_blob_id(root / 'media_host' / relative))):
            raise SystemExit('Immutable host metadata differs: ' + relative)
    if sorted(set(tree) - set(expected)) != checked.get('files_on_revision_not_in_candidate', []):
        raise SystemExit('Immutable host inventory differs from the original readback')
    receipt = deepcopy(saved)
    receipt.pop('pages', None)  # Old Pages/index evidence is not evidence for this player.
    receipt.update(manifest_sha256=digest(root / 'release-manifest.json'),
                   readback_source='unchanged media reused from verified publication',
                   media_reuse={'baseline_manifest_sha256': saved['manifest_sha256'],
                                'baseline_receipt_sha256': digest(baseline / RECEIPT)})
    write(root / RECEIPT, receipt)
    print('RECEIPT', root / RECEIPT, HOST + commit, flush=True)


def drop_superseded_web_copies(pages_root, manifest):
    """Remove the Pages copy of every web video the candidate now hosts.

    Only the exact local path of a hosted lesson's web copy is removed; posters,
    catalogs and other lessons' copies stay.
    """
    removed = []
    for record in manifest['files']:
        parts = Path(record['path']).parts
        if len(parts) == 4 and parts[0] == 'media_host' and parts[2] == 'web':
            relative = Path('production') / parts[1] / 'video' / parts[3]
            if (pages_root / relative).is_file():
                (pages_root / relative).unlink()
                removed.append(relative.as_posix())
    return removed


def retire_withdrawn_pages(root, pages_root, manifest):
    """Archive explicitly redirected lessons before removing their live copies."""
    withdrawn = manifest.get('withdrawn_lessons', [])
    catalog = parse_javascript((root / 'web/lesson_catalog.js').read_text(encoding="utf-8"))
    active = {row['id'] for row in catalog['lessons']}
    aliases = catalog.get('lesson_aliases', {})
    if (len(withdrawn) != len(set(withdrawn)) or any(
            not re.fullmatch(r'[0-9]+_[a-z0-9_]+', identity)
            or identity in active or aliases.get(identity) not in active
            for identity in withdrawn)):
        raise ValueError('Withdrawn Pages lessons need unique, valid active redirects')
    records = []
    archive = root / 'retired-pages'
    if archive.is_symlink():
        raise ValueError('Refusing to archive redirected lessons through a symlink')
    for identity in withdrawn:
        source = pages_root / 'production' / identity
        backup = archive / identity
        if source.is_symlink() or backup.is_symlink():
            raise ValueError('Refusing to archive a redirected lesson through a symlink')
        paths = sorted(source.rglob('*')) if source.exists() else []
        if any(path.is_symlink() for path in paths):
            raise ValueError('Refusing to archive redirected lesson symlinks')
        original = [(path, digest(path)) for path in paths if path.is_file()]
        for path, expected in original:
            target = backup / path.relative_to(source)
            if target.exists() and digest(target) != expected:
                raise ValueError('The retained Pages archive has different bytes')
            copy_checked(path, target, [], archive, expected)
        if any(digest(path) != expected for path, expected in original):
            raise ValueError('Withdrawn Pages bytes changed during archiving')
        for path, expected in original:
            path.unlink()
        for directory in sorted((path for path in paths if path.is_dir()),
                                key=lambda path: len(path.parts), reverse=True):
            directory.rmdir()
        if source.exists():
            source.rmdir()
        for path in sorted(backup.rglob('*')):
            if path.is_file():
                records.append(dict(path=path.relative_to(archive).as_posix(),
                                    sha256=digest(path), bytes=path.stat().st_size))
    return records


def pages(root, key):
    """Write the Pages tree from the candidate, pinned to the verified commit."""
    receipt = read(root / RECEIPT)
    commit = receipt['commit']
    if not receipt['readback']['passed'] or not receipt.get('tag') or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise SystemExit('No verified, tagged media revision to point at')
    manifest = read(root / 'release-manifest.json')
    if receipt['manifest_sha256'] != digest(root / 'release-manifest.json'):
        raise SystemExit('The candidate changed after its media was published')
    validate(root)
    web = {r['path'][len('web/'):]: r for r in manifest['files'] if r['path'].startswith('web/')}
    previous = (PAGES / 'index.html').read_text(encoding='utf-8')
    unchanged = {name for name in VERSIONED
                 if (PAGES / name).is_file() and digest(PAGES / name) == web[name]['sha256']}
    written = []
    for relative, record in sorted(web.items()):
        if relative == 'index.html':
            continue
        if relative.startswith('production/') and relative.split('/')[1] in ARCHIVED_IDS:
            continue
        source, target = root / 'web' / relative, PAGES / relative
        if digest(source) != record['sha256']:
            raise SystemExit(f'Candidate file changed: {relative}')
        copy_checked(source, target, [], PAGES, record['sha256'])
        if digest(target) != record['sha256']:
            raise SystemExit(f'Copy differs: {relative}')
        written.append(relative)

    if ARCHIVED_IDS:
        shown = publication_view(root)
        for name, data in shown.items():
            (PAGES / name).write_text(archive_lessons.render(name, data), encoding='utf-8')
    index = (root / 'web/index.html').read_text(encoding='utf-8')
    catalog = read(PAGES / 'catalog/lessons_en.json')['lessons']
    ready = sum(lesson.get('status') != 'coming_soon' for lesson in catalog)

    def replace(pattern, value, text):
        text, count = re.subn(pattern, value, text)
        if count != 1:
            raise SystemExit(f'Expected exactly one match for {pattern}, found {count}')
        return text

    hosted = [r for r in manifest['files'] if len(Path(r['path']).parts) == 4
              and r['path'].startswith('media_host/') and Path(r['path']).parts[2] == 'web']
    if hosted and 'data-web-root=' not in index:
        raise SystemExit('Hosted web copies need a data-web-root on the page')
    attributes = ('audio-root', 'video4k-root') + (('web-root',) if 'data-web-root=' in index else ())
    for attribute in attributes:
        index = replace(rf'data-{attribute}="\.\./media_host"', f'data-{attribute}="{receipt["media_root"]}"', index)
    versions = {}
    for name in VERSIONED:
        if name in unchanged:
            match = re.search(re.escape(name) + r'\?v=([^"\s]+)', previous)
            version = match.group(1) if match else key
        else:
            version = key
        versions[name] = version
        index = replace(re.escape(name) + r'(?:\?v=[^"\s]+)?(?=")', f'{name}?v={version}', index)
    index = replace(r'0 of \d+ complete', f'0 of {ready} complete', index)
    index = replace(r'(id="available-count">)\d+', rf'\g<1>{ready}', index)
    index = replace(r'(id="total-count">)\d+', rf'\g<1>{len(catalog)}', index)
    index = replace(r'Lesson 1 of \d+', f'Lesson 1 of {len(catalog)}', index)
    if '../media_host' in index or 'data-production-root="production"' not in index:
        raise SystemExit('Pages index still points at local media')
    withdrawn_archive = retire_withdrawn_pages(root, PAGES, manifest)
    superseded = drop_superseded_web_copies(PAGES, manifest)
    index_temporary = PAGES / 'index.html.publishing'
    index_temporary.write_text(index, encoding='utf-8')
    index_temporary.replace(PAGES / 'index.html')
    bundled_index = REPO / 'spacr/resources/tutorial_index.json'
    bundled_temporary = bundled_index.with_suffix('.json.publishing')
    bundled_temporary.write_text(json.dumps(
        build_tutorial_index.build(PAGES / 'lesson_catalog.js'),
        ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    bundled_temporary.replace(bundled_index)
    receipt['pages'] = {'destination': str(PAGES.relative_to(REPO)), 'files_from_candidate': len(written) + 1,
                        'bundled_tutorial_index_sha256': digest(bundled_index),
                        'index_sha256': digest(PAGES / 'index.html'),
                        'candidate_index_sha256': web['index.html']['sha256'],
                        'cache_key': key, 'unchanged_versioned_assets': sorted(unchanged),
                        'versioned_assets': versions,
                        'hosted_web_copies': sum(Path(row['path']).parts[1] in {l['id'] for l in catalog}
                                                 for row in hosted), 'removed_local_web_copies': superseded,
                        'withdrawn_pages_archive': {'directory': str(root / 'retired-pages'),
                                                    'files': withdrawn_archive},
                        'ready': ready, 'routes': len(catalog)}
    write(root / RECEIPT, receipt)
    print('PAGES TREE', PAGES, receipt['pages'], flush=True)


def record(root):
    """Lift the hold in the repository checkpoint, only on complete publication evidence.

    The candidate's own manifest stays as built (held): browser evidence hashes
    it. The lift is recorded beside it, in checkpoint.json and the receipt.
    """
    receipt = read(root / RECEIPT)
    checks = read(root / 'checks/published-media-browser-checks.json')
    manifest_sha = digest(root / 'release-manifest.json')
    target = REPO / 'tools/tutorials/release_candidate'
    checkpoint = read(target / 'checkpoint.json')
    manifest = read(root / 'release-manifest.json')
    full = read(root / 'web/catalog/lessons_en.json')['lessons']
    if (manifest['routes'] != len(full)
            or manifest['ready_lessons'] != sum(row.get('status') != 'coming_soon' for row in full)):
        raise SystemExit('Full candidate counts differ from its unchanged catalog')
    catalog = publication_view(root)['catalog/lessons_en.json']
    from build_appended_candidate import require_baseline_receipt
    require_baseline_receipt(manifest, receipt, manifest_sha)
    if (not receipt.get('tag') or not receipt['readback']['passed'] or 'pages' not in receipt
            or receipt['manifest_sha256'] != manifest_sha or checkpoint['manifest_sha256'] != manifest_sha
            or checks.get('passed') is not True or checks.get('manifest_sha256') != manifest_sha
            or checks.get('media_root') != receipt['media_root']
            or checks.get('index_sha256') != digest(PAGES / 'index.html')
            or checks.get('index_sha256') != receipt['pages'].get('index_sha256')
            or not publication_evidence_matches(checks, receipt['pages'], catalog, manifest)):
        raise SystemExit('Publication evidence is incomplete or describes a different candidate/tree')
    for name, source in ((RECEIPT, root / RECEIPT),
                         ('published-media-browser-checks.json', root / 'checks/published-media-browser-checks.json')):
        (target / name).write_bytes(source.read_bytes())
    checkpoint.update(release_hold=False, media_uploaded=True, pages_tree_ready=True, published=False,
                      media_revision={key: receipt[key] for key in
                                      ('repository', 'branch', 'tag', 'commit', 'media_root')},
                      publication_note='Media revision uploaded and read back; docs.yml publishes '
                                       'the committed Pages tree to the matching nightly or main channel.')
    write(target / 'checkpoint.json', checkpoint)
    print('HOLD LIFTED in', target / 'checkpoint.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    up = commands.add_parser('upload')
    up.add_argument('candidate', type=Path)
    up.add_argument('--branch', required=True)
    up.add_argument('--tag', required=True)
    back = commands.add_parser('readback')
    back.add_argument('candidate', type=Path)
    back.add_argument('--commit', required=True)
    page = commands.add_parser('pages')
    page.add_argument('candidate', type=Path)
    page.add_argument('--cache-key', required=True)
    lift = commands.add_parser('record', help='after verify_release_candidate.py --published passes')
    lift.add_argument('candidate', type=Path)
    resume = commands.add_parser('resume-receipt', help='read-only: receipt for a passed read-back')
    resume.add_argument('candidate', type=Path)
    resume.add_argument('--branch', required=True)
    resume.add_argument('--tag', required=True)
    resume.add_argument('--readback-json', type=Path, required=True,
                        help="the read-back line that run printed, saved as JSON")
    reuse = commands.add_parser('reuse-media', help='reuse exact media from a verified immutable publication; no upload/download')
    reuse.add_argument('candidate', type=Path)
    reuse.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    if args.command == 'reuse-media':
        reuse_media_receipt(candidate, args.baseline)
    elif args.command == 'resume-receipt':
        resume_receipt(candidate, args.branch, args.tag, read(args.readback_json))
    elif args.command == 'upload':
        upload(candidate, args.branch, args.tag)
    elif args.command == 'readback':
        if not readback(candidate, args.commit)['passed']:
            raise SystemExit(1)
    elif args.command == 'pages':
        pages(candidate, args.cache_key)
    else:
        record(candidate)
