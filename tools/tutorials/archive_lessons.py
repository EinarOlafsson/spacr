#!/usr/bin/env python3
"""Archive tutorial lessons out of the library, or restore one, losslessly.

Dry run by default; ``--apply`` writes. Nothing is ever deleted:

* every deployed catalog entry (English, 7 narrated languages, 6 caption
  catalogs and ``lesson_catalog.js``), its navigation places and every
  related-lesson link into it are stored, with their original positions, in
  ``tools/tutorials/archive/lessons/<id>.json``;
* the lesson's Pages poster folder moves (``git mv``) to
  ``tools/tutorials/archive/production/<tree>/<id>/``;
* ``hf-archive`` copies the lesson's Hugging Face media to ``archive/<id>/``
  on the dataset's ``main`` in one commit, server side, after tagging;
* lesson ids, folders and media paths never change, so the player numbers
  the visible lessons 1..N by position and a restored lesson returns to its
  original place and number.

Usage::

    python tools/tutorials/archive_lessons.py archive [IDS...] [--apply]
    python tools/tutorials/archive_lessons.py restore 24 [--apply]
    python tools/tutorials/archive_lessons.py hf-archive --revision REV [--apply]
    python tools/tutorials/archive_lessons.py status

An id may be the full identity (``24_plaque``) or its number (``24``).
After archive or restore the bundled tutorial index and the module workflow
pages are regenerated, since both derive from the visible lessons.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import archived  # noqa: E402

#: The deployed library. ``tools/tutorials/release_candidate/web`` is a
#: recorded publication checkpoint whose manifest hashes its files, so it
#: is evidence of what was published then and is left as it is.
TREES = {
    'docs': REPO / 'docs/source/_extra/tutorials',
}
CATALOG_JSON = tuple(f'catalog/lessons_{lang}.json' for lang in
                     ('en', 'es', 'fr', 'hi', 'it', 'ja', 'pt-BR', 'zh-CN')) + tuple(
    f'catalog/captions_{lang}.json' for lang in ('da', 'de', 'is', 'ko', 'nb', 'sv'))
CATALOG_JS = 'lesson_catalog.js'
NAVIGATION_JS = 'module_navigation.js'
JS_HEADS = {CATALOG_JS: '"use strict";\nwindow.SPACR_LESSON_CATALOG = Object.freeze(',
            NAVIGATION_JS: '"use strict";\nwindow.SPACR_TUTORIAL_NAVIGATION = Object.freeze('}
JS_TAIL = ');\n'
HF_REPO = 'einarolafsson/spacr-tutorials'

#: The 25 lessons the maintainer approved for archiving for spaCR 1.5.1.6.
APPROVED_1_5_1_6 = ('18', '24', '25', '29', '30', '34', '46', '47', '57', '59', '60',
                    '61', '65', '66', '68', '69', '71', '72', '73', '75', '76', '77',
                    '79', '86', '87')

#: Hand-written documentation, and its translations, that links an archived
#: lesson: ``{id: [[path, text with the link, text without it], ...]}``.
#: Applied on archive, reversed on restore.
PROSE_FILE = archived.ARCHIVE_DIR / 'prose.json'


def _prose_edits(identity: str) -> list:
    if not PROSE_FILE.is_file():
        return []
    return json.loads(PROSE_FILE.read_text(encoding='utf-8')).get(identity, [])


def _read_json(path: Path):
    text = path.read_text(encoding='utf-8')
    data = json.loads(text)
    if json.dumps(data, ensure_ascii=False, indent=2) + '\n' != text:
        raise ValueError(f'{path}: unexpected formatting; refusing to rewrite it')
    return data


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _read_js(path: Path):
    text = path.read_text(encoding='utf-8')
    head = JS_HEADS[path.name]
    if not text.startswith(head) or not text.endswith(JS_TAIL):
        raise ValueError(f'{path}: unexpected wrapper; refusing to rewrite it')
    body = text[len(head):-len(JS_TAIL)]
    data = json.loads(body)
    if json.dumps(data, ensure_ascii=False) != body:
        raise ValueError(f'{path}: unexpected formatting; refusing to rewrite it')
    return data


def render(name: str, data) -> str:
    """The exact text a player tree holds for ``name`` with ``data`` in it."""
    if name.endswith('.js'):
        return JS_HEADS[Path(name).name] + json.dumps(data, ensure_ascii=False) + JS_TAIL
    return json.dumps(data, ensure_ascii=False, indent=2) + '\n'


def _write_js(path: Path, data) -> None:
    path.write_text(render(path.name, data), encoding='utf-8')


def _catalog_files(tree: Path):
    return [name for name in (CATALOG_JS, *CATALOG_JSON) if (tree / name).is_file()]


def _load(tree: Path, name: str):
    return _read_js(tree / name) if name.endswith('.js') else _read_json(tree / name)


def _save(tree: Path, name: str, data) -> None:
    (_write_js if name.endswith('.js') else _write_json)(tree / name, data)


def resolve(tokens, english_ids) -> list[str]:
    """Full lesson ids for ``tokens`` (ids or numbers), in catalog order."""
    wanted = []
    for token in tokens:
        token = str(token).strip()
        matches = [i for i in english_ids if i == token or i.split('_', 1)[0] == token.zfill(2)]
        if len(matches) != 1:
            raise SystemExit(f'{token!r} names {len(matches)} lessons in the visible catalog')
        wanted.append(matches[0])
    return [i for i in english_ids if i in wanted]


# ----------------------------------------------------------------------------- archive


def _archive_one(identity: str, record: dict, trees: dict, originals: dict) -> None:
    """Remove ``identity`` from every loaded file in ``trees``; fill ``record``.

    ``originals`` holds each file's lesson order before this run, so every
    entry records all the lessons that preceded it and can return after the
    nearest of them that is visible again, even when its neighbours were
    archived in the same run.
    """
    for tree_key, files in trees.items():
        store = record['trees'].setdefault(tree_key, _empty_store())
        for name, data in files.items():
            if name == NAVIGATION_JS:
                continue
            lessons = data['lessons']
            ids = [lesson['id'] for lesson in lessons]
            if identity not in ids:
                continue
            index = ids.index(identity)
            order = originals[tree_key][name]
            store['catalogs'][name] = {
                'index': index,
                'preceding': list(reversed(order[:order.index(identity)])),
                'previous': ids[index - 1] if index else None,
                'next': ids[index + 1] if index + 1 < len(ids) else None,
                'entry': lessons.pop(index)}
            for alias, target in list((data.get('lesson_aliases') or {}).items()):
                if target == identity:
                    store['aliases'].setdefault(name, {})[alias] = target
                    del data['lesson_aliases'][alias]
        if NAVIGATION_JS in files:
            store['navigation'] = _remove_from_navigation(
                files[NAVIGATION_JS], identity, originals[tree_key][NAVIGATION_JS])


def _strip_links(files: dict, hidden: set, records: dict, tree_key: str) -> None:
    """Drop related-lesson links to ``hidden`` lessons from every lesson left.

    Each dropped link is recorded on the hidden lesson's record with the
    scene's original link list, so a restore puts it back in its place.
    """
    for name, data in files.items():
        if name == NAVIGATION_JS:
            continue
        for lesson in data['lessons']:
            for scene_index, scene in enumerate(lesson.get('scenes') or []):
                links = scene.get('related_lessons')
                if not links or not hidden & set(links):
                    continue
                original = list(links)
                for target in original:
                    if target in hidden and target in records:
                        records[target]['trees'].setdefault(tree_key, _empty_store())[
                            'related'].append({'file': name, 'lesson': lesson['id'],
                                               'scene': scene_index, 'links': original,
                                               'keys': list(scene)})
                kept = [link for link in links if link not in hidden]
                if kept:
                    scene['related_lessons'] = kept
                else:
                    del scene['related_lessons']


def _strip_nav_hosts(nav: dict, hidden: set, records: dict, tree_key: str) -> None:
    """Clear host lessons that point at ``hidden`` lessons, recording each."""
    for s, section in enumerate(nav['sections']):
        for g, group in enumerate(section['groups']):
            host = group.get('host_lesson')
            if host in hidden and host in records:
                group['host_lesson'] = None
                places = records[host]['trees'].setdefault(tree_key, _empty_store())
                places.setdefault('navigation', {}).setdefault('host_lesson', [])
                if [s, g] not in places['navigation']['host_lesson']:
                    places['navigation']['host_lesson'].append([s, g])
    for key, route in nav['routes'].items():
        host = route.get('host_lesson')
        if host in hidden and host in records:
            route['host_lesson'] = None
            places = records[host]['trees'].setdefault(tree_key, _empty_store())
            places.setdefault('navigation', {}).setdefault('route_host_lesson', [])
            if key not in places['navigation']['route_host_lesson']:
                places['navigation']['route_host_lesson'].append(key)


def _relink(files: dict, identity: str, link: dict) -> bool:
    """Put one recorded related-lesson link back after its nearest predecessor."""
    lessons = {lesson['id']: lesson for lesson in files[link['file']]['lessons']}
    lesson = lessons.get(link['lesson'])
    if lesson is None or link['scene'] >= len(lesson.get('scenes') or []):
        return False
    scene = lesson['scenes'][link['scene']]
    links = scene.get('related_lessons') or []
    if identity not in links:
        original = link['links']
        before = list(reversed(original[:original.index(identity)]))
        anchor = next((other for other in before if other in links), None)
        links.insert(links.index(anchor) + 1 if anchor else 0, identity)
    order = {other: position for position, other in enumerate(link['links'])}
    scene['related_lessons'] = sorted(links, key=lambda other: order.get(other, len(order)))
    keys = link.get('keys') or []
    if keys:
        items = sorted(scene.items(), key=lambda item: (keys.index(item[0]) if item[0] in keys
                                                        else len(keys)))
        scene.clear()
        scene.update(items)
    return True


def _empty_store() -> dict:
    return {'catalogs': {}, 'related': [], 'aliases': {}, 'navigation': {}}


def _remove_from_navigation(nav: dict, identity: str, original: dict) -> dict:
    """Take ``identity`` out of every navigation list, recording each place."""
    places = {'lists': [], 'route': None, 'host_lesson': [], 'route_host_lesson': []}

    def take(path, values):
        if identity in values:
            before = original
            for step in path:
                before = before[step]
            values.remove(identity)
            places['lists'].append({'path': path, 'preceding': list(reversed(
                before[:before.index(identity)]))})

    take(['intro', 'lessons'], nav['intro']['lessons'])
    for s, section in enumerate(nav['sections']):
        for g, group in enumerate(section['groups']):
            take(['sections', s, 'groups', g, 'lessons'], group['lessons'])
            if group.get('host_lesson') == identity:
                group['host_lesson'] = None
                places['host_lesson'].append([s, g])
    take(['preserved_lesson_ids'], nav['preserved_lesson_ids'])
    routes = nav['routes']
    if identity in routes:
        keys = list(routes)
        routes.pop(identity)
        places['route'] = {'position': keys.index(identity),
                           'value': copy.deepcopy(original['routes'][identity])}
    for key, route in routes.items():
        if route.get('host_lesson') == identity:
            route['host_lesson'] = None
            places['route_host_lesson'].append(key)
    return places


def _restore_navigation(nav: dict, identity: str, places: dict) -> None:
    for place in places.get('lists', []):
        target = nav
        for step in place['path']:
            target = target[step]
        if identity in target:
            continue
        anchor = next((other for other in place['preceding'] if other in target), None)
        target.insert(target.index(anchor) + 1 if anchor else 0, identity)
    for s, g in places.get('host_lesson', []):
        nav['sections'][s]['groups'][g]['host_lesson'] = identity
    route = places.get('route')
    if route and identity not in nav['routes']:
        items = list(nav['routes'].items())
        items.insert(min(route['position'], len(items)), (identity, route['value']))
        nav['routes'] = dict(items)
    for key in places.get('route_host_lesson', []):
        if key in nav['routes']:
            nav['routes'][key]['host_lesson'] = identity


def _restore_catalog(data: dict, identity: str, stored: dict) -> None:
    lessons = data['lessons']
    ids = [lesson['id'] for lesson in lessons]
    if identity in ids:
        return
    anchor = next((other for other in stored.get('preceding', []) if other in ids), None)
    position = ids.index(anchor) + 1 if anchor else 0
    lessons.insert(position, stored['entry'])


def _load_trees():
    return {key: load_tree(tree) for key, tree in TREES.items() if tree.is_dir()}


def _save_trees(trees) -> None:
    for key, files in trees.items():
        for name, data in files.items():
            _save(TREES[key], name, data)


def archived_view(files: dict, hidden=None) -> dict:
    """``files`` as the library shows them with ``hidden`` lessons archived.

    ``files`` maps a tree's file names (``lesson_catalog.js``,
    ``module_navigation.js``, ``catalog/*.json``) to their parsed contents,
    as a full publication record such as ``release_candidate/web`` holds
    them. The result is what ``archive`` writes for the deployed tree, so a
    test can require the deployed library to equal its publication record
    with exactly the archived lessons taken out. ``files`` is not changed.

    :param files: parsed files of one tree.
    :param hidden: lesson ids to take out; default :data:`archived.ARCHIVED_IDS`.
    """
    files = copy.deepcopy(files)
    hidden = list(archived.ARCHIVED_IDS if hidden is None else hidden)
    originals = {'view': {name: (copy.deepcopy(data) if name == NAVIGATION_JS
                                 else [lesson['id'] for lesson in data['lessons']])
                          for name, data in files.items()}}
    records = {}
    for identity in hidden:
        records[identity] = {'schema': 1, 'id': identity, 'trees': {}}
        _archive_one(identity, records[identity], {'view': files}, originals)
    _strip_links(files, set(hidden), records, 'view')
    return files


def full_view(files: dict, tree_key: str = 'docs') -> dict:
    """``files`` with every archived lesson put back, as ``restore`` would.

    The inverse of :func:`archived_view` for a deployed tree: the complete
    published inventory, rebuilt from the deployed files and the archive
    records, without writing anything.

    :param files: parsed files of the deployed tree.
    :param tree_key: which tree's records to apply.
    """
    files = copy.deepcopy(files)
    identities = list(archived.ARCHIVED_IDS)
    records = {identity: _read_record(identity) for identity in identities}
    for identity in identities:
        store = records[identity]['trees'].get(tree_key, {})
        for name, stored in store.get('catalogs', {}).items():
            if name in files:
                _restore_catalog(files[name], identity, stored)
        for name, aliases in store.get('aliases', {}).items():
            if name in files:
                files[name].setdefault('lesson_aliases', {}).update(aliases)
        if NAVIGATION_JS in files and store.get('navigation'):
            _restore_navigation(files[NAVIGATION_JS], identity, store['navigation'])
    for identity in identities:
        for link in records[identity]['trees'].get(tree_key, {}).get('related', []):
            if link['file'] in files:
                _relink(files, identity, link)
    return files


def full_file(name: str, tree: Path | None = None):
    """One deployed file (``catalog/lessons_en.json``, ``lesson_catalog.js``
    ...) with every archived lesson put back. See :func:`full_view`."""
    tree = tree or TREES['docs']
    return full_view({name: _load(tree, name)})[name]


def load_tree(tree: Path) -> dict:
    """The parsed catalog and navigation files of one player tree."""
    return {name: _load(tree, name) for name in (*_catalog_files(tree), NAVIGATION_JS)
            if (tree / name).is_file()}


def _git(*args) -> str:
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def _move_posters(identity: str, forward: bool, apply: bool) -> list[str]:
    moved = []
    for key, tree in TREES.items():
        live = tree / 'production' / identity
        kept = archived.ARCHIVE_DIR / 'production' / key / identity
        source, target = (live, kept) if forward else (kept, live)
        if not source.is_dir():
            continue
        moved.append(f'{source.relative_to(REPO)} -> {target.relative_to(REPO)}')
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', str(source)],
                                     cwd=REPO, capture_output=True).returncode == 0
            if tracked:
                subprocess.check_call(['git', 'mv', str(source), str(target)], cwd=REPO)
            else:
                source.rename(target)
    return moved


def _prose(identity: str, forward: bool, apply: bool) -> list[str]:
    done = []
    for relative, linked, unlinked in _prose_edits(identity):
        path = REPO / relative
        text = path.read_text(encoding='utf-8')
        old, new = (linked, unlinked) if forward else (unlinked, linked)
        if old not in text:
            continue
        done.append(relative)
        if apply:
            path.write_text(text.replace(old, new, 1), encoding='utf-8')
    return done


def _regenerate() -> None:
    for command in (['tools/build_tutorial_index.py'], ['tools/build_module_workflows.py']):
        script = REPO / command[0]
        if script.is_file():
            subprocess.check_call([sys.executable, str(script)], cwd=REPO)


def _record_path(identity: str) -> Path:
    return archived.ARCHIVE_LESSONS / f'{identity}.json'


def _read_record(identity: str) -> dict:
    return json.loads(_record_path(identity).read_text(encoding='utf-8'))


def _write_record(record: dict) -> None:
    archived.ARCHIVE_LESSONS.mkdir(parents=True, exist_ok=True)
    _record_path(record['id']).write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def _write_index(index: dict) -> None:
    archived.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived.ARCHIVE_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2,
                                                 sort_keys=True) + '\n', encoding='utf-8')


def cmd_archive(args) -> int:
    trees = _load_trees()
    english = trees['docs']['catalog/lessons_en.json']
    english_ids = [lesson['id'] for lesson in english['lessons']]
    identities = resolve(args.ids or APPROVED_1_5_1_6, english_ids)
    originals = {key: {name: (copy.deepcopy(data) if name == NAVIGATION_JS
                              else [lesson['id'] for lesson in data['lessons']])
                       for name, data in files.items()}
                 for key, files in trees.items()}
    index = archived.load_index()
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')
    commit = _git('rev-parse', 'HEAD')
    records = {}
    for identity in identities:
        entry = next(lesson for lesson in english['lessons'] if lesson['id'] == identity)
        record = {'schema': 1, 'id': identity, 'trees': {}}
        _archive_one(identity, record, trees, originals)
        record['posters'] = _move_posters(identity, True, args.apply)
        record['prose'] = _prose(identity, True, args.apply)
        records[identity] = record
        index['lessons'][identity] = {
            'archived_at': stamp, 'source_commit': commit,
            'position': english_ids.index(identity), 'number': entry.get('number'),
            'title': entry.get('title'), 'app_key': entry.get('app_key'),
            'host_app_key': entry.get('host_app_key'), 'reason': args.reason,
            'hf': {'repo': HF_REPO, 'archive_path': f'archive/{identity}/'},
            'local_copy': args.local_copy, 'ws_local_copy': args.ws_local_copy}
    hidden = set(index['lessons'])
    for tree_key, files in trees.items():
        _strip_links(files, hidden, records, tree_key)
    for identity, record in records.items():
        related = sum(len(t['related']) for t in record['trees'].values())
        print(f'{identity}: {sum(len(t["catalogs"]) for t in record["trees"].values())} '
              f'catalog entries, {related} related links, posters {len(record["posters"])}, '
              f'prose {len(record["prose"])}')
        if args.apply:
            _write_record(record)
    visible = len(english['lessons'])
    print(f'{len(identities)} archived; {visible} lessons remain, numbered 1..{visible}')
    if not args.apply:
        print('dry run: nothing written (pass --apply)')
        return 0
    _save_trees(trees)
    _write_index(index)
    _regenerate()
    return 0


def cmd_restore(args) -> int:
    index = archived.load_index()
    identities = []
    for token in args.ids:
        matches = [i for i in index['lessons']
                   if i == token or i.split('_', 1)[0] == str(token).zfill(2)]
        if len(matches) != 1:
            raise SystemExit(f'{token!r} names {len(matches)} archived lessons')
        identities.append(matches[0])
    trees = _load_trees()
    still = set(index['lessons']) - set(identities)
    records = {identity: _read_record(identity) for identity in index['lessons']}
    for identity in identities:
        record = records[identity]
        for tree_key, store in record['trees'].items():
            files = trees.get(tree_key, {})
            for name, stored in store['catalogs'].items():
                _restore_catalog(files[name], identity, stored)
            for name, aliases in store.get('aliases', {}).items():
                files[name].setdefault('lesson_aliases', {}).update(aliases)
            if NAVIGATION_JS in files and store.get('navigation'):
                _restore_navigation(files[NAVIGATION_JS], identity, store['navigation'])
    for tree_key, files in trees.items():
        _strip_links(files, still, records, tree_key)
        if NAVIGATION_JS in files:
            _strip_nav_hosts(files[NAVIGATION_JS], still, records, tree_key)
    for identity in identities:
        record = records[identity]
        for tree_key, store in record['trees'].items():
            files = trees.get(tree_key, {})
            for link in store['related']:
                if link['lesson'] in still or not _relink(files, identity, link):
                    print(f'  kept for later: link from {link["lesson"]} scene {link["scene"]}')
        posters = _move_posters(identity, False, args.apply)
        prose = _prose(identity, False, args.apply)
        print(f'{identity}: restored to position {index["lessons"][identity]["position"]}; '
              f'posters {len(posters)}, prose {len(prose)}')
    for identity in still:
        for store in records[identity]['trees'].values():
            seen, unique = set(), []
            for link in store['related']:
                key = (link['file'], link['lesson'], link['scene'])
                if key not in seen:
                    seen.add(key)
                    unique.append(link)
            store['related'] = unique
    if args.apply:
        for identity in identities:
            index['lessons'].pop(identity)
            _record_path(identity).rename(archived.ARCHIVE_LESSONS / f'{identity}.restored.json')
        for identity in still:
            _write_record(records[identity])
    if not args.apply:
        print('dry run: nothing written (pass --apply)')
        return 0
    _save_trees(trees)
    _write_index(index)
    archived.ARCHIVED_IDS = archived.archived_ids()
    _regenerate()
    print('Restored. The player still pins its HF media revision; the lesson folder '
          'stays there unless a later publication removed it, in which case copy '
          f'archive/<id>/ back from {HF_REPO} main.')
    return 0


def cmd_hf_archive(args) -> int:
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download
    api = HfApi()
    identities = list(args.ids) if args.ids else list(archived.ARCHIVED_IDS)
    if not identities:
        raise SystemExit('nothing archived')
    tree = api.list_repo_tree(HF_REPO, repo_type='dataset', revision=args.revision,
                              recursive=True)
    files = [item for item in tree if hasattr(item, 'size')
             and item.path.split('/', 1)[0] in identities]
    present = {item.path.split('/', 1)[0] for item in files}
    missing = sorted(set(identities) - present)
    print(f'{len(files)} files, {sum(item.size for item in files) / 1e9:.2f} GB, '
          f'for {len(present)} lessons at {args.revision}; missing: {missing or "none"}')
    if not args.apply:
        print('dry run: nothing copied (pass --apply)')
        return 0
    if args.tag:
        api.create_tag(HF_REPO, repo_type='dataset', tag=args.tag, revision=args.branch,
                       exist_ok=True)
        print(f'tagged {args.branch} as {args.tag}')
    operations = []
    scratch = Path(args.scratch)
    for item in files:
        target = f'archive/{item.path}'
        if item.lfs:
            operations.append(CommitOperationCopy(src_path_in_repo=item.path,
                                                  path_in_repo=target,
                                                  src_revision=args.revision))
        else:
            local = hf_hub_download(HF_REPO, item.path, repo_type='dataset',
                                    revision=args.revision, local_dir=scratch)
            operations.append(CommitOperationAdd(path_in_repo=target, path_or_fileobj=local))
    info = api.create_commit(
        HF_REPO, repo_type='dataset', revision=args.branch, operations=operations,
        commit_message=f'Archive {len(present)} tutorial lessons under archive/',
        commit_description=f'Copied from {args.revision}; originals are unchanged.')
    print(f'committed {info.oid} to {args.branch}')
    index = archived.load_index()
    for identity in present:
        if identity in index['lessons']:
            index['lessons'][identity]['hf'].update(
                {'archive_commit': info.oid, 'source_revision': args.revision})
    _write_index(index)
    return 0


def cmd_status(_args) -> int:
    index = archived.load_index()
    for identity, entry in sorted(index['lessons'].items(), key=lambda kv: kv[1]['position']):
        print(f'{identity:32} app_key={entry.get("app_key") or "-":24} '
              f'hf={entry["hf"].get("archive_commit", "not copied")[:10]}')
    print(f'{len(index["lessons"])} archived')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    archive = sub.add_parser('archive', help='hide lessons from the library')
    archive.add_argument('ids', nargs='*', help='ids or numbers; default: the approved 25')
    archive.add_argument('--apply', action='store_true')
    archive.add_argument('--reason', default='Hidden for spaCR 1.5.1.6 (maintainer approval)')
    archive.add_argument('--local-copy', default='/nas_mnt/data/sync/tutorial-archive-20261026')
    archive.add_argument('--ws-local-copy', default='/media/carruthers/mnt1/tutorial-archive-20261026')
    restore = sub.add_parser('restore', help='put archived lessons back in place')
    restore.add_argument('ids', nargs='+')
    restore.add_argument('--apply', action='store_true')
    hf = sub.add_parser('hf-archive', help='copy media to archive/<id>/ on Hugging Face')
    hf.add_argument('ids', nargs='*')
    hf.add_argument('--revision', required=True, help='the revision the player pins')
    hf.add_argument('--branch', default='main')
    hf.add_argument('--tag', default='')
    hf.add_argument('--scratch', default='/mnt/firecuda2/codex/scratch-docs/hf-archive-small')
    hf.add_argument('--apply', action='store_true')
    sub.add_parser('status', help='list archived lessons')
    args = parser.parse_args(argv)
    return {'archive': cmd_archive, 'restore': cmd_restore, 'hf-archive': cmd_hf_archive,
            'status': cmd_status}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
