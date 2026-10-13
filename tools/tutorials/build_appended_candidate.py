"""Publish individual verified lessons without waiting for every voice.

Existing lesson objects and hosted media are preserved from a verified release.
New lessons expose only audio that passes current-source checks. Missing or
incompatible translations are registered and use English. Upload, readback,
candidate playback and publication still use the regular release tools.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import tempfile

from apply_translation_review import translated_lesson
from append_staged_lessons import parse_javascript
from audit_staged_catalogs import CATALOGS
from build_navigation import build as navigation
from build_release_candidate import copy_checked, require_web_receipt
from check_completed_matrix import digest
from stage_lesson import REPO, read, write
from validate_candidate import hosted_web_path, require_consistent_web_hosting, validate


def append_catalogs(published, lessons, voices, reviews, *, replace=False, current_hosts=None):
    """Preserve published objects and bind each added translation to its source."""
    existing = published['lessons_en.json']['lessons']
    identities = {item['id'] for item in existing}
    # New lessons continue from the highest published number; withdrawn
    # lessons (83, 84) leave gaps that are never reused.
    last = max((item['number'] for item in existing), default=0)
    numbers = list(range(last + 1, last + len(lessons) + 1))
    if not lessons or len({item['id'] for item in lessons}) != len(lessons):
        raise ValueError('Select at least one unique lesson')
    positions = {item['id']: index for index, item in enumerate(existing)}
    if replace:
        for item in lessons:
            if item['id'] not in positions or any(
                    item.get(key) != existing[positions[item['id']]].get(key)
                    for key in ('number', 'app_key')):
                raise ValueError('A refresh must preserve the existing lesson identity and route')
            old_host = existing[positions[item['id']]].get('host_app_key')
            if (item.get('host_app_key') != old_host
                    and (current_hosts is None or item['id'] not in current_hosts
                         or item.get('host_app_key') != current_hosts[item['id']])):
                raise ValueError('A changed host must match the current GUI route')
    elif ([item['number'] for item in lessons] != numbers
            or any(item['id'] in identities for item in lessons)):
        raise ValueError('Append new, unique lessons in contiguous number order')
    for item in lessons:
        identity = item['id']
        if (not re.fullmatch(r'[0-9]+_[a-z0-9_]+', identity)
                or int(identity.split('_', 1)[0]) != item['number']
                or not item.get('scenes') or item.get('status') == 'coming_soon'
                or 'af_heart' not in voices.get(identity, {}).get('en', [])):
            raise ValueError('Every appended lesson needs scenes and verified English narration')
    catalogs, compatibility = deepcopy(published), []
    for filename in CATALOGS:
        language = filename.split('_', 1)[1].removesuffix('.json')
        if [row['id'] for row in published[filename]['lessons']] != [row['id'] for row in existing]:
            raise ValueError('Published language catalogs disagree on lesson order')
        for english in lessons:
            identity = english['id']
            translated = deepcopy(english)
            if language != 'en':
                review = reviews.get((identity, language))
                try:
                    if not review or review.get('lesson') != identity or review.get('language') != language:
                        raise ValueError('Missing or mismatched translation review')
                    translated = translated_lesson(review, english, {})
                    status, reason = 'source_bound_review', None
                except (ValueError, KeyError, TypeError) as error:
                    status, reason = 'english_fallback', str(error)
                compatibility.append(dict(lesson=identity, language=language, status=status, reason=reason))
            translated['narration_voices'] = deepcopy(voices[identity])
            if replace:
                catalogs[filename]['lessons'][positions[identity]] = translated
            else:
                catalogs[filename]['lessons'].append(translated)
    return catalogs, compatibility


def verify_tracks(stage, lesson, catalogs):
    """Decode and check current synthesis inputs for every offered audio track."""
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    import verify_audio_release as audio

    identity = lesson['id']
    declared, records = {}, []
    for path in sorted((stage / 'production' / identity / 'audio').glob('*/*.m4a')):
        language, voice = path.parent.name, path.stem
        if language not in audio.LANGUAGES or voice not in audio.LANGUAGES[language][1]:
            raise ValueError(f'Unsupported narration track: {path}')
        localized = next(row for row in catalogs[f'lessons_{language}.json']['lessons'] if row['id'] == identity)
        code = 'b' if language == 'en' and voice.startswith('b') else audio.LANGUAGES[language][0]
        dialect = audio.narration_dialect(language, code, voice)
        speed = audio.resolve_voice_speed(voice)
        plans = audio.prepare_scene_plans(localized, language, dialect, speed, voice=voice)
        errors = audio.check_track(audio.TrackSpec(path, localized, language, code, voice, dialect, speed, plans))
        if errors:
            raise ValueError(f'{identity}/{language}/{voice}: {errors}')
        declared.setdefault(language, []).append(voice)
        records.append(dict(language=language, voice=voice, audio_sha256=digest(path),
                            timing_sha256=digest(path.with_suffix('.json'))))
    if 'af_heart' not in declared.get('en', []):
        raise ValueError('New lessons require verified English Heart narration')
    return declared, records


def verify_retained_tracks(stage, baseline, lesson, catalogs, manifest):
    """Decode only exact published narration; never claim new renderer freshness."""
    from retain_narration import catalog_lesson, require_timing, require_unchanged_lesson
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    import verify_audio_release as audio

    identity = lesson['id']
    old = catalog_lesson(baseline / 'web/catalog/lessons_en.json', identity)
    canonical = deepcopy(old)
    declared = canonical.pop('narration_voices', None)
    if canonical != lesson or not isinstance(declared, dict) or not declared:
        raise ValueError('Retained narration requires unchanged canonical lesson and published voices')
    expected = set()
    for language, voices in declared.items():
        if (language not in audio.LANGUAGES or not isinstance(voices, list) or not voices
                or len(voices) != len(set(voices))
                or any(voice not in audio.LANGUAGES[language][1] for voice in voices)):
            raise ValueError('Invalid published retained voice inventory')
        expected.update((language, voice) for voice in voices)
    if ('en', 'af_heart') not in expected:
        raise ValueError('Retained narration requires published English Heart')
    for name in CATALOGS:
        original = catalog_lesson(baseline / 'web/catalog' / name, identity)
        published = next(row for row in catalogs[name]['lessons'] if row['id'] == identity)
        staged = catalog_lesson(stage / 'catalog' / name, identity)
        require_unchanged_lesson(original, published, staged)
    directory = stage / 'production' / identity / 'audio'
    for suffix in ('.m4a', '.json'):
        actual = {path.relative_to(directory).as_posix() for path in directory.rglob('*' + suffix)}
        if actual != {f'{language}/{voice}{suffix}' for language, voice in expected}:
            raise ValueError('Retained staged audio/timing inventory differs from published voices')
    records = {row['path']: row for row in manifest['files']}
    baseline_tracks = {tuple(Path(path).parts[-2:]) for path in records
                       if path.startswith(f'media_host/{identity}/audio/') and path.endswith('.m4a')}
    if baseline_tracks != {(lang, voice + '.m4a') for lang, voice in expected}:
        raise ValueError('Retained manifest inventory differs from published voices')
    checked = []
    for language, voice in sorted(expected):
        hashes = {}
        for suffix, key in (('.m4a', 'audio_sha256'), ('.json', 'timing_sha256')):
            relative = Path('media_host') / identity / 'audio' / language / (voice + suffix)
            record = records.get(relative.as_posix())
            source = baseline / relative
            staged = directory / language / (voice + suffix)
            if (not record or source.stat().st_size != record['bytes']
                    or staged.stat().st_size != record['bytes']
                    or digest(source) != record['sha256'] or digest(staged) != record['sha256']):
                raise ValueError('Retained narration bytes differ from verified baseline')
            hashes[key] = record['sha256']
        path = directory / language / (voice + '.m4a')
        localized = catalog_lesson(baseline / 'web/catalog' / f'lessons_{language}.json', identity)
        require_timing(read(path.with_suffix('.json')), localized, language, voice, hashes['audio_sha256'])
        errors = audio.check_track(path)
        if errors:
            raise ValueError(f'{identity}/{language}/{voice}: {errors}')
        checked.append(dict(language=language, voice=voice, **hashes))
    return deepcopy(declared), checked


def selected_catalogs(published, lessons, voices, reviews, refresh_ids, retained_ids, current_hosts):
    """Leave retained lesson objects intact while updating other selected lessons."""
    selected = [lesson for lesson in lessons if lesson['id'] not in retained_ids]
    if not selected:
        return deepcopy(published), []
    return update_catalogs(published, selected, voices, reviews,
                           [identity for identity in refresh_ids if identity not in retained_ids],
                           current_hosts=current_hosts)


def update_catalogs(published, lessons, voices, reviews, refresh_ids, *, current_hosts=None):
    """Append new lessons and refresh selected existing lessons in one release."""
    identities = [lesson['id'] for lesson in lessons]
    if (not identities or len(identities) != len(set(identities))
            or len(refresh_ids) != len(set(refresh_ids))
            or set(refresh_ids) - set(identities)):
        raise ValueError('Select unique lessons and a valid refresh subset')
    baseline_ids = {row['id'] for row in published['lessons_en.json']['lessons']}
    if set(refresh_ids) - baseline_ids:
        raise ValueError('Refresh only lessons from the published baseline')
    result, compatibility = deepcopy(published), []
    for replace in (False, True):
        selected = [lesson for lesson in lessons if (lesson['id'] in refresh_ids) == replace]
        if selected:
            result, records = append_catalogs(result, selected, voices, reviews, replace=replace,
                                               current_hosts=current_hosts)
            compatibility.extend(records)
    return result, compatibility


def complete_translation_compatibility(catalogs, updated, previous=()):
    """Retain prior lesson reviews and English fallbacks across publication batches."""
    english = {row['id']: row for row in catalogs['lessons_en.json']['lessons']}
    old = {(row['lesson'], row['language']): row for row in previous}
    languages = {filename.split('_', 1)[1].removesuffix('.json')
                 for filename in CATALOGS if filename in catalogs}
    records = {key: deepcopy(row) for key, row in old.items()
               if key[0] in english and key[1] in languages}
    for filename in CATALOGS:
        language = filename.split('_', 1)[1].removesuffix('.json')
        if language == 'en':
            continue
        for lesson in catalogs[filename]['lessons']:
            identity = lesson['id']
            if (lesson.get('status') != 'coming_soon' and lesson.get('scenes')
                    and lesson == english[identity]):
                key = identity, language
                prior = old.get(key, {})
                records[key] = dict(lesson=identity, language=language,
                    status='english_fallback', reason=prior.get('reason') or
                    'English fallback retained from the published catalog; translation requires review')
    for row in updated:
        records[row['lesson'], row['language']] = deepcopy(row)
    return [records[key] for key in sorted(records)]


def append_javascript_catalog(published, english, count, *, replacements=()):
    """Retain historical JavaScript objects independently of JSON catalogs."""
    previous = published['lessons']
    baseline = english['lessons'] if replacements else english['lessons'][:-count]
    if [row['id'] for row in previous] != [row['id'] for row in baseline]:
        raise ValueError('JavaScript and JSON baseline lesson identities differ')
    if replacements and (len(replacements) != len(set(replacements))
                         or set(replacements) - {row['id'] for row in previous}):
        raise ValueError('Refresh only unique existing JavaScript lesson identities')
    result = deepcopy(published)
    selected = [row for row in english['lessons'] if row['id'] in replacements] if replacements else english['lessons'][-count:]
    for original in selected:
        lesson = deepcopy(original)
        lesson['poster'] = f'{lesson["id"]}/poster.jpg'
        lesson['silent'] = f'{lesson["id"]}/video/{lesson["id"]}_silent.mp4'
        if replacements:
            position = next(i for i, row in enumerate(previous) if row['id'] == lesson['id'])
            result['lessons'][position] = lesson
        else:
            result['lessons'].append(lesson)
    return result


def require_baseline_receipt(manifest, receipt, manifest_hash):
    """Require readback coverage of the exact immutable baseline media set."""
    count = sum(record['path'].startswith('media_host/') for record in manifest['files'])
    checked = receipt.get('readback', {})
    commit = receipt.get('commit', '')
    if (not count or receipt.get('manifest_sha256') != manifest_hash
            or not re.fullmatch(r'[0-9a-f]{40}', commit)
            or receipt.get('media_root') != 'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + commit
            or not receipt.get('tag') or receipt.get('media_files') != count
            or checked.get('commit') != commit or checked.get('passed') is not True
            or any(checked.get(key) != count for key in
                   ('files_expected', 'metadata_matched', 'downloaded_sha256_matched'))
            or checked.get('metadata_failures') != [] or checked.get('download_failures') != []):
        raise ValueError('Baseline readback does not cover its exact immutable media set')


def require_no_new_route_gaps(before, after):
    """Keep existing tutorial debt visible while rejecting newly lost routes."""
    old = {item['app_key'] for item in before['missing_tutorials']}
    new = {item['app_key'] for item in after['missing_tutorials']}
    if not new <= old:
        raise ValueError('Appending lessons introduced missing module routes')


def local_web_path(identity):
    """A web copy's path in the Pages tree."""
    return f'production/{identity}/video/{identity}_silent.mp4'


def copy_preserved_web(published, baseline, root, manifest, *, replacements=(), hosted=()):
    """Retain verified media; ignore leftover production files in the Pages tree.

    ``hosted`` lessons keep their posters here; their web copy moves to the
    media host (see :func:`migrate_web_copy`).
    """
    records = []
    baseline_web = {record['path'][len('web/'):]: record for record in manifest['files']
                    if record['path'].startswith('web/')}
    for path in sorted(published.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(published)
        if relative.parts[0] != 'production':
            copy_checked(path, root / 'web' / relative, records, root)
    for name, record in sorted(baseline_web.items()):
        relative = Path(name)
        if (relative.parts[0] != 'production' or relative.parts[1] in replacements
                or (relative.parts[1] in hosted and name == local_web_path(relative.parts[1]))):
            continue
        copy_checked(baseline / 'web' / relative, root / 'web' / relative,
                     records, root, record['sha256'])
    return records


def migrate_web_copy(baseline, root, manifest, identity, records):
    """Move a preserved lesson's verified web copy onto the media host, byte for byte."""
    record = next((row for row in manifest['files']
                   if row['path'] == 'web/' + local_web_path(identity)), None)
    if record is None:
        raise ValueError(f'No verified local web copy to host: {identity}')
    copy_checked(baseline / 'web' / local_web_path(identity),
                 root / 'media_host' / hosted_web_path(identity), records, root, record['sha256'])


def mark_hosted_web(js_catalog, identities):
    """Name the hosted web copy in each selected lesson_catalog.js object."""
    result = deepcopy(js_catalog)
    known = {lesson['id']: lesson for lesson in result['lessons']}
    if set(identities) - set(known):
        raise ValueError(f'Cannot host web copies of unknown lessons: {sorted(set(identities) - set(known))}')
    for identity in identities:
        known[identity]['web'] = hosted_web_path(identity)
    return result


def ensure_web_root(index):
    """Give the player one data-web-root on the same revision as the 4K masters."""
    if 'data-web-root=' in index:
        return index
    match = re.search(r'data-video4k-root="([^"]+)"', index)
    if not match:
        raise ValueError('Expected a 4K media root to place the web root beside')
    return index[:match.end()] + f'\n      data-web-root="{match.group(1)}"' + index[match.end():]


def synchronize_links(catalogs, lessons):
    """Update chapter links only after proving all canonical prose is unchanged.

    Localized prose, media declarations and timing stay as published. Removing
    link fields from the comparison permits new destinations without requiring
    a new recording of otherwise identical narration.
    """
    result = deepcopy(catalogs)
    english = {row['id']: row for row in catalogs['lessons_en.json']['lessons']}
    for source in lessons:
        identity = source['id']
        old = english[identity]
        before = {key: deepcopy(old.get(key)) for key in source}
        after = deepcopy(source)
        for lesson in (before, after):
            for scene in lesson['scenes']:
                scene.pop('related_lessons', None)
        if before != after:
            raise ValueError(f'Link-only update changes lesson content: {identity}')
        for catalog in result.values():
            target = next(row for row in catalog['lessons'] if row['id'] == identity)
            if len(target['scenes']) != len(source['scenes']):
                raise ValueError(f'Localized chapter count differs: {identity}')
            for scene, current in zip(target['scenes'], source['scenes']):
                scene.pop('related_lessons', None)
                if 'related_lessons' in current:
                    scene['related_lessons'] = deepcopy(current['related_lessons'])
    return result


def checked_web_input(stage, identity):
    """Select current-byte playback evidence, preferring sentence-cue checks."""
    rendition = stage / 'web-renditions' / identity
    video = rendition / 'video' / f'{identity}_silent.mp4'
    proof = read(rendition / 'rendition-checks.json')
    actual_hash = digest(video)
    browser_root = stage / 'browser-web' / identity
    browser_path = browser_root / 'en-af_heart/playback-checks.json'
    sentence_path = browser_root / 'en-af_heart-sentence-cues/playback-checks.json'
    if sentence_path.exists():
        sentence = read(sentence_path)
        if sentence.get('checked_web_rendition', {}).get('sha256') == actual_hash:
            browser_path = sentence_path
    require_web_receipt(identity, proof, read(browser_path), actual_hash)
    return video, proof, browser_path


def build(stage, baseline, identities, *, replace=False, refresh_ids=(), link_ids=(),
          host_web=False, migrate_web=(), retain_narration=(), withdraw=()):
    """Create a new private candidate; never upload or modify the published tree.

    ``withdraw`` unlists published lessons: they leave every catalog, the
    player and the candidate's media; earlier revisions keep their files.

    ``host_web`` puts the selected lessons' web copies on the media host;
    ``migrate_web`` moves preserved lessons' verified web copies there too.
    ``retain_narration`` requires byte-identical baseline audio, timings and
    lesson objects for explicit visual refreshes, without a runtime freshness claim.
    """
    stage, baseline = Path(stage).resolve(), Path(baseline).resolve()
    if replace and refresh_ids:
        raise ValueError('Use either replace-existing or a refresh subset')
    refresh_ids = list(identities) if replace else list(refresh_ids)
    identities = list(identities) if replace else [*identities, *refresh_ids]
    retained_ids = list(retain_narration)
    if (len(retained_ids) != len(set(retained_ids))
            or set(retained_ids) - set(refresh_ids)):
        raise ValueError('Retain narration only for unique, explicitly refreshed lessons')
    validate(baseline, include_hosted_media=True, require_browser=True)
    previous = read(baseline / 'release-manifest.json')
    receipt = read(baseline / 'publication-receipt.json')
    require_baseline_receipt(previous, receipt, digest(baseline / 'release-manifest.json'))
    published = REPO / 'docs/source/_extra/tutorials'
    index = (published / 'index.html').read_text()
    roots = re.findall(r'data-(?:audio|video4k)-root="([^"]+)"', index)
    if len(roots) != 2 or set(roots) != {receipt['media_root']}:
        raise ValueError('Baseline media revision is not the currently published revision')
    catalogs = {name: read(published / 'catalog' / name) for name in CATALOGS}
    original_catalogs = deepcopy(catalogs)
    for name in CATALOGS:
        if catalogs[name] != read(baseline / 'web/catalog' / name):
            raise ValueError('Published lesson sources differ from the verified media baseline')
    withdraw = list(withdraw)
    listed = {lesson['id'] for lesson in catalogs['lessons_en.json']['lessons']}
    if (len(withdraw) != len(set(withdraw)) or set(withdraw) - listed
            or set(withdraw) & set(identities)):
        raise ValueError('Withdraw only unique, published lessons that are not being refreshed')
    withdrawn_keys = {lesson.get('app_key') for lesson in catalogs['lessons_en.json']['lessons']
                      if lesson['id'] in withdraw} - {None}
    catalogs = {name: {**catalog, 'lessons': [lesson for lesson in catalog['lessons']
                                              if lesson['id'] not in withdraw]}
                for name, catalog in catalogs.items()}
    from lesson_redirects import INSTALL_ID, INSTALL_ARCHIVES, installation_placeholder
    installation_merge = INSTALL_ID in identities
    if installation_merge:
        if not set(INSTALL_ARCHIVES) <= set(withdraw) or INSTALL_ID not in refresh_ids:
            raise ValueError('Merged installation requires both archival withdrawals and explicit refresh')
        catalogs = {name: installation_placeholder(catalog, original_catalogs[name])
                    for name, catalog in catalogs.items()}
    # Routes of the remaining published lessons, after any withdrawal.
    previous_navigation = navigation(catalogs['lessons_en.json'])
    current_hosts = {identity: route.get('host_app_key')
                     for identity, route in previous_navigation['routes'].items()}
    if (len(identities) != len(set(identities))
            or any(not re.fullmatch(r'[0-9]+_[a-z0-9_]+', identity) for identity in identities)):
        raise ValueError('Duplicate or invalid appended lesson identity')
    lessons = [read(REPO / 'tools/tutorials/lessons' / (identity + '.json')) for identity in identities]
    reviews = {}
    for lesson in lessons:
        if read(stage / 'production' / lesson['id'] / 'lesson.en.json') != lesson:
            raise ValueError('Staged English differs from the current canonical lesson')
        for filename in CATALOGS:
            language = filename.split('_', 1)[1].removesuffix('.json')
            path = REPO / 'tools/tutorials/lessons/reviews' / f'{lesson["id"]}.{language}.json'
            if path.exists():
                reviews[lesson['id'], language] = read(path)
    planned = {lesson['id']: {'en': ['af_heart']} for lesson in lessons}
    provisional, _ = selected_catalogs(catalogs, lessons, planned, reviews, refresh_ids,
                                      retained_ids, current_hosts)
    checks, voices = {}, {}
    for lesson in lessons:
        identity = lesson['id']
        if identity in retained_ids:
            voices[identity], checks[identity] = verify_retained_tracks(
                stage, baseline, lesson, catalogs, previous)
            label = 'byte-identical published audio tracks verified'
        else:
            voices[identity], checks[identity] = verify_tracks(stage, lesson, provisional)
            label = 'current audio tracks verified'
        print(identity, len(checks[identity]), label, flush=True)
    catalogs, compatibility = selected_catalogs(catalogs, lessons, voices, reviews, refresh_ids,
                                               retained_ids, current_hosts)
    if len(link_ids) != len(set(link_ids)) or set(link_ids) & set(identities):
        raise ValueError('Link updates must select unique, otherwise preserved lessons')
    link_lessons = [read(REPO / 'tools/tutorials/lessons' / (identity + '.json'))
                    for identity in link_ids]
    catalogs = synchronize_links(catalogs, link_lessons)
    compatibility = complete_translation_compatibility(
        catalogs, compatibility,
        read(baseline / 'web/translation-compatibility.json').get('entries', []))
    migrate_web = list(migrate_web)
    if len(migrate_web) != len(set(migrate_web)) or set(migrate_web) & set(identities):
        raise ValueError('Migrate only unique, otherwise preserved lessons')
    # Reject stale playback evidence before allocating or copying the library.
    web_inputs = {lesson['id']: checked_web_input(stage, lesson['id']) for lesson in lessons}
    root = Path(tempfile.mkdtemp(prefix='release-candidate-append-', dir=stage))
    records = copy_preserved_web(published, baseline, root, previous,
                                 replacements=[*refresh_ids, *withdraw], hosted=migrate_web)
    for identity in migrate_web:
        migrate_web_copy(baseline, root, previous, identity, records)
    web_checks = []
    for record in previous['files']:
        if record['path'].startswith('media_host/'):
            if Path(record['path']).parts[1] in (*refresh_ids, *withdraw):
                continue
            copy_checked(baseline / record['path'], root / record['path'], records, root, record['sha256'])
    for lesson in lessons:
        identity = lesson['id']
        source = stage / 'production' / identity
        from native_live_timing import checked_native_timing
        native_timings = checked_native_timing(stage, identity)
        if native_timings is not None:
            copy_checked(native_timings, root / 'media_host' / identity / 'video' / native_timings.name,
                         records, root, digest(native_timings))
        video, proof, browser_path = web_inputs[identity]
        copy_checked(source / 'video' / video.name, root / 'media_host' / identity / 'video' / video.name,
                     records, root, proof['master_sha256'])
        web_target = (root / 'media_host' / hosted_web_path(identity) if host_web
                      else root / 'web/production' / identity / 'video' / video.name)
        copy_checked(video, web_target, records, root, proof['rendition_sha256'])
        copy_checked(source / 'poster.jpg', root / 'web/production' / identity / 'poster.jpg', records, root)
        for track in checks[identity]:
            for suffix, field in (('.m4a', 'audio_sha256'), ('.json', 'timing_sha256')):
                relative = Path('audio') / track['language'] / (track['voice'] + suffix)
                copy_checked(source / relative, root / 'media_host' / identity / relative,
                             records, root, track[field])
        web_checks.append(dict(lesson=identity, rendition_sha256=proof['rendition_sha256'],
                               browser_report_sha256=digest(browser_path)))
    for name, catalog in catalogs.items():
        write(root / 'web/catalog' / name, catalog)
    js_catalog = parse_javascript((published / 'lesson_catalog.js').read_text())
    original_js_catalog = deepcopy(js_catalog)
    js_catalog = {**js_catalog, 'lessons': [lesson for lesson in js_catalog['lessons']
                                            if lesson['id'] not in withdraw]}
    if installation_merge:
        js_catalog = installation_placeholder(js_catalog, original_js_catalog)
    appended = [identity for identity in identities if identity not in refresh_ids]
    if appended:
        js_catalog = append_javascript_catalog(js_catalog, catalogs['lessons_en.json'], len(appended))
    if refresh_ids:
        js_catalog = append_javascript_catalog(js_catalog, catalogs['lessons_en.json'],
                                               len(refresh_ids), replacements=refresh_ids)
    for lesson in js_catalog['lessons']:
        if lesson['id'] in identities:
            from native_live_timing import checked_native_timing
            native_timings = checked_native_timing(stage, lesson['id'])
            if native_timings is not None:
                lesson['visual_timings'] = f"{lesson['id']}/video/{native_timings.name}"
    if link_lessons:
        js_catalog = synchronize_links({'lessons_en.json': js_catalog}, link_lessons)['lessons_en.json']
    js_catalog = mark_hosted_web(js_catalog, [*(identities if host_web else ()), *migrate_web])
    nav = navigation(catalogs['lessons_en.json'])
    for lesson in lessons:
        host = lesson.get('host_app_key')
        if host and nav['routes'].get(lesson['id'], {}).get('host_app_key') != host:
            raise ValueError(f'Lesson host differs from current GUI: {lesson["id"]}')
    require_no_new_route_gaps(previous_navigation, nav)
    if withdrawn_keys & {item['app_key'] for item in nav['missing_tutorials']}:
        raise ValueError('Withdrawing a lesson may not leave a visible module without a tutorial')
    for name, variable, data in [('lesson_catalog.js', 'SPACR_LESSON_CATALOG', js_catalog),
                                 ('module_navigation.js', 'SPACR_TUTORIAL_NAVIGATION', nav)]:
        (root / 'web' / name).write_text('"use strict";\nwindow.' + variable + ' = Object.freeze('
                                       + json.dumps(data, ensure_ascii=False) + ');\n')
    (root / 'web/app_v2.js').write_bytes((REPO / 'tools/tutorials/authoring/web/app_v2.js').read_bytes())
    index = ensure_web_root(index)
    for attribute in ('audio', 'video4k', 'web'):
        index, count = re.subn(rf'data-{attribute}-root="[^"]+"', f'data-{attribute}-root="../media_host"', index)
        if count != 1:
            raise ValueError('Expected one media root per type')
    (root / 'web/index.html').write_text(index)
    write(root / 'web/translation-compatibility.json', dict(policy='report-only; English fallback', entries=compatibility))
    records = [record for record in records if record['path'].startswith('media_host/')]
    records.extend(dict(path=path.relative_to(root).as_posix(), sha256=digest(path), bytes=path.stat().st_size)
                   for path in sorted((root / 'web').rglob('*')) if path.is_file())
    hosted_web = require_consistent_web_hosting(records, js_catalog['lessons'])
    web_bytes = sum(record['bytes'] for record in records if record['path'].startswith('web/'))
    if web_bytes > previous['ceiling_bytes']:
        raise ValueError('Candidate exceeds the existing web media budget')
    english = catalogs['lessons_en.json']['lessons']
    held = [lesson['id'] for lesson in english if lesson.get('status') == 'coming_soon']
    report = dict(scope='Private incremental release; incomplete voices and translations remain tracked',
                  ready_lessons=len(english)-len(held), coming_soon=held, routes=len(english),
                  catalog_languages=len(CATALOGS), narration_tracks=sum(
                      r['path'].startswith('media_host/') and r['path'].endswith('.m4a') for r in records),
                  web_bytes=web_bytes, ceiling_bytes=previous['ceiling_bytes'],
                  media_host_bytes=sum(r['bytes'] for r in records if r['path'].startswith('media_host/')),
                  files=sorted(records, key=lambda record: record['path']), web_checks=web_checks,
                  baseline_manifest_sha256=digest(baseline / 'release-manifest.json'),
                  preserved_lessons=len(english)-len(lessons),
                  appended_lessons=appended,
                  refreshed_lessons=refresh_ids,
                  withdrawn_lessons=withdraw,
                  retained_narration={identity: dict(track_count=len(checks[identity]),
                      resynthesized=False, current_runtime_freshness_claimed=False,
                      baseline_manifest_sha256=digest(baseline / 'release-manifest.json'))
                      for identity in retained_ids},
                  link_only_updates=list(link_ids),
                  hosted_web_lessons=hosted_web,
                  outstanding_module_tutorials=nav['missing_tutorials'],
                  new_lesson_tracks=checks, translation_incompatibilities=compatibility,
                  all_workflows_demonstrated=False, native_speaker_signoff=False,
                  human_listening_signoff=False, release_hold=True, uploaded=False, published=False)
    write(root / 'release-manifest.json', report)
    validate(root, include_hosted_media=True)
    print('CANDIDATE', root, flush=True)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--lesson', action='append', required=True)
    parser.add_argument('--replace-existing', action='store_true',
                        help='Refresh only the selected existing lessons; keep all other lesson media and prose')
    parser.add_argument('--refresh-lesson', action='append', default=[],
                        help='Refresh an existing lesson alongside the new lessons being appended')
    parser.add_argument('--retain-narration', action='append', default=[],
                        help='Refresh visuals with exact verified baseline narration and lesson objects')
    parser.add_argument('--host-web', action='store_true',
                        help='Put the selected lessons\' web copies on the media host, not in the Pages tree')
    parser.add_argument('--migrate-web', action='append', default=[],
                        help='Move a preserved lesson\'s verified web copy onto the media host')
    parser.add_argument('--refresh-links', action='append', default=[],
                        help='Update only chapter destinations; require unchanged prose and preserve media')
    parser.add_argument('--withdraw', action='append', default=[],
                        help='Unlist a published lesson from the catalogs, player and candidate media')
    args = parser.parse_args()
    build(args.stage, args.baseline, args.lesson, replace=args.replace_existing,
          refresh_ids=args.refresh_lesson, link_ids=args.refresh_links,
          host_web=args.host_web, migrate_web=args.migrate_web, retain_narration=args.retain_narration,
          withdraw=args.withdraw)
