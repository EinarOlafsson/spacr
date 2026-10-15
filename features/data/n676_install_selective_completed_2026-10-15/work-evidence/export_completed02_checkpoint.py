#!/usr/bin/env python3
"""Export the recorded selective02 checkpoint only after actual publication checks.

Source preparation only until root supplies real terminal evidence and launches.
No publishing, synthesis, install, media decode or native capture is performed.
"""
from pathlib import Path, PurePosixPath
import sys

ROOT = Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/tutorials')]

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
BASELINE = Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-23-r60-20261015/release-candidate-append-7ew0avek')
BASELINE_SHA = '8065f31b9a758cd0a823efbeb8e631cbc740c1c88ca0ea21d7a41e08eab2dcd4'
PREIMAGE = '972ac89c91db8b3842cd75bb30991b8b1af35898'
LESSON = '02_install_spacr'
LANGUAGES = ('da', 'de', 'es', 'fr', 'hi', 'is', 'it', 'ja', 'ko', 'nb', 'pt-BR', 'sv', 'zh-CN')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def checked(record, expected=None):
    path = Path(record['path'])
    require(path.is_absolute() and record == binding(path), 'Evidence SHA/size/path changed')
    if expected is not None:
        require(path.resolve() == Path(expected).resolve(), 'Evidence names a different artifact')
    return path.resolve()


def git(*args, raw=False):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=not raw)


def safe_relative(value):
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and '..' not in path.parts
            and path.as_posix() == value, 'Unsafe or non-normalized checkpoint path')
    return path


def unique_records(records):
    result = {row['path']: row for row in records}
    require(len(result) == len(records), 'Duplicate manifest paths')
    for row in records:
        safe_relative(row['path'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--hf-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component-decision', type=Path, required=True)
    parser.add_argument('--card-acceptance', type=Path, required=True)
    parser.add_argument('--visual-review', type=Path, required=True)
    parser.add_argument('--source-tests', type=Path, action='append', required=True,
                        help='Owning source-test receipts in chronological order; repeat for supplements')
    parser.add_argument('--source-test-count', type=int, required=True)
    parser.add_argument('--source-peer', type=Path, action='append', required=True)
    args = parser.parse_args()
    stage, candidate, output = args.stage.resolve(), args.candidate.resolve(), args.output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), 'Use a new private WORK export')
    require(candidate.is_relative_to(stage) and candidate != stage, 'Candidate must be inside exact stage')
    source = ROOT / 'tools/tutorials/release_candidate'
    lesson_root = stage / 'production' / LESSON
    require(re.fullmatch('[0-9a-f]{40}', args.source_commit), 'Use exact committed source K')
    require(git('rev-parse', 'HEAD').strip() == args.source_commit, 'Current source is not K')
    require(git('rev-parse', args.source_commit + ':spacr') == git('rev-parse', PREIMAGE + ':spacr'),
            'Application source changed from the accepted preimage')
    dirty = git('status', '--porcelain', '--', 'spacr', 'tests', 'tools/tutorials').splitlines()
    require(not [line for line in dirty if not line[3:].startswith('tools/tutorials/release_candidate/')],
            'Source K has dirty or untracked implementation inputs')
    proof_paths = [args.component_decision, args.card_acceptance, args.visual_review,
                   *args.source_tests, *args.source_peer]
    declared_rows, test_records = {}, []
    for test_path in args.source_tests:
        tests = read(test_path)
        count = tests['tests']['passed']
        require(type(count) is int and count > 0 and tests['tests']['terminal_exit_status'] == 0,
                'Actual source test count/exit must match')
        log_path = checked(tests['tests']['log'])
        text = log_path.read_text()
        require(re.search(r'\b' + str(count) + r' passed\b', text)
                and text.rstrip().endswith('Terminal exit status: 0'), 'Missing actual terminal test PASS')
        require(not re.search(r'\b(?:FAILED|ERROR|INTERRUPTED)\b', text), 'Test log contains failure')
        proof_paths.append(log_path)
        test_records.append(dict(receipt=binding(test_path), log=binding(log_path), passed=count))
        rows = [*tests['files'], *tests.get('unchanged_previous_files', [])]
        if tests.get('renderer_owner_file'):
            rows.append(tests['renderer_owner_file'])
        for row in rows:
            declared_rows[row['path']] = row
    require(args.source_test_count == test_records[-1]['passed'], 'Final supplied source test count differs')
    declared = []
    for row in declared_rows.values():
        path = checked(row)
        require(path.is_relative_to(ROOT), 'Source-test file escaped ROOT')
        relative = path.relative_to(ROOT).as_posix()
        require(path.read_bytes() == git('show', args.source_commit + ':' + relative, raw=True),
                'Actual source-test bytes are not committed K')
        declared.append(relative)
    changes = git('diff', '--name-only', PREIMAGE, args.source_commit).splitlines()
    require(set(changes) <= set(declared) and set(changes), 'K includes an undeclared source change')
    for peer_path in args.source_peer:
        peer = read(peer_path)
        require(peer.get('status') == 'pass' and peer.get('source_peer_passed') is True
                and peer.get('blocking_findings') == [], 'Source peer is not a bounded PASS')
    latest_peer = read(args.source_peer[-1])
    peer_owner = latest_peer.get('owning_evidence', latest_peer.get('owner_evidence'))
    require(peer_owner == binding(args.source_tests[-1]), 'Final source peer does not bind final owning tests')
    for row in [*latest_peer['files'], *latest_peer.get('unchanged_prior_files', [])]:
        checked(row)
    english_path = ROOT / 'tools/tutorials/lessons' / (LESSON + '.json')
    review_paths = [ROOT / 'tools/tutorials/lessons/reviews' / f'{LESSON}.{lang}.json' for lang in LANGUAGES]
    for path in [english_path, *review_paths]:
        relative = path.relative_to(ROOT).as_posix()
        require(path.read_bytes() == git('show', PREIMAGE + ':' + relative, raw=True),
                'Canonical English or official reviews changed')
    english = read(english_path)
    normalized = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for language, path in zip(LANGUAGES, review_paths):
        review = read(path)
        require(review['lesson'] == LESSON and review['language'] == language
                and review['english_sha256'] == normalized, 'Official review source binding changed')
    decision, card, visual = map(read, (args.component_decision, args.card_acceptance, args.visual_review))
    require(decision['lesson'] == LESSON and decision['repair']['scene'] == 7
            and decision['repair']['first_frame'] == 4420 and decision['repair']['frame_count'] == 692,
            'Component authorization names a different repair')
    for row in decision['component_assessments'].values():
        proof_paths.append(checked(row))
    require(card['accepted'] is True and card['lesson'] == LESSON and card['scene'] == 7
            and card['reference_label_visible'] is True and card['terminal_output_fabricated'] is False
            and card['installation_or_environment_commands_executed'] is False, 'Actual reference card was not accepted')
    for field in ('image', 'render_log', 'generator_provenance', 'generator_source'):
        proof_paths.append(checked(card[field]))
    from retained_visual_refresh import checked_retained_visual_timing
    timing_path = checked_retained_visual_timing(stage, LESSON, BASELINE)
    retained_path = lesson_root / 'retained-visual-refresh.json'
    retained, timing = read(retained_path), read(timing_path)
    require(retained['replacements'][0]['scene'] == 7
            and retained['replacements'][0]['first_frame'] == 4420
            and retained['replacements'][0]['frame_count'] == 692
            and retained['replacements'][0]['new_image_sha256'] == card['image']['sha256'],
            'Retained receipt differs from approved scene7 card')
    require(timing['total_duration'] == 430.03333333333336 and round(timing['total_duration'] * 30) == 12901
            and timing['scenes'][11]['duration'] == 29.533333333333335
            and timing.get('complete_native_clips') is not True, 'Legacy visual selection changed')
    master = checked(retained['new_master'], lesson_root / 'video' / (LESSON + '_silent.mp4'))
    finish_path = lesson_root / 'current-video-acceptance.json'
    finish = read(finish_path)
    require(finish['lesson'] == LESSON and finish['retained_visual_timing'] is True
            and finish['native_fifty_voice_timeline'] is False
            and finish['master_sha256'] == digest(master)
            and finish['timing_sha256'] == digest(timing_path)
            and all(finish[key] is True for key in ('master_full_decode_passed',
                    'master_frame_count_matches_scene_boundaries', 'master_all_presentation_times_checked')),
            'Actual repaired-master finish checks are missing or stale')
    require(visual.get('status') == 'sampled_visual_review_pass'
            and visual.get('lesson') == LESSON, 'Actual root sampled visual review is required')
    checked(visual['master'], master)
    checked(visual['timing'], timing_path)
    final_checks_path = lesson_root / 'final-artifact-checks.json'
    final_checks = read(final_checks_path)
    require(final_checks['lesson'] == LESSON and final_checks['passed'] is True
            and final_checks['unique_final_tracks'] == len(final_checks['tracks']) == 50
            and final_checks['master_sha256'] == digest(master)
            and final_checks['canonical_english_sha256'] == normalized,
            'Actual retained final-artifact reconciliation is missing or stale')
    narration = final_checks['retained_narration']
    require(narration['resynthesized'] is False and narration['current_runtime_freshness_claimed'] is False
            and len(narration['tracks']) == 50, 'Final checks do not declare truthful retained narration')
    for track in final_checks['tracks']:
        path = lesson_root / 'audio' / track['language'] / (track['voice'] + '.m4a')
        require(digest(path) == track['sha256'], 'Final reconciled track hash changed')
    for row in final_checks['browser_reports']:
        path = stage / 'browser' / LESSON / row['case'] / 'playback-checks.json'
        require(digest(path) == row['report_sha256'], 'Reconciled browser report changed')
        proof_paths.append(path)
    manifest_path = candidate / 'release-manifest.json'
    manifest, baseline = read(manifest_path), read(BASELINE / 'release-manifest.json')
    require(digest(BASELINE / 'release-manifest.json') == BASELINE_SHA
            and manifest['baseline_manifest_sha256'] == BASELINE_SHA, 'Wrong C23 baseline')
    require(manifest['preserved_lessons'] == 83 and manifest['refreshed_lessons'] == [LESSON]
            and manifest['appended_lessons'] == [] and manifest['withdrawn_lessons'] == []
            and manifest['ready_lessons'] == manifest['routes'] == 84, 'Wrong candidate refresh scope')
    old_records, records = unique_records(baseline['files']), unique_records(manifest['files'])
    for path, relative in ((master, f'media_host/{LESSON}/video/{LESSON}_silent.mp4'),
                           (timing_path, f'media_host/{LESSON}/video/native-live-timings.json'),
                           (retained_path, f'media_host/{LESSON}/video/retained-visual-refresh.json')):
        row = records.get(relative)
        require(row is not None and row['bytes'] == path.stat().st_size
                and row['sha256'] == digest(path) == digest(candidate / relative)
                and (candidate / relative).stat().st_size == row['bytes'],
                'Candidate repaired visual/timing/retention bytes differ from actual stage')
    def other_media(mapping):
        return {p: (r['bytes'], r['sha256']) for p, r in mapping.items()
                if p.startswith('media_host/') and not p.startswith('media_host/' + LESSON + '/')}
    require(other_media(records) == other_media(old_records), 'Other83 media records changed')
    audio_prefix = 'media_host/' + LESSON + '/audio/'
    old_audio = {p: r for p, r in old_records.items() if p.startswith(audio_prefix)}
    new_audio = {p: r for p, r in records.items() if p.startswith(audio_prefix)}
    require(len(old_audio) == 100 and old_audio == new_audio, 'All100 retained audio/sidecar records must match')
    for relative, row in old_audio.items():
        path = candidate / relative
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Candidate retained audio bytes changed')
    catalogs = sorted(p.name for p in (BASELINE / 'web/catalog').glob('*.json'))
    require(len(catalogs) == 14 and catalogs == sorted(p.name for p in (candidate / 'web/catalog').glob('*.json')),
            'Expected exact14 catalog filenames')
    for name in catalogs:
        old, new = read(BASELINE / 'web/catalog' / name)['lessons'], read(candidate / 'web/catalog' / name)['lessons']
        require(old == new and len(new) == len({r['id'] for r in new}) == 84,
                'Retained02 and83 other lesson catalog objects must stay exact')
        require(digest(source / 'web/catalog' / name) == digest(candidate / 'web/catalog' / name), 'Checkpoint catalog differs')
    checkpoint = read(source / 'checkpoint.json')
    publication = read(source / 'publication-receipt.json')
    browser = read(source / 'published-media-browser-checks.json')
    manifest_sha = digest(manifest_path)
    require(Path(checkpoint['private_candidate']).resolve() == candidate
            and checkpoint['manifest_sha256'] == publication['manifest_sha256'] == browser['manifest_sha256'] == manifest_sha
            and digest(source / 'release-manifest.json') == manifest_sha, 'Record/checkpoint points at a different candidate')
    require(checkpoint['release_hold'] is False and checkpoint['media_uploaded'] is True
            and checkpoint['pages_tree_ready'] is True and checkpoint['published'] is False,
            'Actual hosted publication record is not complete')
    require(digest(source / 'publication-receipt.json') == digest(candidate / 'publication-receipt.json')
            and digest(source / 'published-media-browser-checks.json') == digest(candidate / 'checks/published-media-browser-checks.json'),
            'Recorded publication or hosted browser receipt differs')
    revision = publication['commit']
    require(re.fullmatch('[0-9a-f]{40}', revision), 'Media revision must be immutable')
    tag = args.hf_name + '-verified'
    media_root = 'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + revision
    rb = publication['readback']
    media = [r for p, r in records.items() if p.startswith('media_host/')]
    require(publication['branch'] == args.hf_name and publication['tag'] == checkpoint['media_revision']['tag'] == tag
            and rb['commit'] == checkpoint['media_revision']['commit'] == revision
            and publication['media_root'] == browser['media_root'] == checkpoint['media_revision']['media_root'] == media_root,
            'HF branch/tag/revision/root identity differs')
    require(rb['passed'] is True and rb['files_expected'] == rb['metadata_matched'] == rb['downloaded_sha256_matched']
            == publication['media_files'] == len(media)
            and rb['bytes_read_back'] == publication['media_bytes'] == manifest['media_host_bytes'] == sum(r['bytes'] for r in media)
            and not rb['metadata_failures'] and not rb['download_failures'], 'Immutable full HF readback did not pass')
    cases = browser['ready_playback_cases']
    ready = {r['id'] for r in read(candidate / 'web/catalog/lessons_en.json')['lessons'] if r.get('status') != 'coming_soon'}
    pages = ROOT / 'docs/source/_extra/tutorials'
    require(browser['passed'] is True and len(cases) == len(ready) == 84
            and {r['lesson'] for r in cases} == ready and all(r['passed'] is True for r in cases)
            and Path(browser['pages_tree']).resolve() == pages.resolve()
            and browser['index_sha256'] == digest(pages / 'index.html'), 'Actual84 hosted Pages cases missing or stale')
    rows = checkpoint['files']
    require(len(rows) == len(unique_records(rows)) == 145, 'Expected actual145 checkpoint files')
    checked_rows = []
    for row in rows:
        rel = safe_relative(row['path'])
        path = source / rel
        require(path.resolve().is_relative_to(source.resolve()) and path.is_file()
                and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Checkpoint file coverage/hash changed')
        checked_rows.append({**row, 'role': 'checkpoint_recorded_file'})
    for name in ('checkpoint.json', 'publication-receipt.json', 'published-media-browser-checks.json'):
        require(name not in {r['path'] for r in checked_rows}, 'Duplicate extra receipt')
        row = binding(source / name)
        checked_rows.append(dict(path=name, bytes=row['bytes'], sha256=row['sha256'], role='bound_record_evidence'))
    proof_paths += [*review_paths, english_path, retained_path, timing_path, finish_path, final_checks_path,
                    checked(retained['render_log']), checked(read(checked(retained['render_log']))['output_log']),
                    checked(retained['new_spec']), checked(card['generator_provenance'])]
    proof_paths = list(dict.fromkeys(Path(p).resolve() for p in proof_paths))
    evidence = [binding(p) for p in proof_paths]
    # No directories or output files are created until every prerequisite above passes.
    output.mkdir()
    snapshot = output / 'checkpoint'
    snapshot.mkdir()
    for row in checked_rows:
        target = snapshot / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row['path'], target)
        require(target.stat().st_size == row['bytes'] and digest(target) == row['sha256']
                and digest(source / row['path']) == row['sha256'], 'Checkpoint source changed during export')
    require(sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file())
            == sorted(r['path'] for r in checked_rows), 'Unexpected snapshot file coverage')
    review_evidence = []
    for index, row in enumerate(evidence):
        target = output / 'review-evidence' / f'{index:02}-{Path(row["path"]).name}'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['path'], target)
        require(binding(Path(row['path'])) == row and digest(target) == row['sha256'], 'Evidence changed during export')
        review_evidence.append(dict(source=row, snapshot=binding(target)))
    require(git('rev-parse', 'HEAD').strip() == args.source_commit, 'Source commit changed during export')
    require(all((source / row['path']).stat().st_size == row['bytes']
                and digest(source / row['path']) == row['sha256'] for row in checked_rows),
            'Recorded checkpoint changed during export')
    report = dict(schema=1, lesson=LESSON, source_commit=args.source_commit,
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        baseline_manifest_sha256=BASELINE_SHA, candidate_root=str(candidate),
        release_manifest_sha256=manifest_sha, snapshot_root=str(snapshot),
        checkpoint_file_count=len(rows), evidence_extra_count=3, snapshot_file_count=len(checked_rows),
        files=checked_rows, review_evidence=review_evidence, translation_review_count=13,
        preserved_other_lessons=83, unchanged_catalog_lesson_objects=14,
        retained_audio_sidecar_files=100, source_test_count=args.source_test_count,
        source_test_receipts=test_records,
        publication_revision=revision, publication_tag=tag, published_media_root=media_root,
        master=binding(master), timing=binding(timing_path),
        checks=dict(record_completed=True, all84_hosted_cases_passed=True,
                    hf_full_readback_sha256_matched=rb['downloaded_sha256_matched'],
                    coherent145_plus3_snapshot_hashes_and_lengths=True),
        claims=dict(fresh_capture=False, complete_native_clips=False, resynthesized=False,
                    current_runtime_audio_freshness=False, human_listening=False, native_speaker_acceptance=False),
        limits=['Exports exact existing record/readback/Pages evidence; no new remote verification or live Space deployment proof.',
                'Only scene7 non-native command reference changed; original430.033333s/12901frames and selected29.533s Home interval retained.',
                'Component and visual assessments retain their stated sample/listening limits. No installation or analysis-science run is claimed.',
                'Generated checkpoint/evidence only; hosted media and original evidence remain in place.'])
    (output / 'export-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(export=str(output), snapshot_files=len(checked_rows), revision=revision)))


if __name__ == '__main__':
    main()
