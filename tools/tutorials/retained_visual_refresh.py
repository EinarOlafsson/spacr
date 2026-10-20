"""Check a bounded visual overlay against an immutable published timeline.

This deliberately preserves legacy clip selections. It neither replans native
coverage nor claims a fresh capture or freshly synthesized narration.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import re
import struct

from audit_staged_catalogs import CATALOGS
from retain_narration import catalog_lesson, digest
from stage_lesson import REPO, read


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def require_binding(record, path=None):
    target = Path(record['path']).resolve()
    if (not Path(record['path']).is_absolute()
            or (path is not None and target != Path(path).resolve())
            or record != binding(target)):
        raise ValueError('Retained visual artifact binding changed')
    return target


def overlay_command(master, card, output, first_frame, frame_count, total_frames, fps):
    """The only supported refresh operation overlays one full-size static card."""
    last = first_frame + frame_count - 1
    overlay = (f"[0:v][1:v]overlay=0:0:enable='between(n,{first_frame},{last})':"
               "eof_action=repeat:shortest=0,format=yuv420p[v]")
    return ['ffmpeg', '-nostdin', '-v', 'warning', '-y', '-progress', 'pipe:1', '-nostats', '-threads', '4',
            '-i', str(Path(master).resolve()), '-loop', '1', '-framerate', str(fps),
            '-i', str(Path(card).resolve()), '-filter_complex_threads', '1',
            '-filter_complex', overlay, '-map', '[v]', '-an', '-c:v', 'libx264',
            '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', str(fps),
            '-frames:v', str(total_frames), '-threads', '4', str(Path(output).resolve())]


def retained_timing(original, spec_hash, receipt_hash, manifest_hash):
    result = deepcopy(original)
    result['visual_spec_sha256'] = spec_hash
    result['retained_visual_refresh'] = dict(receipt_sha256=receipt_hash,
                                           baseline_manifest_sha256=manifest_hash)
    return result


def checked_retained_visual_timing(stage, identity, baseline):
    """Fail closed unless the bounded refresh preserves published media inputs."""
    from build_appended_candidate import require_baseline_receipt

    stage, baseline = Path(stage).resolve(), Path(baseline).resolve()
    if read(stage / 'production' / identity / 'retained-visual-refresh.json').get('schema') == 2:
        return _checked_mixed_refresh(stage, identity, baseline)
    root = stage / 'production' / identity
    receipt_path = root / 'retained-visual-refresh.json'
    receipt = read(receipt_path)
    if (receipt.get('schema') != 1 or receipt.get('kind') != 'retained_visual_refresh'
            or receipt.get('lesson') != identity
            or Path(receipt['baseline_root']).resolve() != baseline):
        raise ValueError('Expected an explicit matching retained visual refresh receipt')
    manifest_path = require_binding(receipt['baseline_manifest'], baseline / 'release-manifest.json')
    publication = require_binding(receipt['publication_receipt'], baseline / 'publication-receipt.json')
    manifest = read(manifest_path)
    require_baseline_receipt(manifest, read(publication), digest(manifest_path))
    records = {row['path']: row for row in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ValueError('Duplicate published manifest paths')

    def published(path, relative):
        row = records.get(relative)
        if not row or (digest(path), path.stat().st_size) != (row['sha256'], row['bytes']):
            raise ValueError('Original visual input differs from published baseline')

    original = Path(receipt['original_root']).resolve()
    old_spec_path = require_binding(receipt['original_spec'], original / 'scenes.json')
    old_timing_path = require_binding(receipt['original_timing'])
    old_master = require_binding(receipt['original_master'])
    published(old_timing_path, f'media_host/{identity}/video/native-live-timings.json')
    published(old_master, f'media_host/{identity}/video/{identity}_silent.mp4')
    spec_path = require_binding(receipt['new_spec'], root / 'scenes.json')
    master = require_binding(receipt['new_master'], root / 'video' / f'{identity}_silent.mp4')
    old_spec, spec, timing = read(old_spec_path), read(spec_path), read(old_timing_path)
    if (timing.get('visual_spec_sha256') != digest(old_spec_path)
            or timing.get('lesson') != identity or timing.get('complete_native_clips') is True
            or timing.get('retained_visual_refresh') is not None
            or old_spec.get('lesson') != identity
            or len(old_spec['scenes']) != len(timing['scenes'])):
        raise ValueError('Expected the exact legacy published specification and timing')
    canonical = read(REPO / 'tools/tutorials/lessons' / f'{identity}.json')
    if (read(root / 'lesson.en.json') != canonical
            or [row['narration'] for row in old_spec['scenes']]
               != [row['narration'] for row in canonical['scenes']]):
        raise ValueError('Retained visual narration differs from current canonical source')
    for name in CATALOGS:
        if (catalog_lesson(stage / 'catalog' / name, identity)
                != catalog_lesson(baseline / 'web/catalog' / name, identity)):
            raise ValueError('Retained visual catalog objects changed')
    audio_prefix = f'media_host/{identity}/audio/'
    audio = {key[len(audio_prefix):]: row for key, row in records.items()
             if key.startswith(audio_prefix)}
    actual = {p.relative_to(root / 'audio').as_posix() for p in (root / 'audio').rglob('*')
              if p.is_file()}
    if not audio or set(audio) != actual:
        raise ValueError('Retained narration inventory changed')
    for relative, row in audio.items():
        for path in (root / 'audio' / relative, baseline / audio_prefix / relative):
            if (digest(path), path.stat().st_size) != (row['sha256'], row['bytes']):
                raise ValueError('Retained narration bytes changed')
    replacements = receipt.get('replacements', [])
    if len(replacements) != 1:
        raise ValueError('A retained refresh supports exactly one non-native command card')
    replacement = replacements[0]
    number = replacement['scene']
    if type(number) is not int or not 1 <= number <= len(timing['scenes']):
        raise ValueError('Invalid replacement scene')
    index = number - 1
    old_scene, clock = old_spec['scenes'][index], timing['scenes'][index]
    if old_scene.get('clip') or clock.get('native_live_video') is not False:
        raise ValueError('A retained refresh cannot replace native footage')
    fps = timing['fps']
    first = round(clock['speech_start'] * fps)
    count = round(clock['duration'] * fps)
    total = round(timing['total_duration'] * fps)
    if (replacement['first_frame'] != first or replacement['frame_count'] != count
            or first < 0 or count <= 0 or first + count > total
            or old_spec.get('fps') != fps or old_spec.get('size') != [3840, 2160]):
        raise ValueError('Replacement frame range differs from published scene clock')
    expected = deepcopy(old_spec)
    new_scene = spec['scenes'][index]
    card = (root / new_scene['image']).resolve()
    card_hash = digest(card)
    with card.open('rb') as stream:
        header = stream.read(24)
    if header[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or struct.unpack('>II', header[16:24]) != (3840, 2160):
        raise ValueError('Replacement must be a full-size PNG command card')
    if (replacement['original_image_sha256'] != old_scene['capture_sha256']
            or replacement['new_image_sha256'] != card_hash
            or new_scene['capture_sha256'] != card_hash):
        raise ValueError('Replacement card hashes differ')
    expected['scenes'][index].update(image=new_scene['image'], capture_sha256=card_hash)
    visual = clock['visual']
    origin = old_spec['capture_source']['frame_origins'][visual]
    new_origin = replacement['new_origin']
    provenance = require_binding(new_origin['generator_provenance'], card.parent / 'provenance.json')
    generator = read(provenance)
    frames = read(card.parent / 'frames.json')
    if (generator.get('author_source_sha256') != digest(REPO / 'tools/tutorials/render_installation_route_cards.py')
            or generator.get('style_source_sha256') != digest(REPO / 'tools/tutorials/authoring/tools/render_install_keyframes.py')
            or frames.get(visual) != dict(image=card.name, sha256=card_hash, buttons=[],
                kind='generated_command_reference_not_terminal_output')
            or replacement['original_origin'] != origin
            or new_origin != dict(kind='reference_guidance', capture=str(card.parent),
                source_visual=visual, image_sha256=card_hash, installed_identity=None,
                representation='generated_command_reference_not_terminal_output',
                generator_provenance=binding(provenance))):
        raise ValueError('Generated command card provenance changed')
    expected['capture_source']['frame_origins'][visual] = new_origin
    expected['capture_source']['completed_capture'] = False
    expected['capture_source']['frame_pixels_modified'] = True
    if spec != expected:
        raise ValueError('Retained visual specification changed outside the approved card')
    for old, new in zip(old_spec['scenes'], spec['scenes']):
        if digest(original / old['image']) != old['capture_sha256']:
            raise ValueError('Original scene source changed')
        if digest(root / new['image']) != new['capture_sha256']:
            raise ValueError('Staged scene source changed')
        if old.get('clip'):
            for key, checksum in (('video', 'sha256'), ('receipt', 'receipt_sha256')):
                if (digest(original / old['clip'][key]) != old['clip'][checksum]
                        or digest(root / new['clip'][key]) != old['clip'][checksum]):
                    raise ValueError('Retained native scene input changed')
    render_path = require_binding(receipt['render_log'])
    render = read(render_path)
    output_log = require_binding(render['output_log'])
    if (not output_log.read_text(encoding="utf-8").rstrip().endswith('Terminal exit status: 0')
            or render.get('returncode') != 0 or render.get('command') != overlay_command(
            old_master, card, master, first, count, total, fps)):
        raise ValueError('Render did not use the bounded published-master overlay')
    timing_path = root / 'video/native-live-timings.json'
    if read(timing_path) != retained_timing(timing, digest(spec_path), digest(receipt_path), digest(manifest_path)):
        raise ValueError('Retained visual timing changed outside its explicit refresh binding')
    return timing_path


def mixed_overlay_command(master, home, card, output, native_frames):
    """Replace old Home/doctor intervals while preserving every new native frame."""
    if type(native_frames) is not int or native_frames < 840:
        raise ValueError('Home needs at least its original narration coverage')
    graph = ("[0:v]split=2[before][after];"
             "[before]trim=start_frame=0:end_frame=5111,setpts=PTS-STARTPTS[prefix];"
             "[1:v]setpts=PTS-STARTPTS[home];"
             "[2:v]trim=start_frame=0:end_frame=879,setpts=PTS-STARTPTS[doctor];"
             "[after]trim=start_frame=6830:end_frame=10007,setpts=PTS-STARTPTS[tail];"
             "[prefix][home][doctor][tail]concat=n=4:v=1:a=0,format=yuv420p[v]")
    return ['ffmpeg', '-nostdin', '-v', 'warning', '-y', '-progress', 'pipe:1', '-nostats',
            '-threads', '4', '-i', str(Path(master).resolve()), '-threads', '4',
            '-i', str(Path(home).resolve()), '-loop', '1', '-framerate', '30',
            '-i', str(Path(card).resolve()), '-filter_complex_threads', '1',
            '-filter_complex', graph, '-map', '[v]', '-an', '-c:v', 'libx264',
            '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', '30',
            '-frames:v', str(9167 + native_frames), '-threads', '4', str(Path(output).resolve())]


def mixed_retained_timing(original, spec_hash, receipt_hash, manifest_hash, video_hash, receipt_video_hash,
                          native_frames):
    if type(native_frames) is not int or native_frames < 840:
        raise ValueError('Home needs at least its original narration coverage')
    result = retained_timing(original, spec_hash, receipt_hash, manifest_hash)
    result['preserve_visual_intervals'] = True
    delta = (native_frames - 840) / 30
    result['scenes'][8]['duration'] = native_frames / 30
    for index, scene in enumerate(result['scenes']):
        if index >= 8:
            scene['speech_end'] += delta
            scene['scene_end'] += delta
        if index >= 9:
            scene['speech_start'] += delta
    result['total_duration'] += delta
    matches = [row for row in result['native_clip_sources'] if row['scene'] == 9]
    if len(matches) != 1:
        raise ValueError('Expected one inherited Home source entry')
    matches[0].update(clip_sha256=video_hash, receipt_sha256=receipt_video_hash)
    return result


def _same_bytes(record, path):
    require_binding(record)
    current = binding(path)
    if any(record[key] != current[key] for key in ('bytes', 'sha256')):
        raise ValueError('Approved visual bytes differ from staged source')


def _checked_home(replacement, image, clip, root):
    from native_live_timing import complete_native_duration
    native = replacement['native']
    source = Path(native['capture_root']).resolve()
    stage = source.parent.parent
    name = '13_current_nightly_home'
    if source.name != 'home' or source.parent.name != 'captures':
        raise ValueError('Expected the actual Home capture directory')
    locations = dict(video=source / (name + '.mp4'), native_video=source / (name + '.native.mp4'),
        receipt=source / (name + '.capture.json'), gate=source / (name + '.gate.json'),
        facts=source / (name + '.host.json'), pointer=source / (name + '.pointer.json'),
        poster=source / (name + '.png'), frames_manifest=source / 'frames.json',
        provenance=source / 'provenance.json', host_run=stage / 'host-run.json',
        cleanup=stage / 'gnome-host/cleanup.json', profile=stage / 'capture-profile.json',
        exact_source=stage / 'exact-source.json')
    paths = {key: require_binding(native[key], path) for key, path in locations.items()}
    evidence = {key: read(paths[key]) for key in ('receipt', 'gate', 'facts', 'frames_manifest',
        'provenance', 'host_run', 'cleanup', 'profile', 'exact_source')}
    receipt, gate = evidence['receipt'], evidence['gate']
    frames = replacement['frame_count']
    if type(frames) is not int or frames < 840:
        raise ValueError('Home needs complete native frames covering its narration')
    frame = evidence['frames_manifest'].get(name, {})
    source_clip = frame.get('clip', {})
    commit = native['source_commit']
    exact, provenance = evidence['exact_source'], evidence['provenance']
    if (not re.fullmatch('[0-9a-f]{40}', commit)
            or exact.get('commit') != commit or provenance.get('commit') != commit
            or exact.get('fresh') is not True or exact.get('private_profile') is not True
            or provenance.get('completed_capture') is not True
            or provenance.get('app_source_modified') is not False
            or provenance.get('module') != 'home' or provenance.get('selected_scenes') != name):
        raise ValueError('Home capture is not fresh and bound to its actual source')
    inventory = exact.get('sha256', {})
    if not inventory:
        raise ValueError('Missing Home app/recorder source inventory')
    for relative, checksum in inventory.items():
        path = Path(relative)
        if (path.is_absolute() or '..' in path.parts or path.as_posix() != relative
                or digest(REPO / path) != checksum):
            raise ValueError('Home app/recorder source inventory changed')
    if (replacement['new_video_sha256'] != native['video']['sha256']
            or replacement['new_receipt_sha256'] != native['receipt']['sha256']
            or source_clip.get('sha256') != native['video']['sha256']
            or source_clip.get('native_sha256') != native['native_video']['sha256']
            or source_clip.get('receipt_sha256') != native['receipt']['sha256']
            or frame.get('sha256') != native['poster']['sha256']
            or frame.get('image') != paths['poster'].name):
        raise ValueError('Home manifest or native/handoff/poster identity differs')
    _same_bytes(native['poster'], image)
    for key, checksum in (('video', 'sha256'), ('receipt', 'receipt_sha256')):
        if digest(root / clip[key]) != source_clip.get(checksum):
            raise ValueError('Staged Home video or receipt differs from actual capture')
    if (set(clip) != {'video', 'sha256', 'receipt', 'receipt_sha256', 'duration'}
            or clip.get('duration') != frames / 30 or clip.get('sha256') != native['video']['sha256']
            or clip.get('receipt_sha256') != native['receipt']['sha256']
            or receipt.get('video') != paths['video'].name
            or receipt.get('sha256') != native['video']['sha256']
            or receipt.get('size') != [3840, 2160] or receipt.get('full_decode_passed') is not True
            or receipt.get('audio_streams') != 0 or receipt.get('target_fps') != 30
            or complete_native_duration(receipt, 30) != frames / 30
            or len(receipt['frames']) != frames
            or receipt.get('native', {}).get('sha256') != native['native_video']['sha256']
            or receipt['native'].get('video') != paths['native_video'].name
            or receipt['native'].get('size') != gate.get('size')
            or gate.get('size') not in ([2560, 1440], [3840, 2160])
            or gate.get('frames') != frames or not math.isclose(float(gate.get('fps', 0)), 30)
            or not math.isclose(float(gate.get('duration_s', 0)), frames / 30, abs_tol=.002)):
        raise ValueError('Home must preserve all actual frames at 30 fps and 1x')
    host = evidence['host_run']
    row = host.get('clips', {}).get(name, {})
    for verdict in (gate, row, source_clip.get('gate', {}), receipt.get('gate', {})):
        if verdict.get('passed') is not True or verdict.get('reasons') != []:
            raise ValueError('Home requires an unchanged strict gate pass')
    if (host.get('app_exit_code') != 0 or host.get('recorder_rc') != 0
            or host.get('failure') is not None or host.get('gate_failed') != []
            or evidence['cleanup'].get('exit_code') != 0
            or evidence['cleanup'].get('survivors_after_cleanup') != []
            or Path(gate.get('video', '')).name != paths['native_video'].name
            or row.get('file') != paths['gate'].name
            or float(gate['limits']['min_fps']) < 27.9 or float(gate['distinct_fps']) < 27.9
            or row.get('distinct_fps') != gate['distinct_fps']):
        raise ValueError('Home host, cleanup or strict smoothness evidence failed')
    for field in ('app_freeze_seconds', 'app_slow_seconds', 'app_unknown_seconds',
                  'app_gui_sampler_gaps_s', 'app_stall_windows_s', 'adapter_bypasses'):
        if gate.get(field) != []:
            raise ValueError('Home has missing or nonempty freeze/slow/unknown/gap/stall/bypass evidence')
    if (evidence['facts'].get('adapter_bypasses') != []
            or evidence['facts'].get('app_gui_sampler_gaps_s') != []
            or receipt.get('adapter_bypasses') != []
            or gate.get('cursor', {}).get('jump_count') != 0
            or gate['cursor'].get('jumps') != [] or float(gate['cursor']['found_fraction']) < .9
            or float(gate['cursor']['max_step_px']) > float(gate['cursor']['bound_px'])
            or gate.get('top_bar', {}).get('hidden') is not True):
        raise ValueError('Home pointer/GUI/top-bar evidence failed')
    for key in ('show_alpha', 'show_alpha_features', 'show_alpha_species', 'spaceout',
                'anonymous_reports_enabled', 'anonymous_report_notice_due'):
        if evidence['profile'].get(key) is not False:
            raise ValueError('Home private profile has a hidden-mode or notice mismatch')
    approval = read(require_binding(replacement['native_visual_acceptance']))
    if (approval.get('accepted') is not True or approval.get('source_commit') != commit
            or approval.get('source_scene') != name or approval.get('actual_native_frames') != frames
            or approval.get('requested_minimum_seconds') != 28):
        raise ValueError('Missing actual Home visual acceptance')
    for key in ('video', 'gate', 'host_run', 'provenance', 'profile', 'cleanup',
                'frames_manifest', 'poster', 'exact_source'):
        if approval.get(key) != native[key]:
            raise ValueError('Home acceptance binds different actual evidence')
    if approval.get('capture_receipt') != native['receipt']:
        raise ValueError('Home acceptance binds a different capture receipt')
    return dict(kind='current_nightly', capture=str(source), source_visual=name,
        image_sha256=digest(image), installed_identity=None, capture_source_commit=commit,
        capture_provenance=native['provenance'], representation='fresh_full_native_clip')


def _checked_mixed_refresh(stage, identity, baseline):
    from build_appended_candidate import require_baseline_receipt
    root = stage / 'production' / identity
    receipt_path = root / 'retained-visual-refresh.json'
    receipt = read(receipt_path)
    if (identity != '04_platform_installers' or receipt.get('schema') != 2
            or receipt.get('kind') != 'retained_visual_refresh' or receipt.get('lesson') != identity
            or Path(receipt['baseline_root']).resolve() != baseline):
        raise ValueError('Schema2 is only the explicit two-scene installer refresh')
    policy = read(require_binding(receipt['policy']))
    if (policy.get('schema') != 1 or policy.get('kind') != 'retained_mixed_visual_refresh_policy'
            or policy.get('lesson') != identity or policy.get('accepted') is not True
            or policy.get('minimum_native_frames') != 840 or policy.get('fps') != 30
            or any(policy.get(key) is not True for key in ('whole_native_required',
                'other_scene_durations_preserved', 'audio_and_catalogs_retained'))
            or policy.get('supersedes_sha256') != '48959f634aff43cbec64c1c3b416e0e97ad4a4ff711bda8537c41d2f3a004e42'):
        raise ValueError('Missing the superseding full-native duration policy')
    manifest_path = require_binding(receipt['baseline_manifest'], baseline / 'release-manifest.json')
    publication = require_binding(receipt['publication_receipt'], baseline / 'publication-receipt.json')
    manifest = read(manifest_path)
    require_baseline_receipt(manifest, read(publication), digest(manifest_path))
    records = {r['path']: r for r in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ValueError('Duplicate published paths')
    original = Path(receipt['original_root']).resolve()
    old_spec_path = require_binding(receipt['original_spec'], original / 'scenes.json')
    old_timing_path = require_binding(receipt['original_timing'])
    old_master = require_binding(receipt['original_master'])
    for path, name in ((old_timing_path, 'native-live-timings.json'), (old_master, identity + '_silent.mp4')):
        record = records[f'media_host/{identity}/video/{name}']
        if (digest(path), path.stat().st_size) != (record['sha256'], record['bytes']):
            raise ValueError('Original mixed visual input differs from published bytes')
    spec_path = require_binding(receipt['new_spec'], root / 'scenes.json')
    master = require_binding(receipt['new_master'], root / 'video' / (identity + '_silent.mp4'))
    old, spec, timing = read(old_spec_path), read(spec_path), read(old_timing_path)
    if (old.get('lesson') != identity or timing.get('lesson') != identity
            or timing.get('visual_spec_sha256') != digest(old_spec_path)
            or timing.get('complete_native_clips') is True or 'retained_visual_refresh' in timing
            or old.get('fps') != 30 or old.get('size') != [3840, 2160]
            or len(old['scenes']) != 14 or len(timing['scenes']) != 14
            or timing['fps'] != 30 or round(timing['total_duration'] * 30) != 10007):
        raise ValueError('Mixed refresh must inherit the exact legacy installer timeline')
    canonical = read(REPO / 'tools/tutorials/lessons' / (identity + '.json'))
    if (read(root / 'lesson.en.json') != canonical
            or [s['narration'] for s in old['scenes']] != [s['narration'] for s in canonical['scenes']]):
        raise ValueError('Mixed refresh narration changed')
    for name in CATALOGS:
        if catalog_lesson(stage / 'catalog' / name, identity) != catalog_lesson(baseline / 'web/catalog' / name, identity):
            raise ValueError('Mixed refresh catalog object changed')
    prefix = f'media_host/{identity}/audio/'
    audio = {path[len(prefix):]: row for path, row in records.items() if path.startswith(prefix)}
    actual = {p.relative_to(root / 'audio').as_posix() for p in (root / 'audio').rglob('*') if p.is_file()}
    if len(audio) != 100 or set(audio) != actual:
        raise ValueError('Mixed refresh must preserve all 100 narration files')
    for rel, row in audio.items():
        for path in (root / 'audio' / rel, baseline / prefix / rel):
            if (digest(path), path.stat().st_size) != (row['sha256'], row['bytes']):
                raise ValueError('Mixed refresh narration bytes changed')
    replacements = receipt.get('replacements', [])
    if [r.get('scene') for r in replacements] != [9, 10]:
        raise ValueError('Mixed refresh requires only scene9 Home and scene10 doctor')
    expected = deepcopy(old)
    home = card = None
    native_frames = replacements[0].get('frame_count')
    if type(native_frames) is not int or native_frames < 840:
        raise ValueError('Home requires all actual native frames covering its old slot')
    for replacement, number, kind, first, count, visual in zip(replacements, (9, 10),
            ('strict_full_native_clip', 'generated_command_reference_not_terminal_output'),
            (5111, 5111 + native_frames), (native_frames, 879), ('13_current_nightly_home', '05_doctor')):
        index = number - 1
        clock, old_scene, new_scene = timing['scenes'][index], old['scenes'][index], spec['scenes'][index]
        if (replacement.get('kind') != kind or replacement.get('first_frame') != first
                or replacement.get('frame_count') != count or clock['visual'] != visual
                or round(clock['speech_start'] * 30) != (5111 if number == 9 else 5951)
                or round(clock['duration'] * 30) != (840 if number == 9 else 879)
                or replacement['original_origin'] != old['capture_source']['frame_origins'][visual]
                or replacement['original_image_sha256'] != old_scene['capture_sha256']):
            raise ValueError('Mixed replacement identity or clock changed')
        image = (root / new_scene['image']).resolve()
        if digest(image) != replacement['new_image_sha256'] or new_scene['capture_sha256'] != digest(image):
            raise ValueError('Mixed replacement poster bytes changed')
        with image.open('rb') as stream:
            header = stream.read(24)
        if header[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or struct.unpack('>II', header[16:24]) != (3840, 2160):
            raise ValueError('Mixed replacement requires a normal full4K source image')
        expected['scenes'][index].update(image=new_scene['image'], capture_sha256=digest(image))
        if number == 9:
            if not old_scene.get('clip') or clock.get('native_live_video') is not True:
                raise ValueError('The Home replacement must replace only its native interval')
            new_origin = _checked_home(replacement, image, new_scene['clip'], root)
            expected['scenes'][index]['clip'] = new_scene['clip']
            home = root / new_scene['clip']['video']
        else:
            if old_scene.get('clip') or clock.get('native_live_video') is not False:
                raise ValueError('Doctor must remain a generated non-native reference')
            card = image
            provenance = require_binding(replacement['new_origin']['generator_provenance'], image.parent / 'provenance.json')
            generator, frames = read(provenance), read(image.parent / 'frames.json')
            if (generator.get('author_source_sha256') != digest(REPO / 'tools/tutorials/render_installation_route_cards.py')
                    or generator.get('style_source_sha256') != digest(REPO / 'tools/tutorials/authoring/tools/render_install_keyframes.py')
                    or frames.get(visual) != dict(image=image.name, sha256=digest(image), buttons=[], kind=kind)):
                raise ValueError('Doctor generated source provenance changed')
            new_origin = dict(kind='reference_guidance', capture=str(image.parent), source_visual=visual,
                image_sha256=digest(image), installed_identity=None, representation=kind,
                generator_provenance=binding(provenance))
            approval = read(require_binding(replacement['card_visual_acceptance']))
            if (approval.get('accepted') is not True or approval.get('lesson') != identity
                    or approval.get('scene') != 10 or approval.get('terminal_output_fabricated') is not False
                    or approval.get('installation_or_environment_commands_executed') is not False):
                raise ValueError('Missing actual doctor reference visual acceptance')
            _same_bytes(approval['image'], image)
            _same_bytes(approval['generator_provenance'], provenance)
            history = read(require_binding(replacement['historical_doctor']))
            rows = [r for r in history if r.get('scene') == '05_doctor']
            if (len(rows) != 1 or rows[0].get('accepted') is not True or rows[0].get('returncode') != 0
                    or rows[0].get('command') != ['spacr-doctor', '--no-gpu-probe']
                    or '13 passed, 1 warnings, 0 failed, 0 errored, 4 skipped' not in rows[0].get('output', '')
                    or 'czi: missing pylibCZIrw' not in rows[0]['output']
                    or 'installer selected cpu; torch currently uses cpu' not in rows[0]['output']
                    or 'no GPU probe was run' not in rows[0]['output']):
                raise ValueError('Doctor reference is not bound to the actual recorded CPU result')
        if replacement['new_origin'] != new_origin:
            raise ValueError('Mixed replacement has an untruthful origin')
        expected['capture_source']['frame_origins'][visual] = new_origin
    expected['capture_source'].update(completed_capture=False, frame_pixels_modified=True)
    if spec != expected:
        raise ValueError('Mixed specification changed outside the two approved replacements')
    for index, (old_scene, new_scene) in enumerate(zip(old['scenes'], spec['scenes'])):
        if (digest(original / old_scene['image']) != old_scene['capture_sha256']
                or digest(root / new_scene['image']) != new_scene['capture_sha256']):
            raise ValueError('Mixed source image changed')
        if old_scene.get('clip'):
            for key, checksum in (('video', 'sha256'), ('receipt', 'receipt_sha256')):
                if digest(original / old_scene['clip'][key]) != old_scene['clip'][checksum]:
                    raise ValueError('Original native source changed')
                if index != 8 and digest(root / new_scene['clip'][key]) != old_scene['clip'][checksum]:
                    raise ValueError('Unselected legacy native source changed')
    render = read(require_binding(receipt['render_log']))
    log = require_binding(render['output_log'])
    if (render.get('returncode') != 0 or not log.read_text(encoding="utf-8").rstrip().endswith('Terminal exit status: 0')
            or render.get('command') != mixed_overlay_command(old_master, home, card, master, native_frames)):
        raise ValueError('Mixed render did not preserve the two exact source intervals')
    expected_timing = mixed_retained_timing(timing, digest(spec_path), digest(receipt_path),
        digest(manifest_path), replacements[0]['new_video_sha256'], replacements[0]['new_receipt_sha256'], native_frames)
    timing_path = root / 'video/native-live-timings.json'
    if read(timing_path) != expected_timing:
        raise ValueError('Mixed timeline changed outside its spec, receipt and Home source bindings')
    return timing_path
