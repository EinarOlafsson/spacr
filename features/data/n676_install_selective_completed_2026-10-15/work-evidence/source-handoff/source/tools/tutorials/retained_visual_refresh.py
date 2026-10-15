"""Check a bounded visual overlay against an immutable published timeline.

This deliberately preserves legacy clip selections. It neither replans native
coverage nor claims a fresh capture or freshly synthesized narration.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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
    if (not output_log.read_text().rstrip().endswith('Terminal exit status: 0')
            or render.get('returncode') != 0 or render.get('command') != overlay_command(
            old_master, card, master, first, count, total, fps)):
        raise ValueError('Render did not use the bounded published-master overlay')
    timing_path = root / 'video/native-live-timings.json'
    if read(timing_path) != retained_timing(timing, digest(spec_path), digest(receipt_path), digest(manifest_path)):
        raise ValueError('Retained visual timing changed outside its explicit refresh binding')
    return timing_path
