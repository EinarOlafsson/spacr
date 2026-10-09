"""Compose source-bound installation scenes from verified, distinct recordings.

Published package output, current nightly UI and reference guidance retain their
original evidence and explicit roles. The result is not one installation session.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
from stage_lesson import read, write

KINDS = {'published_installation', 'current_nightly', 'reference_guidance', 'official_browser'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_identity(proof):
    """Find the retained public installation identity in an accepted composition."""
    if isinstance(proof.get('installed_identity'), dict):
        return proof['installed_identity']
    parent = proof.get('public_installation_capture')
    return installed_identity(parent) if isinstance(parent, dict) else None


def compose(lesson, plan, destination):
    """Check every source before copying unchanged native bytes into a new folder."""
    from PIL import Image
    identity = hashlib.sha256(__import__('json').dumps(
        lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    visuals = {row['visual'] for row in lesson['scenes']}
    if (plan.get('lesson') != lesson['id'] or plan.get('english_sha256') != identity
            or set(plan.get('scenes', {})) != visuals or destination.exists()):
        raise ValueError('Use a new destination and an exact source-bound scene plan')
    frames, origins, inputs, copies = {}, {}, {}, []
    for visual, selection in plan['scenes'].items():
        kind = selection.get('kind')
        if kind not in KINDS:
            raise ValueError('Each installation scene needs an explicit evidence role')
        source = Path(selection['capture']).resolve()
        proof = read(source / 'provenance.json')
        manifest = read(source / 'frames.json')
        if not proof.get('completed_capture'):
            raise ValueError('Installation composition requires completed original evidence')
        original = deepcopy(manifest[selection['source_visual']])
        public = installed_identity(proof)
        if kind == 'published_installation' and (not public or not public.get('version')
                                                  or not public.get('package')):
            raise ValueError('Published output requires an actual installed package identity')
        if kind == 'reference_guidance' and not str(original.get('kind', '')).startswith('generated_'):
            raise ValueError('A reference card must retain its explicit generated-guidance identity')
        if kind == 'current_nightly':
            narrated = [row['narration'] for row in lesson['scenes'] if row['visual'] == visual]
            if not all('nightly' in text.lower() for text in narrated) or not original.get('clip'):
                raise ValueError('Current UI needs genuine footage and an explicit nightly qualifier')
        source_key = str(source)
        if source_key not in inputs:
            inputs[source_key] = {'capture': source_key,
                'provenance_sha256': digest(source / 'provenance.json'),
                'frames_sha256': digest(source / 'frames.json'),
                'provenance': proof}
            exact = source.parent.parent / 'exact-source.json'
            if exact.exists():
                inputs[source_key]['exact_source_sha256'] = digest(exact)
                inputs[source_key]['exact_source'] = read(exact)
        image = source / original['image']
        if (Path(original['image']).name != original['image'] or digest(image) != original['sha256']):
            raise ValueError('Original installation frame bytes or path changed')
        with Image.open(image) as frame:
            if frame.size != (3840, 2160) or frame.format != 'PNG':
                raise ValueError('Installation scenes require original native 4K PNGs')
            frame.verify()
        output = visual + '.png'
        copies.append((image, output, original['sha256']))
        original['image'] = output
        if original.get('clip'):
            clip = original['clip']
            video, receipt_path = source / clip['video'], source / clip['receipt']
            receipt = read(receipt_path)
            if (Path(clip['video']).name != clip['video']
                    or Path(clip['receipt']).name != clip['receipt']
                    or digest(video) != clip['sha256']
                    or digest(receipt_path) != clip['receipt_sha256']
                    or receipt.get('sha256') != clip['sha256']
                    or not receipt.get('full_decode_passed')
                    or receipt.get('audio_streams') != 0
                    or receipt.get('size') != [3840, 2160]):
                raise ValueError('Native installation companion footage is unverified')
            if kind != 'current_nightly':
                raise ValueError('Native UI companions must retain their nightly evidence role')
            clip['video'], clip['receipt'] = visual + '.mp4', visual + '.capture.json'
            copies += [(video, clip['video'], clip['sha256']),
                       (receipt_path, clip['receipt'], clip['receipt_sha256'])]
        frames[visual] = original
        origins[visual] = {'kind': kind, 'capture': source_key,
                          'source_visual': selection['source_visual'],
                          'image_sha256': original['sha256'],
                          'installed_identity': public if kind == 'published_installation' else None}
    destination.mkdir(parents=True)
    try:
        for source, relative, expected in copies:
            output = destination / relative
            shutil.copyfile(source, output)
            if digest(output) != expected:
                raise ValueError('Copied installation evidence differs from its original')
        write(destination / 'frames.json', frames)
        write(destination / 'provenance.json', {
            'completed_capture': True, 'lesson': lesson['id'], 'english_sha256': identity,
            'not_one_application_session': True, 'explicitly_distinguished_in_narration': True,
            'app_source_modified': False, 'frame_pixels_modified': False,
            'original_evidence_preserved': True, 'frame_origins': origins,
            'source_captures': list(inputs.values()), 'native_mac_claim': False,
            'native_compositor_claim': False, 'published': False})
    except BaseException:
        shutil.rmtree(destination)
        raise
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    frames = compose(read(args.lesson), read(args.plan), args.destination.resolve())
    print(f'{len(frames)} source-bound installation visuals; original pixels and receipts retained')


if __name__ == '__main__':
    main()
