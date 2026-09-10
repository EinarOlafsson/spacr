#!/usr/bin/env python3
"""Compose accepted Report GUI/browser references for stage_lesson.

Only three new JSON manifests are written. Existing images, receipts and source
data are never copied, rendered or changed. The destination must be a new sibling
under the same private captures directory. Acceptance is technical composition,
not a new application run, QC verdict, human review or publication approval.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path, hashes):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Expected an existing bounded regular receipt: ' + str(path))
    raw = path.read_bytes()
    hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def _same_hash(path, expected, hashes):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not re.fullmatch('[0-9a-f]{64}', str(expected)):
        raise ValueError('Expected a regular file with a valid SHA256: ' + str(path))
    actual = _sha(path)
    if actual != expected:
        raise ValueError('Source hash differs: ' + str(path))
    hashes[str(path.resolve())] = actual


def _frame(path, expected, capture_root, hashes, declared=None):
    from PIL import Image
    path = Path(path)
    if (path.is_symlink() or not path.resolve().is_relative_to(capture_root)
            or path.suffix.lower() != '.png' or path.stat().st_size > 64 * 1024 * 1024):
        raise ValueError('Frame must be a bounded original PNG inside its capture directory')
    _same_hash(path, expected, hashes)
    with Image.open(path) as image:
        if image.format != 'PNG' or image.size != (3840, 2160):
            raise ValueError('Frame is not a genuine 3840 by 2160 PNG: ' + str(path))
        image.verify()
    if declared is not None and declared != [3840, 2160]:
        raise ValueError('Declared browser frame geometry differs from actual PNG')
    return path.resolve()


def _build(gui, browser, destination):
    """Read-only verification and assembly; no destination is created here."""
    gui, browser, destination = Path(gui), Path(browser), Path(destination)
    if any(path.is_symlink() for path in (gui, browser, destination)):
        raise ValueError('Capture roots and destination must not be symlinks')
    gui, browser, destination = gui.resolve(), browser.resolve(), destination.resolve()
    if (destination.exists() or not gui.is_dir() or not browser.is_dir()
            or gui == browser or gui.parent != browser.parent
            or destination.parent != gui.parent or gui.parent.name != 'captures'):
        raise ValueError('Use two accepted capture siblings and a new sibling destination under captures')
    hashes = {}
    provenance = _read(gui / 'provenance.json', hashes)
    scientific = _read(gui / 'scientific_acceptance.json', hashes)
    report = _read(gui / 'report_output_evidence.json', hashes)
    gui_frames = _read(gui / 'frames.json', hashes)
    review = _read(browser / 'browser_review.json', hashes)
    browser_frames = _read(browser / 'frames.json', hashes)
    if (provenance.get('completed_capture') is not True or provenance.get('module') != 'report'
            or not re.fullmatch('[0-9a-f]{40}', str(provenance.get('commit')))
            or not isinstance(provenance.get('version'), str) or not provenance['version']):
        raise ValueError('A completed Report GUI capture with source commit/version is required')
    if (scientific.get('accepted') is not True or report.get('accepted') is not True
            or scientific != report or report.get('lesson') != '29_report'
            or report.get('original_preserved') is not True
            or report.get('private_material_files_unchanged') is not True):
        raise ValueError('The GUI scientific and generated-report receipts must agree and be accepted')
    required_true = ('accepted', 'browser_reviewed', 'dom_verified', 'original_preserved',
                     'private_source_exact_post_generation_state_preserved', 'receipt_unchanged', 'html_unchanged')
    required_false = ('native_human_publish', 'published', 'open_control_clicked',
                      'html_css_dom_injected', 'report_recomputed', 'horizontal_overflow')
    if (review.get('lesson') != '29_report'
            or any(review.get(key) is not True for key in required_true)
            or any(review.get(key) is not False for key in required_false)
            or any(review.get(key) != [] for key in ('blocked_requests', 'javascript_errors',
                   'console_errors', 'request_failures', 'downloads'))):
        raise ValueError('A clean accepted technical browser receipt is required')
    receipt_path = (gui / 'report_output_evidence.json').resolve()
    if (Path(review['receipt']).resolve() != receipt_path
            or review['receipt_sha256'] != hashes[str(receipt_path)]
            or Path(review['html']).resolve() != Path(report['output']).resolve()
            or review['html_sha256'] != report['output_sha256']
            or Path(review['source']).resolve() != Path(report['source']).resolve()):
        raise ValueError('Browser receipt is not bound to this exact GUI report and HTML')
    _same_hash(report['output'], report['output_sha256'], hashes)
    if review['frames_manifest_sha256'] != hashes[str((browser / 'frames.json').resolve())]:
        raise ValueError('Browser frame manifest differs from accepted browser receipt')
    _same_hash(browser / 'browser_dom.json', review['browser_dom_sha256'], hashes)
    if (browser_frames.get('kind') != 'standalone_chrome_report'
            or browser_frames.get('frames') != review.get('frames')
            or not isinstance(browser_frames.get('frames'), list)
            or not 1 <= len(browser_frames['frames']) <= 64
            or not isinstance(gui_frames, dict) or not 1 <= len(gui_frames) <= 64):
        raise ValueError('GUI/browser frame inventories are missing, changed or unexpectedly large')
    if (review.get('viewport_css') != {'width': 1920, 'height': 1080}
            or review.get('device_scale_factor') != 2
            or not review.get('browser_version') or not review.get('chrome_executable')
            or not review.get('verification_code_sha256')):
        raise ValueError('Actual HiDPI Chrome version and verification-code provenance are required')
    frames = {}
    for key, frame in gui_frames.items():
        if not re.fullmatch('[A-Za-z0-9_-]+', key):
            raise ValueError('Unexpected GUI frame key')
        path = _frame(gui / frame['image'], frame['sha256'], gui, hashes)
        frames[key] = {**deepcopy(frame), 'image': os.path.relpath(path, destination),
                       'dimensions': [3840, 2160], 'source_capture': 'gui', 'source_frame_key': key}
    for frame in browser_frames['frames']:
        name = Path(frame['name'])
        if name.name != frame['name'] or name.suffix != '.png' or not re.fullmatch('[A-Za-z0-9_-]+', name.stem):
            raise ValueError('Unexpected browser PNG name')
        key = 'browser_' + name.stem
        if key in frames:
            raise ValueError('Composed frame key collision: ' + key)
        path = _frame(frame['path'], frame['sha256'], browser, hashes, frame['dimensions'])
        if (path.name != name.name or frame.get('bytes') != path.stat().st_size
                or frame.get('viewport_css') != {'width': 1920, 'height': 1080}
                or frame.get('device_scale_factor') != 2):
            raise ValueError('Browser frame identity or HiDPI metadata differs')
        frames[key] = {'image': os.path.relpath(path, destination), 'sha256': frame['sha256'],
                       'dimensions': [3840, 2160], 'capture_surface': 'chrome_hidpi_viewport',
                       'buttons': [], 'dialogs': [], 'source_capture': 'browser',
                       'source_frame_key': name.stem, 'viewport_css': frame['viewport_css'],
                       'device_scale_factor': frame['device_scale_factor'], 'scroll_css': frame['scroll_css']}
    scope = ('Reference-only composition of accepted Report GUI generation and technical Chrome viewing; '
             'not a new app run, scientific or QC validation, native-human sign-off or publication approval.')
    receipts = {path: digest for path, digest in hashes.items() if Path(path).suffix == '.json'}
    composed_provenance = {
        'completed_capture': False, 'kind': 'accepted_capture_reference_composition',
        'module': 'report', 'lesson': '29_report', 'commit': provenance['commit'], 'version': provenance['version'],
        'commit_scope': 'Application commit/version from the GUI capture; Chrome version and browser helper hashes recorded separately.',
        'acceptance_scope': scope, 'source_receipts_sha256': receipts,
        'gui_capture': {'path': str(gui), 'provenance': provenance},
        'browser_capture': {'path': str(browser), 'browser_version': review['browser_version'],
                            'chrome_executable': review['chrome_executable'],
                            'verification_code_sha256': review['verification_code_sha256']},
        'report_output': str(Path(report['output']).resolve()), 'report_html_sha256': report['output_sha256'],
        'gui_frame_count': len(gui_frames), 'browser_frame_count': len(browser_frames['frames']),
        'composed_frame_count': len(frames), 'images_modified': False, 'source_manifests_modified': False,
        'native_human_publish': False, 'published': False, 'qc_pass_claimed': False,
        'composition_helper_sha256': _sha(__file__)}
    acceptance = {'accepted': False, 'lesson': '29_report', 'acceptance_scope': scope,
                  'composition_verified': False, 'browser_reviewed': True,
                  'browser_review_scope': 'Inherited accepted technical DOM/rendering checks only.',
                  'source_receipts_sha256': receipts, 'report_html_sha256': report['output_sha256'],
                  'original_source': report['original_source'], 'original_preserved_by_source_receipts': True,
                  'new_source_data_check_performed': False, 'open_control_clicked': False,
                  'images_modified': False, 'scientific_validation_performed': False,
                  'native_human_publish': False, 'native_speaker_signoff': False,
                  'qc_pass_claimed': False, 'published': False}
    return destination, frames, composed_provenance, acceptance, hashes


def compose_report_capture(gui, browser, destination):
    destination, frames, provenance, acceptance, hashes = _build(gui, browser, destination)
    # Recheck all manifests, HTML, DOM evidence and PNG bytes before any write.
    for path, expected in hashes.items():
        if _sha(path) != expected:
            raise ValueError('Source changed while composing: ' + path)
    destination.mkdir()  # Exclusive: existing captures are never overwritten.
    for filename, content in [('frames.json', frames), ('scientific_acceptance.json', acceptance),
                              ('provenance.json', provenance)]:
        if filename == 'scientific_acceptance.json':
            content = {**content, 'accepted': True, 'composition_verified': True,
                       'reason': 'Both accepted source receipts and every referenced native 4K PNG were verified; no image or source receipt changed.'}
        elif filename == 'provenance.json':
            content = {**content, 'completed_capture': True}
        with (destination / filename).open('x', encoding='utf-8') as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    return {'destination': str(destination), 'gui_frames': provenance['gui_frame_count'],
            'browser_frames': provenance['browser_frame_count'], 'total_frames': len(frames)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui-capture', required=True, type=Path)
    parser.add_argument('--browser-capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compose_report_capture(args.gui_capture, args.browser_capture, args.output), indent=2))


if __name__ == '__main__':
    main()
