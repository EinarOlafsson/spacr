#!/usr/bin/env python3
"""Validate fresh lesson 20 scientific evidence and explicitly selected clips.

This creates only the requested derived JSON; it never edits capture sources,
runs Cellpose, decodes video, or claims accuracy, narration alignment or listening
review. Existing hash-bound full-decode receipts and clip gate reports are used.

Sources are host capture directories. --selected-scenes is the same absolute
capture-directory -> scene-name-list JSON used by compose_capture_set.py.
Each --verifier-receipts file must be a fresh verification_run.json containing
source_run (host stage), capture, started_at/finished_at (UTC), command array,
exit_code=0, reference_absent_at_start=true, reference_sha256 and dependencies.
Dependencies must include proof/settings/frames/provenance/exact-source JSON,
all three preview NPY files and all figure NPZ files, unchanged across verification.
Near-miss approval must already exist on both the original gate and summary row.
For result scenes whose starting frame still has a dialog, --terminal-review
can supply an explicit AI visual review (reviewer label as named in REVIEWER_LABELS) of a hash-bound final clip frame.
That review establishes an unobstructed terminal result, never the whole clip.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/tutorials')]
NEUTRAL_STAGE = Path('/tmp/spacr-tutorial-current')
DEFAULT_CHECKOUT = REPO
LESSON_ID = '20_cellpose_masks'
ACCEPTANCE_SCOPE = ('Fresh direct-reference input/output attribution, exact live filtering '
                    'and explicitly selected verified clips; not segmentation accuracy or listening review')
SCOPE_FLAGS = ('accuracy_validated', 'ground_truth_used', 'preview_equivalence_fixed',
               'human_listening_review', 'native_speaker_signoff',
               'narration_scene_alignment_reviewed', 'published')
PRIVATE_BINDS = []
PREVIEWS = ('preview_cell.npy', 'preview_cell_filtered.npy',
            'preview_cell_restored.npy')
#: Who may sign a terminal-frame review. The label is the real reviewer, not a fixed name.
REVIEWER_LABELS = ('Codex AI visual review', 'Claude AI visual review (WS)')
UNOBSTRUCTED = {'20a_actual_zoomed_preview',
                '22b_actual_zoomed_filtered_preview',
                '23b_actual_zoomed_restored_preview'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Not a regular file: {path}')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    attributes = ('st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, key) == getattr(after, key) for key in attributes),
            f'File changed while hashing: {path}')
    return digest.hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        digest = checksum(path)
        if expected is not None:
            require(re.fullmatch(r'[0-9a-f]{64}', str(expected)) is not None,
                    f'Invalid expected SHA256 for {path}')
            require(digest == expected, f'Evidence digest changed: {path}')
        previous = self.hashes.get(str(path))
        require(previous is None or previous == digest, f'Evidence changed: {path}')
        self.hashes[str(path)] = digest
        return digest

    def read(self, path, expected=None):
        require(Path(path).stat().st_size <= 32 * 1024 * 1024,
                f'JSON evidence is unexpectedly large: {path}')
        self.bind(path, expected)
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        self.bind(path, expected)
        return value

    def unchanged(self):
        for path, digest in self.hashes.items():
            require(checksum(path) == digest, f'Evidence changed during validation: {path}')


def child(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            f'Unsafe evidence-relative path: {relative}')
    result = root / relative
    require(result.resolve().is_relative_to(root.resolve()),
            f'Evidence escapes capture root: {result}')
    return result


def mapped(value, source):
    """Resolve sandbox stage paths without modifying their recorded spelling."""
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts,
            f'Expected an absolute evidence path: {value}')
    for neutral, host in PRIVATE_BINDS:
        if path.is_relative_to(neutral):
            return host / path.relative_to(neutral)
    if path.is_relative_to(NEUTRAL_STAGE):
        return source.parent.parent / path.relative_to(NEUTRAL_STAGE)
    return path


def utc_time(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            f'Expected a UTC timestamp: {value}')
    return parsed


def recorded_checkout(receipt, fallback=DEFAULT_CHECKOUT):
    """The command's recorded working directory, independent of this helper."""
    command = receipt.get('command')
    require(isinstance(command, list) and all(isinstance(value, str) for value in command),
            'Missing verifier command array')
    directories = [command[index + 1] for index, value in enumerate(command[:-1])
                   if value == '--chdir']
    require(len(directories) <= 1, 'Verifier command has ambiguous working directories')
    if directories:
        root = Path(directories[0])
        require(root.is_absolute(), 'Verifier working directory must be host absolute')
        return root.resolve()
    scripts = [Path(value) for value in command if Path(value).name == 'verify_cellpose_apply.py']
    if len(scripts) == 1 and scripts[0].is_absolute():
        return scripts[0].resolve().parents[2]
    return Path(fallback).resolve()


def check_verifier(evidence, receipt_path, receipt, source, exact_path, arrays,
                   checkout=DEFAULT_CHECKOUT):
    require(Path(receipt['capture']).absolute() == source,
            'Verifier receipt names a different capture')
    require(Path(receipt['source_run']).absolute() == source.parent.parent,
            'Verifier receipt names a different host stage')
    require(type(receipt.get('exit_code')) is int and receipt['exit_code'] == 0,
            'Direct-reference verifier did not exit successfully')
    require(receipt.get('reference_absent_at_start') is True,
            'No recorded assertion that the reference was absent before verification')
    require(receipt.get('dependencies_unchanged') is True,
            'Verifier did not confirm unchanged dependencies before and after execution')
    started, finished = utc_time(receipt['started_at']), utc_time(receipt['finished_at'])
    require(started <= finished, 'Verifier timestamp interval is invalid')
    command = receipt.get('command')
    require(isinstance(command, list) and all(isinstance(value, str) for value in command),
            'Missing verifier command array')
    scripts = [index for index, value in enumerate(command)
               if Path(value).name == 'verify_cellpose_apply.py']
    require(len(scripts) == 1, 'Receipt did not run the direct Cellpose verifier')
    index = scripts[0]
    neutral_capture = NEUTRAL_STAGE / source.relative_to(source.parent.parent)
    require(command[index + 1:] == [str(neutral_capture), '--device', 'cuda'],
            'Verifier command must target this neutral capture on CUDA')
    script = Path(command[index])
    root = recorded_checkout(receipt, checkout)
    resolved = script if script.is_absolute() else child(root, script)
    require(resolved.resolve() == (root / 'tools/tutorials/verify_cellpose_apply.py').resolve(),
            'Verifier command does not identify the recorded direct-reference script')
    evidence.bind(resolved)
    dependencies = receipt.get('dependencies')
    require(isinstance(dependencies, dict), 'Verifier dependencies are absent')
    required = {str(source / name) for name in (
        'scientific_acceptance.json', 'configured_settings.json',
        'frames.json', 'provenance.json', *PREVIEWS)}
    required.add(str(exact_path))
    required.update(str(path) for path in arrays)
    require(required <= set(dependencies),
            f'Verifier did not bind required dependencies: {sorted(required - set(dependencies))}')
    for path, digest in dependencies.items():
        require(Path(path).is_absolute(), 'Verifier dependency paths must be host absolute paths')
        evidence.bind(path, digest)
    reference_path = source / 'independent_reference.json'
    timestamp = reference_path.stat().st_mtime
    require(started.timestamp() - 0.001 <= timestamp <= finished.timestamp() + 0.001,
            'Reference file is not fresh within the verifier interval')
    return evidence.read(reference_path, receipt['reference_sha256'])


def terminal_dialog_evidence(evidence, source, name, frame, clip, receipt, reviews):
    if 'dialogs' in frame and frame['dialogs'] == []:
        return {'scope': 'original start-frame metadata', 'dialogs_covering_result': False}
    require(name in {'22b_actual_zoomed_filtered_preview',
                     '23b_actual_zoomed_restored_preview'},
            f'Selected result scene lacks unobstructed-frame evidence: {name}')
    review = reviews.get((str(source), name))
    require(isinstance(review, dict), f'Missing explicit terminal visual review: {name}')
    require(review.get('reviewer') in REVIEWER_LABELS
            and review.get('dialogs_covering_result') is False
            and review.get('video_sha256') == clip['sha256'],
            f'Terminal review does not verify this unobstructed result: {name}')
    utc_time(review['reviewed_at'])
    second, duration = float(review['time_seconds']), float(receipt['duration'])
    require(math.isfinite(second) and max(0, duration - 1) <= second < duration,
            f'Terminal review is not within the final second of the clip: {name}')
    image = Path(review['image'])
    require(image.is_absolute(), 'Terminal review image must have a host absolute path')
    evidence.bind(image, review['image_sha256'])
    from PIL import Image
    with Image.open(image) as terminal:
        require(terminal.format == 'PNG' and terminal.size == (3840, 2160),
                'Terminal review image has invalid handoff geometry')
        terminal.verify()
    command = review.get('command')
    require(isinstance(command, list) and command
            and all(isinstance(value, str) for value in command)
            and Path(command[0]).name == 'ffmpeg', 'Missing terminal extraction ffmpeg command')
    require(command.count('-i') == 1 and '-ss' in command
            and '-frames:v' in command and command[-1] == str(image),
            'Terminal extraction must name one clip input, one frame, time and output image')
    require(command[command.index('-i') + 1] == str(child(source, clip['video']))
            and command[command.index('-frames:v') + 1] == '1',
            'Terminal extraction command identifies a different input or frame count')
    time_text = command[command.index('-ss') + 1]
    components = [float(value) for value in time_text.split(':')]
    require(1 <= len(components) <= 3, 'Invalid terminal extraction timestamp')
    extracted_second = sum(value * 60 ** index for index, value in enumerate(reversed(components)))
    require(math.isclose(extracted_second, second, abs_tol=1 / 30),
            'Terminal review time and extraction command disagree')
    return {'scope': 'AI-reviewed terminal frame only; initial dialog metadata retained',
            'dialogs_covering_result': False, 'review': review}


def check_clip(evidence, source, name, frame, summary, terminal_reviews):
    clip = frame.get('clip')
    required = {'video', 'sha256', 'receipt', 'receipt_sha256', 'native_video', 'native_sha256', 'gate'}
    require(isinstance(clip, dict) and required <= set(clip)
            and not clip.get('pending_post_processing'),
            f'Selected scene lacks a completed clip: {name}')
    image = child(source, frame['image'])
    evidence.bind(image, frame['sha256'])
    from PIL import Image
    with Image.open(image) as poster:
        require(poster.format == 'PNG' and poster.size == (3840, 2160),
                f'Selected poster has invalid handoff geometry: {name}')
        poster.verify()
    video = child(source, clip['video'])
    native = child(source, clip['native_video'])
    evidence.bind(video, clip['sha256'])
    evidence.bind(native, clip['native_sha256'])
    receipt = evidence.read(child(source, clip['receipt']), clip['receipt_sha256'])
    require(receipt.get('video') == video.name and receipt.get('sha256') == clip['sha256'],
            f'Clip receipt identifies different video bytes: {name}')
    require(receipt.get('full_decode_passed') is True
            and receipt.get('audio_streams') == 0
            and receipt.get('size') == [3840, 2160]
            and bool(receipt.get('frames'))
            and float(receipt.get('duration', 0)) > 0,
            f'Selected clip lacks a full silent handoff decode: {name}')
    require(receipt.get('target_fps') == 30
            and Fraction(receipt['encoded_fps']) == 30,
            f'Selected clip is not the required 30 fps: {name}')
    require(receipt.get('native', {}).get('sha256') == clip['native_sha256']
            and receipt['native'].get('video') == native.name
            and receipt['native'].get('size') == [2560, 1440],
            f'Native source receipt/geometry differs: {name}')
    require(receipt.get('adapter_bypasses') == [], f'Human-pointer bypass in {name}')
    row = summary.get('clips', {}).get(name)
    require(isinstance(row, dict), f'Selected scene lacks its host gate summary: {name}')
    gate = evidence.read(child(source, row['file']))
    require(Path(gate['video']).name == native.name and gate.get('size') == [2560, 1440],
            f'Gate identifies a different native video: {name}')
    require(math.isclose(float(gate.get('fps', 0)), 30, abs_tol=0.001),
            f'Gate used the wrong native frame rate: {name}')
    for value in (row, clip.get('gate', {}), receipt.get('gate', {})):
        require(value.get('passed') == gate.get('passed')
                and value.get('reasons') == gate.get('reasons'),
                f'Gate/manifest/receipt summary disagree: {name}')
    require(row.get('distinct_fps') == gate.get('distinct_fps'),
            f'Distinct-rate summary differs: {name}')
    facts = evidence.read(source / f'{name}.host.json')
    evidence.bind(source / f'{name}.pointer.json')
    require(facts.get('adapter_bypasses') == [] and gate.get('adapter_bypasses') == [],
            f'Adapter bypass evidence is missing or nonempty: {name}')
    cursor = gate.get('cursor', {})
    require(cursor.get('jump_count') == 0 and not cursor.get('jumps')
            and float(cursor.get('found_fraction', 0)) >= 0.9,
            f'Selected clip lacks continuous visible pointer evidence: {name}')
    require(gate.get('top_bar', {}).get('hidden') is True,
            f'Selected clip has no explicit hidden-top-bar evidence: {name}')
    reasons = gate.get('reasons')
    if gate.get('passed') is True:
        require(reasons == []
                and not gate.get('app_freeze_seconds')
                and float(gate['limits']['min_fps']) >= 27.9
                and float(gate['distinct_fps']) >= 27.9,
                f'Selected clip is not a strict smoothness pass: {name}')
        decision = 'strict_pass'
    else:
        require(gate.get('passed') is False, f'Selected scene has no verified gate verdict: {name}')
        rate = float(gate.get('distinct_fps', 0))
        require(rate >= 27.0,
                f'Selected scene is below the 27.0 distinct-fps approval floor: {name} ({rate})')
        require(gate.get('accepted_near_miss') == 'capture loss'
                and row.get('accepted_near_miss') == 'capture loss'
                and isinstance(reasons, list) and len(reasons) == 1
                and reasons[0].startswith('choppy (capture loss):')
                and not gate.get('app_freeze_seconds')
                and not gate.get('app_slow_seconds')
                and not gate.get('app_unknown_seconds')
                and bool(gate.get('capture_loss_seconds')),
                f'Failed clip lacks a valid prior capture-loss-only approval: {name}')
        # A legacy aggregate counter can advance after the GUI returns from a
        # stall. Such delayed delivery cannot establish capture-loss-only scope.
        import numpy as np
        samples = facts.get('app_paint_samples') or []
        intervals = [float(b[0]) - float(a[0]) for a, b in zip(samples, samples[1:])]
        positive = [value for value in intervals if np.isfinite(value) and value > 0]
        cadence = facts.get('app_gui_sampler_interval_s')
        if cadence is None and len(positive) >= 3 and np.median(positive) <= 0.2:
            cadence = float(np.median(positive))
        delayed = facts.get('app_gui_sampler_gaps_s') or []
        if cadence is not None:
            delayed = delayed or [value for value in positive if value >= max(0.35, 3.5 * cadence)]
        require(not delayed, f'GUI sampler delays cannot justify capture-loss-only approval: {name}')
        decision = 'approved_capture_loss_near_miss'
    dialog_evidence = None
    if name in UNOBSTRUCTED:
        dialog_evidence = terminal_dialog_evidence(
            evidence, source, name, frame, clip, receipt, terminal_reviews)
    return {'scene': name, 'capture': str(source), 'decision': decision,
            'video_sha256': clip['sha256'], 'native_sha256': clip['native_sha256'],
            'image_sha256': frame['sha256'], 'receipt_sha256': clip['receipt_sha256'],
            'gate_sha256': evidence.hashes[str(child(source, row['file']).absolute())],
            'original_gate_passed': gate['passed'], 'original_gate_reasons': reasons,
            'original_start_frame_dialogs': frame.get('dialogs'),
            'result_dialog_evidence': dialog_evidence}


def validate(args):
    import numpy as np
    import tifffile

    checkout = args.checkout.resolve()
    PRIVATE_BINDS.clear()
    for value in args.private_bind:
        neutral, separator, host = value.partition('=')
        require(separator and Path(neutral).is_absolute() and Path(host).is_absolute()
                and Path(neutral).is_relative_to(NEUTRAL_STAGE)
                and Path(host).is_dir(), 'Private bind must map a neutral subtree to a real host directory')
        PRIVATE_BINDS.append((Path(neutral), Path(host)))
    PRIVATE_BINDS.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    from compose_cellpose_apply_capture import require_scope, require_closed_dialogs
    from cellpose_apply_evidence import minimum_area_labels, require_pixels
    from capture_acceptance import assess_pipeline

    sources = [path.absolute() for path in args.sources]
    require(len(sources) == len(set(sources)) and sources, 'Capture sources must be distinct')
    require(all(path.is_dir() and not path.is_symlink() for path in sources),
            'Capture sources must be real host directories')
    output = args.output.absolute()
    require(not output.exists() and all(not output.is_relative_to(source) for source in sources),
            'Output must be new and outside the raw capture directories')
    evidence = Evidence()
    evidence.bind(Path(__file__).absolute())
    lesson = evidence.read(args.english.absolute())
    require(lesson.get('id') == LESSON_ID, 'Expected the current lesson 20 English source')
    visuals = [scene['visual'] for scene in lesson['scenes']]
    require(visuals and len(visuals) == len(set(visuals)), 'English scene visuals must be unique')
    english_hash = hashlib.sha256(json.dumps(lesson, sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
    picks = evidence.read(args.selected_scenes.absolute())
    require(isinstance(picks, dict) and set(picks) == {str(path) for path in sources},
            'Selection must explicitly name every supplied capture source and no others')
    receipts = {}
    for path in args.verifier_receipts:
        receipt = evidence.read(path.absolute())
        key = str(Path(receipt['capture']).absolute())
        require(key not in receipts, 'Duplicate verifier receipt for one capture')
        receipts[key] = (path.absolute(), receipt)
    require(set(receipts) == set(picks), 'Every selected source needs exactly one verifier receipt')
    terminal_reviews = {}
    if args.terminal_review is not None:
        terminal = evidence.read(args.terminal_review.absolute())
        require(terminal.get('schema') == 1 and isinstance(terminal.get('reviews'), list),
                'Terminal review JSON requires schema 1 and reviews list')
        for review in terminal['reviews']:
            key = (str(Path(review['capture']).absolute()), review['scene'])
            require(key not in terminal_reviews and key[0] in picks and key[1] in picks[key[0]],
                    'Terminal review must name a unique explicitly selected scene')
            terminal_reviews[key] = review
    observations, selected, merged_frames = [], [], {}
    common_inventory = None
    for source in sources:
        names = picks[str(source)]
        require(isinstance(names, list) and names and all(isinstance(name, str) for name in names)
                and len(names) == len(set(names)),
                'Every source needs a nonempty, duplicate-free explicit scene list')
        proof = evidence.read(source / 'scientific_acceptance.json')
        require(proof.get('lesson') == lesson['id'] and proof.get('accepted') is False
                and proof.get('synthetic') is False,
                'Expected the unchanged raw, real-data, unaccepted lesson 20 observation')
        provenance = evidence.read(source / 'provenance.json')
        require(provenance.get('module') == 'cellpose_masks', 'Wrong recording module')
        summary = evidence.read(source / 'host_gate_summary.json')
        require(provenance.get('completed_capture') is True
                or (provenance.get('completed_capture') is False
                    and provenance.get('status') == 'gate_failed'),
                'Only completed or gate-failed full recordings may contribute selected clips')
        exact_path = source.parent.parent / 'exact-source.json'
        exact = evidence.read(exact_path)
        inventory = exact.get('sha256')
        require(exact.get('fresh') is True and exact.get('private_profile') is True
                and isinstance(inventory, dict) and bool(inventory),
                'Missing fresh private exact-source inventory')
        if common_inventory is None:
            common_inventory = inventory
        require(inventory == common_inventory, 'Selected recordings use different app/recorder sources')
        receipt_path, verifier = receipts[str(source)]
        source_checkout = recorded_checkout(verifier, checkout)
        for relative, digest in inventory.items():
            evidence.bind(child(source_checkout, relative), digest)
        for relative in ('tools/tutorials/verify_cellpose_apply.py',
                         'tools/tutorials/compose_cellpose_apply_capture.py',
                         'tools/tutorials/cellpose_apply_evidence.py'):
            evidence.bind(source_checkout / relative)
        arrays = sorted(source.glob('batch_figure_*_arrays.npz'))
        require(len(arrays) == 3, 'Expected three complete batch figure arrays')
        outcome = evidence.read(source / 'batch_outcome.json')
        console = evidence.read(source / 'batch_console.json')
        require(isinstance(console, list) and all(isinstance(value, str) for value in console),
                'Expected preserved real batch console blocks')
        pipeline = assess_pipeline(outcome, console, len(arrays), requires_figure=True)
        require(pipeline['accepted'] is True and pipeline == proof.get('pipeline'),
                'Raw pipeline acceptance differs from the preserved worker/console evidence')
        reference = check_verifier(evidence, receipt_path, verifier, source, exact_path, arrays, checkout)
        require_scope(proof, reference)
        frames = evidence.read(source / 'frames.json')
        require(set(names) <= set(frames), 'Selection names an absent or unfinished scene')
        originals = proof.get('original_inputs')
        require(isinstance(originals, dict) and len(originals) == 3,
                'Expected all three original real-image inputs')
        work = mapped(proof['private_folder'], source)
        names_of_inputs = {Path(path).name for path in originals}
        require(len(names_of_inputs) == 3, 'Original input names collide')
        path_mapping = {proof['private_folder']: str(work)}
        for original, digest in originals.items():
            original_path = mapped(original, source)
            path_mapping[original] = str(original_path)
            evidence.bind(original_path, digest)
            evidence.bind(work / Path(original).name, digest)
            image = tifffile.imread(work / Path(original).name)
            require(image.shape == (512, 512) and image.dtype == np.uint16,
                    'Input is not a recorded single-channel 512-square uint16 crop')
        rows = reference.get('files', [])
        require(len(rows) == 3 and {row['name'] for row in rows} == names_of_inputs,
                'Independent reference does not cover each input exactly once')
        outputs = proof.get('outputs', [])
        require(len(outputs) == 3 and {row['image'] for row in outputs} == names_of_inputs,
                'Raw recording outputs do not cover each input exactly once')
        output_hashes = {row['image']: row['sha256'] for row in outputs}
        source_hashes = {Path(path).name: digest for path, digest in originals.items()}
        for row in rows:
            require(row.get('exact_mask_equal') is True and row.get('checked_pixels') == 512 * 512
                    and row.get('source_sha256') == source_hashes[row['name']]
                    and row.get('saved_mask_sha256') == output_hashes[row['name']],
                    'Reference output attribution differs from the raw recording')
            evidence.bind(work / 'masks' / row['name'], row['saved_mask_sha256'])
            mask = tifffile.imread(work / 'masks' / row['name'])
            require(mask.shape == (512, 512) and np.issubdtype(mask.dtype, np.integer)
                    and mask.min() >= 0, 'Saved mask is not a nonnegative instance-label image')
        checkpoint = mapped(reference['checkpoint'], source)
        evidence.bind(checkpoint, reference['checkpoint_sha256'])
        path_mapping[reference['checkpoint']] = str(checkpoint)
        figures = reference.get('figures', [])
        require(len(figures) == 3
                and {row['source_image'] for row in figures} == names_of_inputs
                and {row['file'] for row in figures} == {path.name for path in arrays},
                'Figure reference must map all three arrays one-to-one to all inputs')
        for row in figures:
            require(row.get('exact_values_checked') == {
                'original': 512 * 512, 'overlay': 512 * 512 * 3, 'flow': 512 * 512},
                'Figure reference is missing complete original/overlay/flow pixel checks')
            evidence.bind(child(source, row['file']))
        require(len(proof.get('figures', [])) == 3,
                'Expected three preserved native batch figure PNGs')
        for row in proof['figures']:
            evidence.bind(child(source, row['path']), row['sha256'])
        raw, filtered, restored = [np.load(source / name, allow_pickle=False) for name in PREVIEWS]
        require(raw.shape == (512, 512), 'Preview has the wrong dimensions')
        require_pixels(filtered, minimum_area_labels(raw, proof['filter']['cutoff']))
        require_pixels(restored, minimum_area_labels(raw, 0))
        require(not np.array_equal(raw, tifffile.imread(work / 'masks/cell_pair_02.tif')),
                'The narrated preview/batch mismatch is absent')
        require(reference.get('preview_default_reference', {}).get('exact_mask_equal') is True
                and reference['preview_default_reference'].get('checked_pixels') == raw.size,
                'Preview lacks complete direct-default reference checks')
        require(proof.get('original_inputs_preserved') is True
                and proof.get('private_inputs_preserved') is True,
                'Raw source-preservation checks did not complete')
        for name in names:
            require(name not in merged_frames, 'Selected recordings duplicate a lesson scene')
            selected.append(check_clip(evidence, source, name, frames[name], summary, terminal_reviews))
            merged_frames[name] = frames[name]
        observations.append({'capture': str(source), 'raw_observation': proof,
                             'recorded_checkout': str(source_checkout),
                             'verifier_receipt_path': str(receipt_path),
                             'independent_reference': reference, 'verifier_receipt': verifier,
                             'original_provenance': provenance, 'host_gate_summary': summary,
                             'selected_scenes': names, 'omitted_scenes': sorted(set(frames) - set(names)),
                             'resolved_private_paths': path_mapping})
    require(set(merged_frames) == set(visuals),
            'Explicit selected clips must cover every current English scene exactly once')
    # The old composer checks a result snapshot, while host manifests describe
    # the START of an action clip. Feed only snapshots actually verified above:
    # original empty-dialog snapshots or explicitly reviewed terminal snapshots.
    # The original frames and their initial dialog metadata remain hash-bound.
    validated_result_views = {
        row['scene']: {'dialogs': []}
        for row in selected if row['result_dialog_evidence'] is not None
        and row['result_dialog_evidence']['dialogs_covering_result'] is False
    }
    require_closed_dialogs(validated_result_views)
    evidence.unchanged()
    result = {'schema': 1, 'lesson': lesson['id'], 'accepted': True,
              'scope': ACCEPTANCE_SCOPE,
              'created_at': datetime.now(timezone.utc).isoformat(),
              'canonical_english_sha256': english_hash,
              'explicit_private_binds': [[str(a), str(b)] for a, b in PRIVATE_BINDS],
              'native_observations': observations, 'selected_clips': selected,
              'source_hashes': evidence.hashes, 'accuracy_validated': False,
              'ground_truth_used': False, 'preview_equivalence_fixed': False,
              'human_listening_review': False, 'native_speaker_signoff': False,
              'narration_scene_alignment_reviewed': False, 'published': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--selected-scenes', required=True, type=Path)
    parser.add_argument('--english', required=True, type=Path)
    parser.add_argument('--verifier-receipts', required=True, nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkout', type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument('--terminal-review', type=Path)
    parser.add_argument('--private-bind', action='append', default=[], metavar='NEUTRAL=HOST')
    args = parser.parse_args()
    print(validate(args))


if __name__ == '__main__':
    main()
