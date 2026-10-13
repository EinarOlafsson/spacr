import collections
import hashlib
import json
import os
import re
from pathlib import Path

from tensorflow.tsl.profiler.protobuf import xplane_pb2

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '551-stardist'
gpu = root / 'gpu-r3'
profile = gpu / 'profiling'
pending_path = gpu / 'inference-pending-gpu-verification.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


pending = json.loads(pending_path.read_text())
cpu = json.loads((root / 'cpu-r3/benchmark.json').read_text())
assert pending['accepted'] is False and pending['gpu_execution_verification_pending'] is True
assert cpu['accepted'] is True and pending['device'] == 'gpu'
assert pending['source_files'] == cpu['source_files']
assert all(digest(path) == row['sha256'] for path, row in pending['source_files'].items())
assert len(pending['cases']) == len(cpu['cases']) == 6
for old, new in zip(cpu['cases'], pending['cases']):
    assert old['case'] == new['case']
    assert new['cpu_gpu_foreground_iou'] >= 0.99
    assert digest(gpu / (new['case'] + '-labels.npy')) == new['labels_sha256']
    assert digest(root / 'cpu-r3' / (old['case'] + '-labels.npy')) == old['labels_sha256']
assert pending['worker_hello']['device'] == 'gpu'
assert pending['observed_gpu_processes']

hlo = {}
hlo_hashes = {}
for path in sorted((profile / 'hlo').glob('*gpu_after_optimizations.txt')):
    text = path.read_text()
    module = re.search(r'^HloModule ([^,]+)', text).group(1)
    program = int(re.match(r'module_(\d+)\.', path.name).group(1))
    hlo_hashes[str(path)] = digest(path)
    for number, line in enumerate(text.splitlines(), 1):
        match = re.search(r'^  %([^ ]+) = .* custom-call\(.*op_type="(Conv[23]D)" op_name="([^"]+)"', line)
        if match:
            key = (module, program, match.group(1))
            assert key not in hlo
            assert 'custom_call_target="__cudnn$convBiasActivationForward"' in line
            hlo[key] = dict(op_type=match.group(2), op_name=match.group(3), program_id=program,
                            hlo_path=str(path), hlo_line=number, hlo_sha256=digest(path))
assert {row['op_type'] for row in hlo.values()} == {'Conv2D', 'Conv3D'}

events = []
trace_hashes = {}
for path in sorted((profile / 'trace').rglob('*.xplane.pb')):
    trace_hashes[str(path)] = digest(path)
    space = xplane_pb2.XSpace()
    space.ParseFromString(path.read_bytes())
    assert not space.errors
    for plane in space.planes:
        if not plane.name.startswith('/device:GPU:'):
            continue

        def value(stat):
            field = stat.WhichOneof('value')
            result = getattr(stat, field) if field else None
            if field == 'ref_value':
                assert result in plane.stat_metadata
                return plane.stat_metadata[result].name
            return result.hex() if field == 'bytes_value' else result

        for line in plane.lines:
            for event in line.events:
                metadata = plane.event_metadata[event.metadata_id]
                stats = {plane.stat_metadata[stat.metadata_id].name: value(stat)
                         for stat in [*metadata.stats, *event.stats]}
                key = (stats.get('hlo_module'), stats.get('program_id'), stats.get('hlo_op'))
                if key not in hlo or event.duration_ps <= 0 or not stats.get('kernel_details'):
                    continue
                operation = hlo[key]
                if stats.get('name') == 'autotuner' or stats.get('name') != operation['op_name']:
                    continue
                if stats.get('program_id') != operation['program_id']:
                    continue
                if stats.get('tf_op') != 'StatefulPartitionedCall:StatefulPartitionedCall':
                    continue
                if not any(token in metadata.name for token in ('implicit_convolve', 'scudnn_', '_xmma_fprop_')):
                    continue
                assert stats.get('correlation_id', 0) > 0 and stats.get('scope_range_id', 0) > 0
                events.append(dict(op_type=operation['op_type'], operation=operation,
                                   kernel_name=metadata.name, plane=plane.name, stream=line.name,
                                   duration_ps=event.duration_ps, offset_ps=event.offset_ps,
                                   line_timestamp_ns=line.timestamp_ns, stats=stats,
                                   trace_path=str(path), trace_sha256=digest(path)))


def require_both(candidates):
    assert {event['op_type'] for event in candidates} == {'Conv2D', 'Conv3D'}


require_both(events)
controls = {}
for removed in ('Conv2D', 'Conv3D'):
    try:
        require_both([event for event in events if event['op_type'] != removed])
    except AssertionError:
        controls['missing_' + removed + '_rejected'] = True
    else:
        raise AssertionError('negative execution control admitted')
counts = collections.Counter(event['op_type'] for event in events)
evidence = dict(schema=1, accepted=True,
                scope='Positive-duration actual GPU compute kernels during unchanged normal XLA inference, excluding autotuning, layout conversion, offsets, copies and delay kernels. GPU trace module/instruction/program/name metadata joins the exact optimized HLO Conv2D and Conv3D operations.',
                decode_reference='Installed TensorFlow xplane_builder.h: reference-valued stats resolve through XPlane.stat_metadata, not event_metadata.',
                trace_files_sha256=trace_hashes, hlo_files_sha256=hlo_hashes,
                compute_kernel_counts=dict(counts), negative_controls=controls,
                verifier_sha256=digest(__file__),
                pending_capture_sha256=digest(pending_path),
                cpu_benchmark_sha256=digest(root / 'cpu-r3/benchmark.json'),
                events=events)
evidence_path = gpu / 'gpu-execution-verification.json'
assert not evidence_path.exists() and not (gpu / 'benchmark.json').exists()
evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')
accepted = dict(pending, accepted=True, gpu_execution_verification_pending=False,
                gpu_execution_verification_path=str(evidence_path),
                gpu_execution_verification_sha256=digest(evidence_path),
                observed_gpu_convolution_compute_kernels=dict(counts))
(gpu / 'benchmark.json').write_text(json.dumps(accepted, indent=2) + '\n')
print(json.dumps(dict(accepted=True, compute_kernel_counts=dict(counts), negative_controls=controls,
                      trace_files_sha256=trace_hashes, hlo_files_sha256=hlo_hashes,
                      examples={op: next(event for event in events if event['op_type'] == op)
                                for op in ('Conv2D', 'Conv3D')}), indent=2))
