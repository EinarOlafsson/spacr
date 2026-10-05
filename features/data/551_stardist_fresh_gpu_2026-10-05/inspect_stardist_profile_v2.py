import argparse
import hashlib
import json
import os
from pathlib import Path

from tensorflow.tsl.profiler.protobuf import xplane_pb2

parser = argparse.ArgumentParser()
parser.add_argument('profile', type=Path)
args = parser.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
paths = list((args.profile / 'trace').rglob('*.xplane.pb'))
assert paths
planes, events = [], []
for path in paths:
    space = xplane_pb2.XSpace()
    space.ParseFromString(path.read_bytes())
    assert not space.errors, list(space.errors)
    for plane in space.planes:
        planes.append({'name': plane.name, 'lines': len(plane.lines),
                       'events': sum(len(line.events) for line in plane.lines)})
        if not plane.name.startswith('/device:GPU:'):
            continue

        def stat_value(stat):
            field = stat.WhichOneof('value')
            value = getattr(stat, field) if field else None
            if field == 'ref_value':
                assert value in plane.stat_metadata, value
                return plane.stat_metadata[value].name
            if field == 'bytes_value':
                return value.hex()
            return value

        for line in plane.lines:
            for event in line.events:
                if event.duration_ps <= 0:
                    continue
                metadata = plane.event_metadata[event.metadata_id]
                stats = {plane.stat_metadata[stat.metadata_id].name: stat_value(stat)
                         for stat in [*metadata.stats, *event.stats]}
                events.append({'plane': plane.name, 'line': line.name,
                               'name': metadata.name, 'display_name': metadata.display_name,
                               'duration_ps': event.duration_ps, 'offset_ps': event.offset_ps, 'line_timestamp_ns': line.timestamp_ns, 'stats': stats})
report = {'planes': planes, 'gpu_positive_duration_events': len(events),
          'stat_names': sorted({key for event in events for key in event['stats']}),
          'trace_files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
          'events': events}
target = args.profile / 'decoded-gpu-events-v2.json'
assert not target.exists()
target.write_text(json.dumps(report, indent=2) + '\n')
summary = {key: value for key, value in report.items() if key != 'events'}
summary['first_gpu_events'] = events[:15]
summary['convolution_related_gpu_event_samples'] = [event for event in events
                                                   if 'conv' in json.dumps(event).lower()][:25]
print(json.dumps(summary, indent=2))
