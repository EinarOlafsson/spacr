import json
import os
import sys
from pathlib import Path

import PySide6
from PySide6.QtCore import qVersion

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path(__file__).parent
mode = sys.argv[1]
cgroup = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
cgroup_root = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
receipt = {'pid': os.getpid(), 'cgroup': cgroup, 'memory_max': (cgroup_root / 'memory.max').read_text().strip(),
           'memory_swap_max': (cgroup_root / 'memory.swap.max').read_text().strip(),
           'cpus_allowed': sorted(os.sched_getaffinity(0)), 'pyside6': PySide6.__version__, 'qt': qVersion()}
(root / (mode + '-startup.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
sys.path.insert(0, os.getcwd())
import pytest
targets = ['tests/test_evos_documented_optional_z_441.py']
if mode == 'final':
    targets += [
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_the_three_built_in_patterns_are_unchanged_character_for_character',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_the_custom_arm_still_wraps_the_users_expression_the_same_way',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_the_image_format_is_interpolated_exactly_as_it_always_was',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_convention_parses_every_example_it_claims[evos]',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_convention_parses_every_example_it_claims[arrayscan]',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_example_fills_every_group_the_importer_reads[evos]',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_example_fills_every_group_the_importer_reads[arrayscan]',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_no_convention_claims_a_filename_written_by_another',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_record_carries_the_evidence_for_itself[evos]',
        'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_record_carries_the_evidence_for_itself[arrayscan]',
        'tests/test_cov_utils_meta_naming.py::test_extract_filename_metadata_missing_group_is_reported',
        'tests/test_cov_utils_meta_naming.py::test_extract_filename_metadata_plateid_falls_back_to_src_basename',
    ]
import hashlib
pins = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [
    'spacr/regex_infer.py', 'spacr/utils.py', 'spacr/io.py', 'spacr/convert.py',
    'tests/test_evos_documented_optional_z_441.py', 'tests/test_a_metadata_type_for_every_common_microscope.py']}
(root / (mode + '-source-pins.json')).write_text(json.dumps(pins, indent=2) + '\n')

blocked = []
def deny_network(event, args):
    if event == 'socket.getaddrinfo' or (event == 'socket.connect' and isinstance(args[1], tuple)):
        blocked.append(event)
        raise RuntimeError('External network forbidden in fixture verification')
sys.addaudithook(deny_network)
try:
    result = pytest.main([*targets, '-q', '--tb=short',
        '--junitxml=' + str(root / (mode + '.xml')), '--basetemp=' + str(root / (mode + '-tmp')),
        '-o', 'cache_dir=' + str(root / 'pytest-cache')])
finally:
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == pin for name, pin in pins.items())
    (root / (mode + '-network.json')).write_text(json.dumps({'blocked_network_events': blocked}) + '\n')
raise SystemExit(result)
