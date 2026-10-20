import atexit
import json
import os
from pathlib import Path
import socket
import sys

folder = Path(__file__).parent
events = []
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_sendto = socket.socket.sendto
original_getaddrinfo = socket.getaddrinfo
def refuse(kind, value):
    events.append({'kind': kind, 'address': repr(value)})
    raise RuntimeError('Network prohibited in zoom-only CI baseline')
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('connect', address)
    return original_connect(self, address)
def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('connect_ex', address)
    return original_connect_ex(self, address)
def sendto(self, data, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('sendto', args)
    return original_sendto(self, data, *args)
def getaddrinfo(host, *args, **kwargs):
    refuse('getaddrinfo', host)
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.socket.sendto = sendto
socket.getaddrinfo = getaddrinfo
atexit.register(lambda: (folder / 'final-network.json').write_text(json.dumps(events, indent=2) + '\n'))
(folder / 'final-startup.json').write_text(json.dumps({'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'qt_platform': os.environ.get('QT_QPA_PLATFORM')}, indent=2) + '\n')
import pytest
raise SystemExit(pytest.main(['-q', '--junitxml=/mnt/wd4tb/scratch/n441-scanr-sourced-stem-20261015/final.xml', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_scanr_user_stems_disclose_the_inferred_tiff_suffix', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_convention_parses_every_example_it_claims[scanr]', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_example_fills_every_group_the_importer_reads[scanr]', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_no_convention_claims_a_filename_written_by_another', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_every_record_carries_the_evidence_for_itself[scanr]', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_the_three_built_in_patterns_are_unchanged_character_for_character', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_the_custom_arm_still_wraps_the_users_expression_the_same_way', 'tests/test_a_metadata_type_for_every_common_microscope.py::test_the_image_format_is_interpolated_exactly_as_it_always_was']))
