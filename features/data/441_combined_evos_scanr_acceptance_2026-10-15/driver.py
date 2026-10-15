import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015')
folder = Path(__file__).parent
head = '8fa5d1bb8ced57df64f9f3dab1759d512c4a2556'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == head
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = ['spacr/regex_infer.py', 'spacr/utils.py', 'spacr/io.py', 'spacr/convert.py',
         'tests/test_a_metadata_type_for_every_common_microscope.py', 'tests/test_evos_documented_optional_z_441.py']
paths += ['spacr/qt/i18n_catalogs/' + name + '.py' for name in ['en', 'sv', 'de', 'es', 'fr', 'pt', 'hi', 'ko', 'is', 'zh_CN']]
before = {p: sha(root / p) for p in paths}
(folder / 'before-pins.json').write_text(json.dumps(before, indent=2) + '\n')
blocked = []
def offline(event, args):
    if event == 'socket.getaddrinfo' or event == 'socket.connect' and isinstance(args[1], tuple):
        blocked.append(event)
        raise RuntimeError('No external network permitted in combined metadata acceptance')
sys.addaudithook(offline)
sys.path.insert(0, str(root))
import PySide6
from PySide6.QtCore import qVersion, QLibraryInfo
print(json.dumps({'python': sys.executable, 'qt': qVersion(), 'pyside': PySide6.__file__, 'plugins': QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)}), flush=True)
import pytest
class Capture:
    nodeids = []
    outcomes = []
    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed or report.skipped:
            self.outcomes.append({'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome})
capture = Capture()
prefix = 'tests/test_a_metadata_type_for_every_common_microscope.py::'
nodes = [prefix + 'test_no_convention_claims_a_filename_written_by_another',
         prefix + 'test_every_convention_parses_every_example_it_claims[evos]',
         prefix + 'test_every_convention_parses_every_example_it_claims[scanr]',
         prefix + 'test_scanr_user_stems_disclose_the_inferred_tiff_suffix',
         'tests/test_evos_documented_optional_z_441.py::test_documented_manual_name_is_not_misidentified_as_arrayscan']
status = pytest.main(['-q', '-s', '-p', 'pytestqt.plugin', *nodes, '--junitxml=' + str(folder / 'tests.xml'), '--basetemp=' + str(folder / 'pytest-tmp'), '-o', 'cache_dir=' + str(folder / 'pytest-cache')], plugins=[capture])
after = {p: sha(root / p) for p in paths}
actual = {}
for name in ['spacr.regex_infer', 'spacr.utils', 'spacr.io', 'spacr.convert']:
    module = sys.modules.get(name)
    if module is None:
        continue
    expected = root / (name.replace('.', '/') + '.py')
    path = Path(module.__file__).resolve()
    assert path == expected.resolve() and sha(path) == before[str(expected.relative_to(root))], name
    actual[name] = {'path': str(path), 'sha256': sha(path)}
assert 'spacr.regex_infer' in actual
result = {'head': head, 'pytest_exit': status, 'collected_nodeids': capture.nodeids, 'outcomes': capture.outcomes, 'actual_modules': actual, 'all_16_source_test_catalog_pins_unchanged': before == after, 'pins': before, 'network_events': blocked}
(folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
assert capture.nodeids == nodes
assert before == after
assert len([x for x in capture.outcomes if x['when'] == 'call' and x['outcome'] == 'passed']) == 5
assert not any(x['outcome'] == 'skipped' for x in capture.outcomes)
raise SystemExit(status)
