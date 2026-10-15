"""Run only the existing nine actual compiled metadata-widget consumers."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

folder = pathlib.Path(__file__).parent
config = json.loads((folder / 'config.json').read_text())
assert config['ready'] is True, 'Root authorization after producer terminal is required'
root = pathlib.Path(config['root'])
assert pathlib.Path.cwd() == root
assert isinstance(config['head'], str) and len(config['head']) == 40, 'Full final Root commit is required'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == config['head']
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in {**config['source_pins'], **config['approved_catalog_review_pins']}.items():
    assert sha(root / name) == expected, name
paths = list(config['source_pins'])
paths += ['spacr/qt/i18n_catalogs/' + name + '.py' for name in ['en', *config['languages']]]
paths += ['docs/i18n/reviewed/runtime/' + name + '/2026-10-15-metadata-convention-status.json' for name in config['languages']]
before = {name: sha(root / name) for name in paths}
blocked = []
model_guard_events = []
def offline(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).lower()
        if name.endswith(('.pt', '.pth', '.ckpt', '.onnx', '.safetensors')):
            model_guard_events.append(name)
            raise RuntimeError('No checkpoint/model decoding permitted in actual metadata consumers')
    if event == 'socket.getaddrinfo' or event == 'socket.connect' and isinstance(args[1], tuple):
        blocked.append(event)
        raise RuntimeError('No external network permitted in actual metadata consumers')
sys.addaudithook(offline)
class NoCollector:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('build_i18n_catalogs', 'tools.build_i18n_catalogs'):
            raise RuntimeError('No canonical collector or runtime writer permitted')
        if fullname.split('.')[0] in {'cellpose', 'stardist', 'instanseg', 'omnipose', 'onnxruntime', 'transformers', 'safetensors'}:
            model_guard_events.append(fullname)
            raise RuntimeError('No model decoder backend import permitted')
sys.meta_path.insert(0, NoCollector())
sys.path.insert(0, str(root))
import PySide6
from PySide6.QtCore import QLibraryInfo, qVersion
print(json.dumps({'python': sys.executable, 'pyside': PySide6.__file__, 'qt': qVersion(), 'qt_plugins': QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)}, ensure_ascii=False), flush=True)
import pytest
class Capture:
    nodeids = []
    outcomes = []
    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.skipped or report.failed:
            self.outcomes.append({'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome})
capture = Capture()
base = 'tests/qt/test_n441_metadata_actual_locales.py::test_actual_metadata_widgets_use_all_ten_installed_translations'
nodes = [base + '[' + language + ']' for language in config['languages']]
status = pytest.main(['-q', '-s', '-p', 'pytestqt.plugin', *nodes, '--junitxml=' + str(folder / 'actual-nine.xml'),
                      '--basetemp=' + str(folder / 'pytest-tmp'), '-o', 'cache_dir=' + str(folder / 'pytest-cache')], plugins=[capture])
after = {name: sha(root / name) for name in paths}
module_provenance = {}
for module_name, relative in config['module_paths'].items():
    module = sys.modules.get(module_name)
    assert module is not None and getattr(module, '__file__', None), module_name
    actual = pathlib.Path(module.__file__).resolve()
    assert actual == (root / relative).resolve(), (module_name, str(actual))
    assert sha(actual) == config['source_pins'][relative], module_name
    module_provenance[module_name] = {'path': str(actual), 'sha256': sha(actual)}
result = {'head': config['head'], 'source_catalog_review_pins': before, 'all_pins_unchanged': after == before,
          'collected_nodeids': capture.nodeids, 'outcomes': capture.outcomes, 'pytest_exit': status,
          'blocked_network_events': blocked, 'blocked_model_events': model_guard_events,
          'actual_imported_modules': module_provenance, 'driver_sha256': sha(pathlib.Path(__file__)),
          'config_sha256': sha(folder / 'config.json'),
          'scope': 'Existing nine actual widget callbacks with compiled tr and real temporary persisted preferences; existing stableIDs/order/raw input-byte assertions retained.'}
(folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
assert after == before, 'Source/catalog/review inputs changed during actual consumers'
assert capture.nodeids == nodes, 'Unexpected collection scope'
assert sum(row['when'] == 'call' and row['outcome'] == 'passed' for row in capture.outcomes) == 9
assert not any(row['outcome'] == 'skipped' for row in capture.outcomes), 'No skipped acceptance'
raise SystemExit(status)
