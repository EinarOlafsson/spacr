import json
from pathlib import Path

TARGETS = {
    "test_main_window_constructs_and_switches": "mask",
    "test_a_screen_can_be_rebuilt_before_it_was_ever_built": "mask",
}

current = None
original = None
records = []

def pytest_configure(config):
    global original
    from spacr.qt.app import MainWindow
    original = MainWindow.__init__
    def seeded(self, *args, **kwargs):
        from spacr import restart_state
        target = TARGETS[current.name]
        assert restart_state._save_session(target, {})
        assert restart_state._last_session()['module'] == target
        record = {'nodeid': current.nodeid, 'persisted_target': target, 'initial_app': kwargs.get('initial_app')}
        records.append(record)
        original(self, *args, **kwargs)
        record['screens_after_constructor'] = list(self._screens)
        record['starts_at_home'] = self._stack.currentWidget() is self._startup
    MainWindow.__init__ = seeded

def pytest_runtest_setup(item):
    global current
    current = item

def pytest_unconfigure(config):
    if original is not None:
        from spacr.qt.app import MainWindow
        MainWindow.__init__ = original
    output = Path(config.getoption('basetemp')).parent / ('seeded-' + Path(config.getoption('basetemp')).name + '.json')
    output.write_text(json.dumps(records, indent=2) + '\n')
