import json
from pathlib import Path

TARGETS = {
    'test_a_half_built_screen_finishes_inside_its_open': 'queue',
    'test_a_navigation_that_arrives_mid_open_waits_for_it': 'convert',
    'test_opening_a_module_repolishes_its_screen_once[measure]': 'measure',
    'test_classify_controls_exist_before_the_first_screen_sheet': 'classify_merged',
    'test_a_missing_package_is_named_rather_than_silent': 'measure',
    'test_visible_home_tile_opens_real_screen[measure]': 'regression',
    'test_the_run_is_given_exactly_what_a_window_that_built_everything_gives': 'classify_merged',
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
