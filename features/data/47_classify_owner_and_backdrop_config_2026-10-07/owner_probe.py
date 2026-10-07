import json
from PySide6.QtWidgets import QApplication

def pytest_runtest_call(item):
    app = QApplication.instance()
    if app is not None:
        print('OWNER_CENSUS', json.dumps({'node': item.nodeid, 'widgets': len(app.allWidgets()), 'top_levels': len(app.topLevelWidgets())}), flush=True)
