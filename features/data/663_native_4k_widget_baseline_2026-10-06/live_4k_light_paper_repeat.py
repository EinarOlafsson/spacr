import hashlib
import json
import resource
from pathlib import Path

from PySide6.QtWidgets import QApplication

import live_4k_probe as probe


app = QApplication([])
app.setQuitOnLastWindowClosed(False)
store = probe.prefs.QSettings(
    str(probe.OUT.parent / 'probe-light-repeat-prefs.ini'),
    probe.prefs.QSettings.IniFormat)
probe.prefs._settings = lambda: store
probe.prefs._set_ambient_custom_colors(('#3b82f6', '#ff00ff'))
record = probe.measure(app, 'data_art_tissue_facets', '#f6f7f9',
                       0.25, True, False)
path = probe.OUT.parent / 'live_4k_light_paper_repeat.json'
path.write_text(json.dumps({
    'source_sha256': hashlib.sha256(probe.EXPECTED.read_bytes()).hexdigest(),
    'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    'record': record}, indent=2) + '\n')
print(json.dumps(record), flush=True)
