import os
from pathlib import Path
import json
import runpy
import sys
import traceback

stage = Path(__file__).parent
for key, folder in [('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),('XDG_CACHE_HOME','cache'),('SPACR_HOME','app-state'),('SPACR_LOG_DIR','logs')]:
    target = stage / folder
    target.mkdir(exist_ok=True)
    os.environ[key] = str(target)
os.environ.update(QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='',SPACR_NO_SETUP='1',SPACR_LANGUAGE='en',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import spacr
from PySide6.QtWidgets import QApplication, QLabel, QAbstractButton
from PySide6.QtCore import QTimer
original_exec = QApplication.exec
result = {'passed':False,'version':spacr.__version__,'package':spacr.__file__,'python':sys.executable,'platform':'offscreen','entry_point':str(Path(sys.executable).parent / 'spacr')}
assert Path(spacr.__file__).resolve().is_relative_to(Path('/home/carruthers/Documents/repo/spacr'))

def inspect_and_exit():
    app = QApplication.instance()
    try:
        windows = [w for w in app.topLevelWidgets() if type(w).__name__ == 'MainWindow' and w.isVisible()]
        assert len(windows) == 1, [(type(w).__name__,w.isVisible()) for w in app.topLevelWidgets()]
        window = windows[0]
        assert window._stack.currentWidget() is window._startup
        texts = sorted({w.text() for w in window._startup.findChildren(QLabel) + window._startup.findChildren(QAbstractButton) if w.text()})
        assert any('Measure' in text for text in texts), texts
        assert any('Mask' in text for text in texts), texts
        screenshot = stage / 'installed-home.png'
        assert window.grab().save(str(screenshot))
        result.update(passed=True,title=window.windowTitle(),home_text=texts,screenshot=str(screenshot))
    except Exception:
        result['failure'] = traceback.format_exc()
    (stage / 'launch-acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    app.exit(0 if result['passed'] else 1)

def scheduled_exec(self):
    QTimer.singleShot(5000,inspect_and_exit)
    return original_exec()
QApplication.exec = scheduled_exec
sys.argv = ['spacr']
runpy.run_path(result['entry_point'],run_name='__main__')
