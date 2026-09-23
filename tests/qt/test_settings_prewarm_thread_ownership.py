"""Cold settings imports leave path-driven redraws on the GUI thread."""
import os
import subprocess
import sys
import textwrap


def test_cold_prewarm_owns_signals_on_gui_and_delivers_worker_answers(tmp_path):
    script = textwrap.dedent('''
        import sys
        import threading
        import time
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from spacr.qt import thread_guard
        thread_guard.install()
        from spacr.qt.app import _start_settings_prewarm
        assert 'spacr.qt.path_probe' not in sys.modules, 'must exercise a cold import'
        warm = _start_settings_prewarm()
        warm.join(30)
        assert not warm.is_alive()
        assert 'spacr.qt.screens.settings_model' in sys.modules
        assert 'spacr.qt.imagery' in sys.modules
        from spacr.qt import path_probe
        assert path_probe.probes.thread() == app.thread()
        assert thread_guard.born_off_thread() == []
        received = []
        def redraw(path, present):
            received.append((path, present, QThread.currentThread() == app.thread()))
        path_probe.probes.answered.connect(redraw)
        worker = threading.Thread(target=lambda: path_probe.probes.answered.emit('image.tif', True))
        worker.start()
        worker.join(5)
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert received == [('image.tif', True, True)], received
        assert thread_guard.offences() == []
    ''')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                       HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / 'config'))
    result = subprocess.run([sys.executable, '-c', script], env=environment,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
