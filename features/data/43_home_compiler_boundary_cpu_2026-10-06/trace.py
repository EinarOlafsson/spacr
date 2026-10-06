import os, sys, traceback, threading, time, json
os.environ['SPACR_TIMING']='1'
os.environ['SPACR_TIMING_IMPORTS']='0'
start=time.perf_counter()
ready=False
seen=False
def audit(event,args):
    global seen
    if event=='import' and args[0]=='scipy' and not seen:
        seen=True
        print('FIRST SCIPY', threading.current_thread().name, time.perf_counter()-start, 'home_ready', ready, flush=True)
        traceback.print_stack(file=sys.stdout)
sys.addaudithook(audit)
from spacr.qt import timing
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

def observe(entry):
    global ready
    if entry.get('detail')=='__home__':
        ready=True
        print('HOME READY',json.dumps(entry), 'scipy_present', 'scipy' in sys.modules,flush=True)
        QTimer.singleShot(4000, QApplication.instance().quit)
timing.subscribe_readiness(observe)
import spacr.qt
sys.exit(spacr.qt.run(['--no-setup']))
