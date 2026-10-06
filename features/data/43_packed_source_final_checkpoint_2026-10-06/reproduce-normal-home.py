import os, sys, threading, traceback, time
os.environ['SPACR_TIMING']='0'
os.environ['SPACR_TIMING_IMPORTS']='0'
start=time.perf_counter()
ready=False
seen=False
first_scipy_ready=False
def audit(event,args):
    global seen, first_scipy_ready
    if event=='import' and args[0]=='scipy' and not seen:
        seen=True
        first_scipy_ready=ready
        print('FIRST SCIPY',threading.current_thread().name,time.perf_counter()-start,'home_ready',ready,flush=True)
        traceback.print_stack(file=sys.stdout)
sys.addaudithook(audit)
from spacr.qt import app, timing
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
original=app._queue_ambient_compilation
def complete():
    global ready
    ready=True
    print('NORMAL POST-PAINT CALLBACK',time.perf_counter()-start,'scipy_present','scipy' in sys.modules,'reports',len(timing._READINESS),flush=True)
    original()
    QTimer.singleShot(2500,QApplication.instance().quit)
app._queue_ambient_compilation=complete
import spacr.qt
result=spacr.qt.run(['--no-setup'])
assert ready and not timing._READINESS
assert seen and first_scipy_ready, "SciPy did not first import after genuine readiness"
sys.exit(result)
