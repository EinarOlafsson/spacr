# Installed 6.11.2 reproduction of PYSIDE-3452, 2026-10-07

Qt's [PYSIDE-3452 fix and regression test](https://github.com/qtproject/pyside-pyside-setup/commit/7646cc22a72bc9510226ff21968b160794424374)
use the Shiboken samplebinding extension `sample.ObjectType`. That extension
is not shipped in the installed PySide6 wheel: running the upstream test
unchanged stops at `ModuleNotFoundError: No module named 'shiboken_paths'`.
The accompanying `dealloc-resurrect-qobject.py` therefore preserves its exact
weakref-callback `getCppPointer` → `wrapInstance` → deletion sequence while
using installed `PySide6.QtCore.QObject` in place of `sample.ObjectType`.
It imports no spaCR code.

The script was run in a core-disabled, CUDA-hidden, 4 GiB capped child with
Python 3.12.13, PySide6/shiboken6 6.11.2 and GDB. The first archived native
log shows the weakref callback recovered the **same** apparently valid Python
wrapper, then SIGSEGV at `BindingManager::releaseWrapper+0x50`, the exact
faulting function offset in the contextual `QThreadWrapper` GDB capture.
Registers show a damaged C++ pointer-array address (`r15=0x555555e9b` while
the original QObject pointer was `0x555555e9c090`). A second captured run
also faulted after the same callback, at another Shiboken address; its full
native backtrace is archived. The crash-only GDB command used no breakpoint.

This proves that the installed 6.11.2 binding has the upstream stale-wrapper
map defect and can fail at the same native fault site. It does not prove that
the spaCR QThread crash involved a weakref callback or resurrection: its
original GDB capture did not record the wrapper pointer or such a callback.
The ordinary small-mask reader probe in this folder passed twelve deferred
deletions under the GUI GC timer.

As of 2026-10-07, the official GitHub release tags and PyPI stable releases
for both PySide6 and shiboken6 end at 6.11.2. Although an upstream 6.11.3
**branch** exists, it is untagged and its current `basewrapper.cpp` does not
contain this fix. The dev commit does contain it. There is thus no tagged,
stable patched wheel to validate as a compatibility replacement. No package
pin, downgrade, app patch, or CI guard adjustment was made.
