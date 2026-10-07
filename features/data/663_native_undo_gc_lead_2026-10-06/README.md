# Native undo-stack GC investigation

The [original author report](https://forum.qt.io/topic/164878/shiboken6-6.11.1-sigsegv-in-qundostack-clear-pushed-qundocommand-double-deleted-when-stack-and-command-are-cyclic-garbage) suggested a specific upstream ownership lead. Independent local probes reproduce a parentless command-before-stack cyclic-GC crash under PySide6/Shiboken6.11.2 on both Python3.12.13 and3.13.14. The actual GDB backtrace reaches QUndoStack::clear and its destructor before CPython garbage collection. This is a native reproduction of the library lead.

Both allocation orders pass thirty collections with a QObject parent, even when that parent is cyclic/unreachable; stack-before-command also passes parentless. Explicit clear passes the original Python3.12 case. Runs are isolated subprocesses under2GiB, CUDA hidden, with core dumps disabled. Exact scripts/results and native trace are archived.

All four production SettingsWidgets callers already pass a QWidget parent; GateEditorPanel parents its stack to itself. Make Masks uses its own snapshot history. No production change is justified by this lead alone. The earlier actual Home/Preferences probes did not contain Mask/Annotate undo state, but neither historical spaCR crash has a native C-stack matching this reproduction. The installed Save and hosted puncta causes remain OPEN.
