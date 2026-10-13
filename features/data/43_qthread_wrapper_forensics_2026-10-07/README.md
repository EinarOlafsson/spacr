# N43 native QThread wrapper identification, 2026-10-07

This supplements the source-bound Qt shard 0 receipt in
`features/data/43_qt0_parent_source_sigsegv_2026-10-07`. A bounded local
three-file replay also stopped in GDB on SIGSEGV while a parent mask source
test was inside `qtbot.waitUntil`. The original GDB log is preserved as
`gdb-reproducing-cohort.log.gz`. It was run under a 4 GiB cap, CUDA hidden,
Python 3.12.13, PySide6/Qt 6.11.2, from the primary-selector diagnostic
worktree before commit `6aa514f5b2`; the exact uncommitted file hashes at
the GDB stop were not recorded. Its native stack was:

```text
Shiboken::BindingManager::releaseWrapper
Shiboken::Object::destroy
PySide6/QtCore.abi3.so + unknown wrapper destructor frames
QObject::event → sendPostedEvents → QEventLoop::exec
```

The installed libraries used for disassembly have SHA-256
`d040e13f904ba5f6f43f5d97d8610b0eb53bfb12721710dfea16829355fe0cb1`
(`PySide6/QtCore.abi3.so`) and
`c0ddb602926c9255ce6655f0ec0c7afc731016c99f96c4ada6e0595cddf29dae`
(`libshiboken6.abi3.so.6.11`). The GDB frames and the ELF instructions give
a unique type identification:

1. The fault PC `0x7ffff5c4c670` is `releaseWrapper+0x50`:
   `mov (%r15,%rbx,8),%rsi`, dereferencing a native-pointer slot. In this
   function, `rbp` holds the `SbkObject*`; its value was not captured.
2. QtCore frame #2, runtime `0x7fffae52ca00`, is the return address after
   `call Shiboken::Object::destroy` at ELF offset `0x12c9fb` (return
   `0x12ca00`). Among the destroy callsites, this is the unique offset
   compatible with a page-aligned mapped base, `0x7fffae400000`.
3. Frame #3, runtime `0x7fffae52ca18`, maps to offset `0x12ca18` in the
   immediately adjacent deleting-destructor thunk. The function calls
   `Object::destroy`, then jumps to `QThread::~QThread()`; the thunk calls
   that function and deletes the C++ object.
4. QtCore vtable relocation at `0x396730` points to this destructor, and
   its typeinfo name at `0x2d9d20` is `14QThreadWrapper`.

The committed excerpts `shiboken-release-wrapper.txt`,
`qtcore-destructor.txt`, `qtcore-relocations.txt`, and `qtcore-rtti.txt`
make the derivation checkable without rerunning the flaky cohort. The
specific Python-owned QThread instance remains unidentified: the active
`PrimaryMaskSelector._SourceWorker` is a candidate because `_finished`
posts `deleteLater()` during the event loop, but a queued worker from a
preceding test could have been deleted instead. Five subsequent GDB
three-file runs on committed `6aa514f5b2` passed 31/31 and yielded no fault
registers; their different timing and source identity do not prove a fix.
No production change or acceptance claim follows from this type evidence.
