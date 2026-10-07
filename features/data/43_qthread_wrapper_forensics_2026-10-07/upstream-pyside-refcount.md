# Upstream Shiboken wrapper-map lead, 2026-10-07

The installed PySide6/shiboken6 is 6.11.2. Its `QtCore.abi3.so` and
`libshiboken6.abi3.so.6.11` hashes are in the main receipt. The local
Shiboken binary exports `BindingManager::releaseWrapper(SbkObject*)` but not
`BindingManager::unregisterWrapper(SbkObject*)` according to `nm -D | c++filt`.

In [official v6.11.2 `basewrapper.cpp`](https://github.com/qtproject/pyside-pyside-setup/blob/v6.11.2/sources/shiboken6/libshiboken/basewrapper.cpp),
`Shiboken::Object::destroy(self, cppData)` acquires the GIL but does not take
an explicit strong Python reference to `self`. It calls `clearReferences`,
which decrements its stored Python referents, then `_destroyParentInfo`.
For an object with a parent, that helper calls `removeParent(self, false)`,
which decrements the parent-held Python wrapper reference. The source warns
that the wrapper can become invalid at that point. The subsequent
`hasWrapper(cppData)` check is intended to prevent touching a wrapper already
removed from the map. If it passes, `releaseWrapper(self)` reads the wrapper's
`d->cptr` array, the same access identified at the local native fault PC.

The official post-6.11.2 commit
[PYSIDE-3452 `7646cc22a72b`](https://github.com/qtproject/pyside-pyside-setup/commit/7646cc22a72bc9510226ff21968b160794424374)
fixes an independently reproduced wrapper-map lifetime defect. Before that
fix, Python wrapper deallocation left a zero-refcount wrapper registered
until `deallocData`, although a weakref callback, `__del__`, or another Python
reentry could look it up and resurrect it. The upstream test makes such a
lookup during weakref notification. The fix unregisters the wrapper right
after `PyObject_GC_UnTrack`, before any Python callback; it also makes
`unregisterWrapper` tolerate a null pointer array. This commit was dated
2026-08-22, after the v6.11.2 source release dated 2026-08-18.

This is a concrete upstream mechanism compatible with an invalid wrapper
pointer at `releaseWrapper`. It is **not** proof that the spaCR crash took that
path. The original GDB capture has no `SbkObject*` register, weakref callback
identity, or native QThread pointer; the bounded real-reader probe in this
folder did not reproduce a crash. No app source change, PySide downgrade,
guard change, or acceptance claim follows from this finding. A future native
capture needs the wrapper and `d->cptr` values at the actual fault to tie it
to a specific QObject lifetime.
