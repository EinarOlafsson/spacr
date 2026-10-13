# e41 live-preview dialog ownership failure

Hosted Coverage9 failed two original assertions because a successful `close()`
on a never-shown `LiveSettingsDialog` did not invoke Qt's `done()` path. The
dialog still owned panel controls when it could be destroyed. A shown-dialog
close did invoke `done()`. The source fix returns borrowed controls through
the existing `done()` cleanup after a successful hidden close. An ignored
close leaves the visible dialog and its controls intact.

The frozen hosted log contains other Coverage9 failures outside this repair.
The focused Qt 6.12.0 run passed 23 owning and adjacent lifetime cases under
4 GiB. Coverage observes both branches of the new close guard. This is a
dialog-ownership repair, not a native-crash diagnosis or a full CI verdict.

Run `python verify.py --git --current` on the source-bound revision.
