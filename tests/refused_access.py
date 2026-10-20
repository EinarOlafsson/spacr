"""Make a file or folder refuse this process, on POSIX and on Windows alike.

``os.chmod(path, 0o000)`` is how the suite makes a file unreadable or a folder
unwritable, and on Windows it does nothing: Windows keeps only a read-only
attribute, ignores it for folders, and never uses mode bits to refuse a read.
Every "a file that cannot be read" test then passed its unreadable file
straight through and failed on Windows. The refusal there is an ACL deny
entry for the current account, removed again on the way out.

The deny entries are deliberately narrow -- read DATA, not READ_CONTROL --
because pytest's temporary folders grant access only through OWNER RIGHTS,
and a deny that also covers READ_CONTROL leaves an entry this process can no
longer read, and so no longer remove.
"""
from __future__ import annotations

import contextlib
import getpass
import os
import subprocess

_WINDOWS_RIGHTS = {"read": "(RD)", "write": "(W)"}


@contextlib.contextmanager
def refused(path, posix_mode: int, *, windows: str):
    """Refuse ``path`` for the duration of the block.

    :param path: the file or folder.
    :param posix_mode: the mode set on POSIX, as the test always did.
    :param windows: ``"read"`` (file data or folder listing) or ``"write"``
        (file data, or creating entries in a folder), denied on Windows.
    """
    path = str(path)
    if os.name == "nt":
        account = getpass.getuser()
        subprocess.run(["icacls", path, "/deny",
                        f"{account}:{_WINDOWS_RIGHTS[windows]}"],
                       check=True, capture_output=True, timeout=30)
        try:
            yield
        finally:
            subprocess.run(["icacls", path, "/remove:d", account],
                           check=True, capture_output=True, timeout=30)
        return
    before = os.stat(path).st_mode
    os.chmod(path, posix_mode)
    try:
        yield
    finally:
        os.chmod(path, before)
