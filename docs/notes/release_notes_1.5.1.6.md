## spaCR 1.5.1.6

### Upgrading

- If you go back to 1.5.1.5 after an interrupted 1.5.1.6 installation, 1.5.1.5 reports that another installer holds the installation, because it cannot read the newer lock. If no spaCR installer is running, remove the locks, then retry:
  - macOS: `~/Library/Application Support/.spaCR.spacr-installer-lock`, `~/Library/Application Support/spaCR/.venv.spacr-update-lock` and `~/Library/Application Support/spaCR/.installing` (folders).
  - Linux: `~/.local/share/.spacr.spacr-installer-lock` and `~/.local/share/spacr/.venv.spacr-update-lock` (folders; under `$XDG_DATA_HOME` when it is set).
  - Windows: `%LOCALAPPDATA%\.spaCR.spacr-installer-lock` and `%LOCALAPPDATA%\spaCR\.venv.spacr-update-lock` (files).
  - A folder of the same name ending in `.takeover` beside any of them can go too.
  For an installation in another folder, the locks sit beside and inside that folder under the same names.

### Improved

- An installer lock records its owner: process, start, host, user, process group (macOS, Linux) or child processes (Windows). A later installer takes over a lock only when that owner and everything it started are provably gone, and otherwise names the process that still holds it.
- A lock written by a newer installer is never taken over by an older one. When its process is gone, the older installer names the lock file to remove instead of a process that has ended.
