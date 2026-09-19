# Notes from `spacr/qt/model_install.py`

## Why it exists (item 419 point 3, 2026-09-19)

- Two places installed a segmentation backend with a synchronous
  `subprocess.run(pip install ...)` on the GUI thread: the Make Masks Mode
  box (c60d48e35) and `spacr/qt/widgets/model_zoo_picker.py`'s
  `install_backend_package` (d1ee35065). Both froze the window for the whole
  install and said so in their own warning. This module is the non-blocking
  replacement. The picker is another session's file and was NOT changed here;
  switching `install_backend_package` to `PackageInstall` is what it needs to
  stop freezing too.
- `PackageInstall` is a `QProcess`, not a thread running `subprocess`: the
  output arrives through `readyReadStandardOutput` on the event loop, and
  there is no Python thread to join on shutdown.
- `_RUNNING`: an install or download keeps itself alive until it ends. A
  `QProcess` deleted with its owner is killed, and the Mask module's settings
  form is rebuilt while the screen is open (the owning combo goes with it);
  pip killed half way through an install can leave the environment broken.
  Callers therefore pass no parent. `CheckpointDownload` owns an unparented
  `QThread` for the same reason: a `QThread` destroyed while running is a
  `qFatal`, not an exception.
- `CheckpointDownload.cancel` returns at once. `model_zoo.fetch` polls the
  cancel callable between chunks and removes its partial file, so the thread
  ends on its own; waiting for it would block the GUI for up to a network
  timeout.
- `can_install_packages`: a frozen build has no pip (its interpreter is the
  application binary), so the offer says so instead of starting a process
  that cannot work.
- `SegmentationBackendCombo` is a `QComboBox` whose rows store the setting's
  own values, so `SettingsWidgets._read_widget` and `set_value_for_key` treat
  it as any other dropdown. A loaded settings file naming a backend that is
  not installed still selects it (greyed): the run then fails with the
  backend's own ImportError naming the extra, which is more honest than
  silently switching to Cellpose.
