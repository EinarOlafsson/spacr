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

## Where a backend installs changed (item 423, 2026-09-19)

- This module was written to the maintainer's 2026-09-16 words, "install
  them in the spacr environment". On 2026-09-19, answering item 423's open
  questions, he said instead: "Isolated env per backend! But with the
  addition of adding cellpose 3 and its cyto, nucleus, and cyto2 and cyto3
  models." So `SegmentationBackendCombo.offer_install` now opens the Model
  Zoo's own install dialog, which builds the backend a venv of its own under
  `~/.spacr/backends/<name>` off the GUI thread, with progress and Cancel,
  and never changes spaCR's own environment. The gesture the maintainer
  asked for on 2026-09-16 -- a greyed row that installs itself when it is
  chosen -- is unchanged.
- It was not only a preference. Item 423 changed
  `model_zoo.INSTALLABLE_BACKENDS`'s second field from a pip extra
  (`spacr[samcell]`) to `backend:samcell`, the zoo's own URI for "installs
  as an environment", so `backend_row(name)[2]` is no longer something pip
  can be handed. Keeping the old route would have run
  `pip install "backend:samcell"`.
- `missing()` therefore asks `_segmentation_backends._backend_state(name)`
  rather than importing: a backend is installed when it has an environment
  of its own OR when its package imports here, which is what an older spaCR
  left behind and what item 423 still honours. A state that cannot be read
  falls back to the import, so an unreadable folder does not grey a row that
  works.
- `PackageInstall` is unchanged and is still the off-the-GUI-thread pip
  runner. Nothing in the segmentation-backend rows uses it now; it stays
  because it is the tested way to run a pip from a widget, and
  `CheckpointDownload` beside it is what the zoo's downloads use.
