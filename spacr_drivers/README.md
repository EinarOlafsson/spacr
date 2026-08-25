# Real-data drivers

One script per core module, each of which drives that module end to end on
real data rather than on a fixture. They exist so a claim of the form "measure
reproduces the reference database" can be re-checked by running one command
instead of being taken on trust.

    python spacr_drivers/drive_measure_on_plate1.py [DATASET_ROOT] [SETTINGS]

Every driver follows the same contract:

* **The dataset root is the first argument**, and defaults to where the data
  sits on the machine the runs were recorded on. Give a path to run against a
  copy anywhere else.
* **A missing dataset is a refusal, not a half-run.** Preconditions are checked
  before any heavy import, so pointing a driver at an unmounted disk answers in
  under a second, names every input it could not find, and exits 2.
* **The dataset is never written to.** Inputs are copied into a scratch tree
  (`$SPACR_DRIVER_SCRATCH`, else the system temporary directory) and the run
  works there. `stage()` refuses a destination inside the dataset root.
* **Settings come from the file the recorded run used**, loaded with spaCR's
  own loader, and are put through spaCR's pre-flight check before the run
  starts.
* **The shared GPU also drives the display.** Anything that touches CUDA caps
  itself at 80% of the card.

`_support.py` holds that contract; the drivers hold what each module needs.
`tests/test_real_data_drivers.py` tests the contract, including that each
driver refuses cleanly when the data is absent -- which is the half that runs
on a machine with none of these datasets on it.
