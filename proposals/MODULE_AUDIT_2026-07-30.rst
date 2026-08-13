Module audit -- 2026-07-30
==========================

.. note::

   AN INTERNAL ARCHITECTURE RECORD, NOT API DOCUMENTATION. This lived in
   ``docs/source/`` and was published in the API toctree, where a reader
   looking for how to call something met an audit addressed to the
   maintainer instead. Moved here 2026-08-12; see instruction 86.

   Its recommendation sections ("High-value additions", "GPU opportunities")
   were cut and preserved in ``instructions/open/86_*`` -- work belongs on
   the instruction list, not in a reference page.

This page records the repository-wide module audit completed on
2026-07-30. It is intentionally tied to the application registry in
``spacr.qt.app.APPS``: every visible module is listed once, with the backend
that actually runs it and the boundary that keeps long work off Qt's main
thread.

Cross-cutting guarantees
------------------------

* Pipeline modules execute through ``spacr.qt.bridge.make_thread``. Worker
  exceptions, non-zero ``SystemExit`` values, stderr, and partial progress are
  surfaced to the module console and diagnostic log.
* Interactive data tools use retained ``QThread`` workers for database scans,
  model discovery, report scans, page loads, and large image decoding.
  ``Make Masks`` keeps small image changes immediate but moves an estimated
  eight MiB or larger decode to a worker.
* Package-wide debug tracing is opt-in. Enabling verbose logging records calls
  and returns for spaCR functions, while pipeline exceptions retain their
  traceback.
* Sphinx AutoAPI scans the whole ``spacr`` package. Every public top-level
  function and class has a docstring, and a structural test prevents new
  undocumented public symbols.
* Every typed setting has explanatory tooltip text. The Qt label owns the
  hover tooltip and teal API dot; the per-app API map is tested against the
  module that the app actually runs.
* Registry tests require GUI, headless CLI, pre-flight validation, defaults,
  and pipeline dispatch to agree. Module tests cover normal paths, invalid
  inputs, empty inputs, mismatched shapes/counts, worker failures, and repeat
  runs where applicable.
* Static analysis has no unused imports or dead local assignments in the
  audited package code. Four remaining warnings belong to independently
  user-edited tutorial engine/script files that this audit deliberately
  preserves.

Module-by-module execution map
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 29 35 18

   * - Module
     - Primary backend
     - Responsiveness and failure boundary
     - Acceleration
   * - Mask
     - ``core``, ``object``, ``io``
     - Pipeline worker; streamed batches and explicit QC ledger
     - Cellpose CUDA with CPU fallback
   * - Timelapse
     - ``core``, ``object``, ``timelapse``
     - Pipeline worker; frame batches and tracking errors surfaced
     - Cellpose/Trackastra where available
   * - Motility Assay
     - ``timelapse``
     - Pipeline worker; bounded CPU workers
     - CPU
   * - Measure
     - ``measure``
     - Pipeline worker; batched fields and database writes
     - CPU; candidate for selective GPU kernels
   * - Annotate
     - ``qt.screens.annotate``, ``qt.annotate_engine``
     - Page-load thread plus asynchronous database save worker
     - I/O bound
   * - Classify (CV)
     - ``deep_spacr``, ``io``
     - Pipeline worker and PyTorch data-loader workers
     - PyTorch CUDA with CPU fallback
   * - Classify (ML)
     - ``ml``
     - Pipeline worker
     - CPU today; XGBoost CUDA is a candidate
   * - Map Barcodes
     - ``sequencing``
     - Pipeline worker plus validated process workers/writer
     - CPU and I/O bound
   * - Regression
     - ``ml``
     - Pipeline worker
     - CPU
   * - Align & Stitch
     - ``align``, ``spacrops``
     - Dedicated job thread; incremental mosaic writes
     - CPU/OpenCV; GPU registration is a candidate
   * - Format Converter
     - ``convert``
     - Preview and conversion jobs run outside the GUI thread
     - I/O bound
   * - Import Project
     - ``foreign``
     - Mapping validation and import job report failures explicitly
     - I/O bound
   * - External Masks
     - ``external_masks``
     - Pipeline worker; read-only plan before writes
     - CPU and I/O bound
   * - Plate Queue
     - ``qt.plate_queue``
     - Retained queue runner; per-item status and stop handling
     - Delegates to each module
   * - Batch Runner
     - ``batch``
     - Sequential worker jobs with atomic persisted state
     - Delegates to each module
   * - Database Browser
     - ``qt.screens.db_browser``
     - Paged background queries and exports
     - I/O bound
   * - Make Masks
     - ``qt.mask_engine``
     - Large image/mask decode on ``QThread``; failures clear stale state
     - CPU interactive editing
   * - Train Cellpose
     - ``submodules``
     - Pipeline worker; CUDA probe safely falls back to CPU
     - Cellpose CUDA
   * - Cellpose Masks
     - ``spacr_cellpose``
     - Pipeline worker
     - Cellpose CUDA with CPU fallback
   * - Model Compare
     - ``model_compare``
     - Model/field loading and inference in retained workers
     - Cellpose CUDA with CPU fallback
   * - Model Zoo
     - ``model_zoo``
     - Discovery, validation, download, and benchmark workers
     - Benchmark backend dependent
   * - Plate Viewer
     - ``plate_qc``
     - Database load and heatmap preparation in worker
     - CPU
   * - Annotator Agreement
     - ``agreement``
     - Database read and agreement computation in worker
     - CPU
   * - Image UMAP
     - ``core``, ``hyperparam``
     - Pipeline/search workers; repeated runs have isolated lifecycle state
     - CPU UMAP today; RAPIDS is a candidate
   * - Activation
     - ``deep_spacr``
     - Pipeline worker
     - PyTorch CUDA with CPU fallback
   * - Training Runs
     - ``train_compare``
     - Scan and curve preparation in worker
     - CPU and I/O bound
   * - Report
     - ``report``
     - Scan/render job in worker; output-open errors surfaced
     - CPU and I/O bound
   * - Plaque Assay
     - ``submodules``
     - Pipeline worker
     - CPU
   * - Recruitment
     - ``submodules``
     - Pipeline worker
     - CPU
   * - Invasion Assay
     - ``submodules``
     - Pipeline worker; schema and threshold validation
     - CPU
   * - Replication Assay
     - ``submodules``
     - Pipeline worker; parasite-to-vacuole assignment validation
     - CPU

Notable corrections from the audit
-----------------------------------

* FASTQ parsing now validates four-line records, paired chunk counts,
  sequence/quality lengths, regex groups, duplicate barcodes, writer exit
  status, and reverse-complements R2 quality with its sequence.
* Format Converter GUI/CLI dispatch now uses the same current conversion
  backend.
* Replication runs the parasite-count assay promised by its UI; the legacy
  area proxy remains explicitly headless.
* Plot significance thresholds no longer lose the ``p <= 0.001`` marker to a
  misspelled variable.
* Make Masks finds TIFF masks saved for PNG/JPEG sources, preserves label IDs
  above 255, rejects mismatched shapes, and threads large decodes.
* Classification image preloading now closes PIL files deterministically,
  uses bounded I/O threads instead of forking a live Qt/PyTorch process, and
  cannot strand a full prefetch queue during shutdown.
* Training-dataset rules now reject unknown columns/operators, empty class or
  annotation definitions, invalid modes, and zero-crop selections with
  actionable exceptions instead of returning a silent ``(None, None)``.
* Cellpose training/test/apply no longer force ``gpu=True`` on machines where
  CUDA is unavailable.
* The legacy GUI keep-alive callback no longer consumes and discards worker
  errors before the console can display them.
* Seven quarantined legacy-GUI defects are now covered as passing regressions:
  nested panel disabling, annotation databases without ``cell_area``, figure
  titles, fresh erase mode, zero-width zooms, fallback card spacing, and
  progress-label placement.
* Batch Runner worker progress, failure, and completion signals now cross an
  explicit queued Qt boundary. Queue settlement therefore always happens on
  the GUI thread, including after a worker exception.
* Settings controls embedded in composite widgets now resolve their outer
  form label correctly. The informative tooltip and teal API link remain on
  that label instead of becoming attached to an inner field.
* Every experimental Home layout now categorizes External Masks, and its
  staged-module count agrees with the canonical registry.
* Reusable action buttons consistently expose positive and negative roles:
  blue/red outline at rest, translucent hover feedback, and a readable solid
  busy/pressed state.

Verification boundary
---------------------

The root test suite was executed in bounded segments, including the complete
remaining suffix after every first-failure correction. Qt tests were likewise
run in fresh 20-file-or-smaller processes because the locally available
Python-3.13 pytest/Qt shim develops native allocator corruption only after
roughly 200 accumulated GUI tests. One hundred of 103 Qt files were exercised;
the three omitted files are an independent, user-edited tutorial group, one of
which currently contains a known unmatched parenthesis. Tutorial CLI and
engine tests still pass.

All 624 other Python sources and tests parse successfully, ``git diff
--check`` is clean, and the API documentation, settings-category, app-registry,
declared-dependency, and debug-logging contracts pass. A rendered Sphinx build
was not attempted because Sphinx is not installed in the spaCR environment;
the structural AutoAPI contract was run instead.
