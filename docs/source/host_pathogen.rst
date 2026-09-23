Host–Pathogen Analysis
========================

From **Home → Assays → Toxoplasma**, open **Host–Pathogen Analysis**.
This alpha module combines vacuole-level marker recruitment with host and well
summaries. Existing :doc:`Recruitment <recruitment>` remains available.

Prepare the counting units
--------------------------

Run Mask and Measure first. Keep uninfected host cells in Measure
(``uninfected=True``) when the intended infection denominator is all measured
hosts. Previously discarded host cells cannot be reconstructed here.

Supply a cell table, a table with one object per whole vacuole (default:
``pathogen``), and a host reference compartment table (default:
``cytoplasm``). Individual-parasite masks cannot substitute for whole-vacuole
masks. Select marker channels and per-channel ratio thresholds calibrated
with appropriate controls; the module does not learn these thresholds.

Replication counts require either a parasite table with explicit parent-vacuole
identities or an existing count column. Choose one count source. Without one,
replication remains unmeasured rather than being inferred from vacuole area.
Host identity alone is insufficient when a host contains multiple vacuoles.

When Measure links individual organelle-role objects to whole-vacuole masks,
one parent must cover strictly more than half of a child's pixels. Outside
objects and ambiguous overlaps retain a missing parent. The stored parent is
``pathogen_id``; the stored overlap feature is role-prefixed, for example
``organelle_pathogen_overlap_fraction``. Use such an object table as parasites
only when those masks actually represent individual parasites.

Interpret the reports
---------------------

The output directory ``results/host_pathogen`` contains ``vacuoles.csv``,
``cells.csv``, ``wells.csv``, ``marker_states.csv``,
``replication_distribution.csv``, ``orphan_parasites.csv`` and the exact
``settings.json``. With multiple input projects, the combined report is saved
under the first project. Database paths separate sources even when plate or
field identifiers repeat.

Unlinked vacuoles and missing, invalid or zero host-reference intensities yield
unknown marker states. Joint marker fractions include unknown vacuoles in
their denominator; replication fractions use only vacuoles with counts.
Host infection fractions describe retained measured host cells. These
denominators answer different questions and should be reported explicitly.

For scripted use, see :func:`spacr.host_pathogen.analyze_host_pathogen` and
:func:`spacr.host_pathogen.summarize_tables`. The headless module name is
``host_pathogen`` in ``spacr-run``.

Choosing a Replication Assay method
--------------------------------------

The separate **Replication Assay → Method** selector defaults to
``direct_count``, which counts individually segmented parasites per assigned
vacuole. ``size_proxy`` uses the existing endodyogeny analysis and its
**Size Proxy (Legacy)** controls. That readout aggregates pathogen area per
host cell; multiple vacuoles in one host are combined. It is an area-derived
proxy, not a parasite count or measured volume.

Both available routes return ``replication_method`` metadata. The whole-vacuole
deep-learning option is marked coming soon and cannot run before its model is
available. See :func:`spacr.submodules.analyze_replication` and
:func:`spacr.submodules.analyze_endodyogeny` for the exact settings and outputs.
