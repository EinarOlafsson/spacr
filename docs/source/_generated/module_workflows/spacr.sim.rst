Workflow inputs and outputs
---------------------------

Pooled-screen simulation sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the Python sweep API to explore stated screen-design assumptions. It expands combinations, runs simulations in a process pool and writes synthetic summary statistics; it returns None. Set a small explicit max_workers value and a bounded parameter grid before running. The output does not replace measurements or barcode counts for experimental Regression. Use the findings as planning evidence, with their assumptions recorded.

**Use from Python:** :func:`spacr.sim.run_multiple_simulations`. This API-only workflow has no Home tile or menu entry.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Pooled-screen simulation assumptions** — Python settings dictionary: iterable sweep values for screen size, occupancy, classifier accuracy and sequencing assumptions, plus replicates, src, name, variable, plot and max_workers. generate_parameters expands their Cartesian product; begin with a small sweep and an explicit worker bound.

**Outputs**

* **Synthetic screen simulation summaries** — src/<YYMMDD>/<name>/simulations.db; the sweep appends summary rows to simulations and optionally writes plots. These are synthetic performance estimates, not measured experimental hits.
  Relevant tables, depending on the route: ``simulations``.

**Before this module**

* :ref:`Power / Design <workflow-module-power>`: Translate planning assumptions into the simulation settings dictionary manually; Power / Design does not export a ready-to-run simulation grid.

**After this module**

* :ref:`Experiment Design <workflow-module-experiment_design>`: Use simulated performance to reconsider sampling and plate constraints manually; the simulation database is not an importable plate layout.

:doc:`API reference </api/spacr/sim/index>`.

