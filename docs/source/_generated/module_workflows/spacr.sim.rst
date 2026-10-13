Workflow inputs and outputs
---------------------------

Pooled-screen simulation sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the Python sweep API to explore stated screen-design assumptions. It expands combinations, runs simulations in a process pool and writes synthetic summary statistics; it returns None. Set a small explicit max_workers value and a bounded parameter grid before running. The output does not replace measurements or barcode counts for experimental Regression. Use the findings as planning evidence, with their assumptions recorded. Review the synthetic summaries when choosing planning assumptions; transfer those assumptions manually into Power / Design or Experiment Design. Neither tool imports this simulation database, and neither exports the simulation settings grid.

**Use from Python:** :func:`spacr.sim.run_multiple_simulations`. This API-only workflow has no Home tile or menu entry.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Pooled-screen simulation assumptions** — Python settings dictionary: iterable sweep values for screen size, occupancy, classifier accuracy and sequencing assumptions, plus replicates, src, name, variable, plot and max_workers. generate_parameters expands their Cartesian product; begin with a small sweep and an explicit worker bound.

**Outputs**

* **Synthetic screen simulation summaries** — src/<YYMMDD>/<name>/simulations.db; the sweep appends summary rows to simulations and optionally writes plots. These are synthetic performance estimates, not measured experimental hits.
  Relevant tables, depending on the route: ``simulations``.

:doc:`API reference </api/spacr/sim/index>`.

