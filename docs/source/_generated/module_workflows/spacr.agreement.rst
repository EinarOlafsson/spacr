Workflow inputs and outputs
---------------------------

Annotator Agreement
~~~~~~~~~~~~~~~~~~~

Compare independent annotation columns and inspect discordant objects; agreement is separate from classification accuracy.

**Open:** Annotate → Annotator Agreement.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Annotate <workflow-module-annotate>`: Choose independent annotation columns for the same objects.

:doc:`API reference </api/spacr/agreement/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=23_agreement>`__.

