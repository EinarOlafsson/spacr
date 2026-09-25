Unified run history
===================

Open **Help → Run History** (or type *Run History* in the command palette,
Ctrl+K) to search every pipeline recorded by spaCR. Refreshing a large journal
happens on a background worker and does not create another run record.

The dashboard combines:

* module, status, start/end time, wall time and process CPU time;
* exact resolved settings;
* SHA-256 records for inputs, outputs and models;
* structured warnings, provenance warnings and failure tracebacks;
* package, spaCR, Git and platform versions;
* declared seeds and runtime random-state identifiers.

Search terms are combined: ``adamw plate_03 warning`` shows only records
containing all three terms anywhere in settings, paths, warnings, failures or
environment data. Module and status filters can be applied at the same time.
Interrupted and corrupt folders stay visible rather than disappearing.

Select a row to inspect its details. **Load settings in module** opens the
original module and propagates the exact recorded settings into its controls;
it does not start a run. **Open run folder** and **Copy path** expose the
underlying ``~/.spacr/runs/...`` folder.

Right-click selected rows to open their folders or delete them; **Clear all**
deletes every run the table lists. Deleting asks first and removes only the
journal folders; outputs written into your projects are not touched.

Headless search
---------------

:func:`spacr.run_journal.search_runs` provides the same resilient records
without Qt:

.. code-block:: python

   from spacr.run_journal import search_runs

   failed = search_runs("database locked", status="failed")
   for record in failed:
       print(record["run_id"], record["failure"])
