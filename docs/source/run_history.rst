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

Storage: pruning and caches
---------------------------

**Preferences → Storage** keeps the spaCR home folder from growing without
bound. For daily logs, run logs (``~/.spacr/logs/runs``) and run folders
(``~/.spacr/runs``) it holds two caps: **Keep**, an age in days within which
nothing is ever deleted, and **Cap**, a size above which the oldest older
entries are deleted (**no cap** deletes every older entry). The run in
progress and open log files are never deleted. **Prune now…** lists what the
caps would delete and asks before deleting; nothing is pruned automatically.

The cache table lists the spaCR model, Cellpose, Hugging Face, Torch, backend
environment and news caches. **Measure** reads their sizes, **Clear…** empties
the selected cache after asking, and **Move…** moves it into
``<folder>/spacr-<cache>`` on another drive. The new place is recorded in
``~/.spacr/cache_locations.json`` and used at every start; a cache whose
variable (for example ``HF_HOME``) is set outside spaCR is not moved.
Clearing and moving wait until no run is in progress.

Headless search
---------------

:func:`spacr.run_journal.search_runs` provides the same resilient records
without Qt:

.. code-block:: python

   from spacr.run_journal import search_runs

   failed = search_runs("database locked", status="failed")
   for record in failed:
       print(record["run_id"], record["failure"])
