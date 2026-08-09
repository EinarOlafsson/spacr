Threading and cancellation audit
================================

spaCR runs analysis entry points through :class:`spacr.qt.bridge.PipelineWorker`
on a dedicated ``QThread``. Large imports, image decoding, model fitting and
plate processing therefore do not run on Qt's GUI thread. Stop is cooperative:
spaCR never uses ``QThread.terminate()`` on an analysis that may be writing an
array, TIFF, database transaction, or checkpoint.

Safe cancellation boundaries
----------------------------

The worker installs a thread-local :class:`spacr.cancellation.CancellationToken`.
Long workflows call :func:`spacr.cancellation.checkpoint` only after a durable
unit, or before the next unit begins:

* Mask checks between source plates, object types, and Cellpose batches.
* Measure keeps at most one pool-width field batch outstanding and checks after
  every batch has returned and saved its results.
* Format Converter checks before source groups; its atomic
  :class:`spacr.checkpoint.CheckpointStore` record checks after each complete
  field.
* UMAP checks before each grid/adaptive trial. Completed trial artifacts and
  adaptive state are persisted before cancellation is raised.
* Batch Runner checks between jobs. While a child ``spacr-run`` process is
  active it polls the token, terminates that child, persists the current job as
  resumable, and then stops the queue.

A Stop click therefore may wait for the current unit. The console reports
``Stopped safely`` when the boundary is reached, and the reproducibility
manifest records ``status: cancelled`` rather than a failure traceback.

Shutdown behavior
-----------------

The process-wide :class:`spacr.qt.bridge.RunRegistry` owns strong references to
every active worker and thread. On application shutdown it requests
cancellation for all jobs and waits against one bounded deadline. If any job
has not reached a safe boundary, the close event is refused and the live
references are retained. The user can close again after the current unit
finishes. The same rule applies when an individual module screen is closed.

``PipelineWorker.finished`` invokes ``QThread.quit`` directly because that Qt
method is thread-safe. This avoids a shutdown deadlock in which the GUI thread
waits for a worker while the worker's queued quit request waits for the GUI
event loop.

API usage
---------

Headless and plugin pipelines use the same small API::

   from spacr.cancellation import checkpoint

   for field in fields:
       checkpoint()          # previous field is durable; next has not started
       process_field(field)
       write_field_atomically(field)

Calling ``checkpoint()`` outside a managed GUI worker is a no-op. Code that
catches broad ``Exception`` values must re-raise
:class:`spacr.cancellation.PipelineCancelled`; treating it as a failed field
would defeat the worker's distinct cancellation status.

Stress coverage
---------------

``tests/qt/test_threading_cancellation_audit.py`` exercises fifty rapid
Start/Stop cycles, registry-wide shutdown, repeated cancellation, screen close
while active, and refusal to destroy a stubborn worker.
``tests/test_cancellation.py`` verifies thread isolation, idempotence,
durability-before-cancel, resumable queues, and termination of an active batch
subprocess.

.. automodule:: spacr.cancellation
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:
