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

Stop first asks how to stop. **Finish current work** requests cancellation
and waits for the current unit; if the run has not reached a boundary after
five minutes, the question is asked again. The console reports ``Stopped
safely`` when the boundary is reached, and the reproducibility manifest
records ``status: cancelled`` rather than a failure traceback.

**Force stop** returns control to the window immediately. A worker that does not respond
-- typically one inside a long Cellpose or PyTorch call -- is parked rather
than terminated: its references are kept and it finishes in the background,
so anything it is writing may be left half-written. **Force restart** saves the
module and its settings, then starts spaCR again and reopens that module.

Shutdown behavior
-----------------

Application shutdown requests cancellation of registered runs, background job
runners and standalone Qt workers within one shared wait budget. If any worker
remains running, the window stays open and its live references are retained;
workers that outlast the wait are parked rather than terminated. This also
applies to read-only work such as a Run History refresh or release-news fetch.
Wait for the worker to finish, then close the window again. An individual
module screen may also refuse closure while its analysis is still running.

``PipelineWorker.finished`` invokes ``QThread.quit`` directly because that Qt
method is thread-safe. This avoids a shutdown deadlock in which the GUI thread
waits for a worker while the worker's queued quit request waits for the GUI
event loop.

API usage
---------

Headless and plugin pipelines use the same API::

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
while active, and refusal to destroy an unresponsive worker.
``tests/test_cancellation.py`` verifies thread isolation, idempotence,
durability-before-cancel, resumable queues, and termination of an active batch
subprocess.

.. automodule:: spacr.cancellation
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:
