# FlowView

FlowView records the stages inside one spaCR run and renders them as a live
node graph or as a deterministic SVG, HTML, or JSON file. It complements the
Pipeline Graph: Pipeline Graph shows which project artifacts produced other
artifacts across runs, while FlowView shows what one run is doing stage by
stage.

FlowView is disabled by default. Importing `spacr.flowview` is headless-safe
and does not import PySide6. The live panel needs the GUI extra:

```console
python -m pip install "spacr[flowview]"
```

## Classify graph

The first supported blueprint follows the active Classify family through
eight stages:

1. Source folder
2. PNG list or measurement tables
3. Dataset build
4. Train/validation split
5. Active CV or ML model
6. Training loop
7. Evaluation
8. Scores written to the database

Only the family used by the run is drawn. The graph does not show a dormant
CV or ML branch.

Create the blueprint from the same settings passed to Classify, attach a
collector, and then export any immutable snapshot:

```python
from spacr import flowview

settings = {
    "src": "/data/screen",
    "classifier_family": "ml",
    "dataset_mode": "metadata",
    "model_type_ml": "xgboost",
    "test_split": 0.2,
}

graph = flowview.classify_graph(settings, run_id="classify-2026-08-29")
collector = flowview.Collector(graph)
flowview.enable(collector)

# After stage events have been emitted:
collector.drain()
snapshot = collector.snapshot()
flowview.export(snapshot, "classify-run.svg")
flowview.export(snapshot, "classify-run.html", fmt="html")
flowview.export(snapshot, "classify-run.json", fmt="json")
```

SVG keeps labels and metrics as editable text. HTML is self-contained: it
embeds the same SVG, thumbnails, and a complete inspector table. JSON is the
canonical run record. Re-exporting an unchanged graph produces identical
bytes.

## Instrument a stage

Use `flowview.stage` as either a context manager or decorator. Reuse a
blueprint node ID when recording one of its stages:

```python
from spacr import flowview

with flowview.stage(
    "Dataset build",
    node_id="dataset",
    consumes=["Measurement tables"],
    produces=["Training dataset"],
    params={"dataset_mode": "metadata"},
) as stage:
    for index, batch in enumerate(batches, start=1):
        process(batch)
        stage.progress(index, len(batches))
        stage.metric("rows", rows_processed)
        stage.thumbnail(thumbnail_path)
```

```python
@flowview.stage(
    "Evaluation",
    node_id="evaluation",
    consumes=["Trained model"],
    produces=["Scores"],
)
def evaluate(model, validation_data):
    return model.evaluate(validation_data)
```

When tracing is disabled, a decorated function is returned unchanged and a
context manager uses a shared no-op stage. The disabled entry cost is one
boolean check; calls to `progress`, `metric`, and `thumbnail` then do no work.
Enable tracing explicitly with `flowview.enable(collector)`, or set
`SPACR_FLOWVIEW=1` before importing FlowView. Call `flowview.disable()` when
the run no longer needs tracing.

Instrumentation cannot turn a successful analysis into a failed one. Event
emission is failure-isolated, and an exception raised by the instrumented
code still propagates unchanged. FlowView records the failed node and marks
known downstream nodes as skipped.

## Thumbnails

Never put image arrays on an event queue. Prepare a bounded on-disk cache and
send only the resulting path:

```python
from pathlib import Path

from spacr.flowview.thumbs import ThumbnailCache

cache = ThumbnailCache(
    Path(settings["src"]) / ".spacr" / "flowview" / graph.run_id,
    max_bytes=50 * 1024 * 1024,
)
path = cache.store(
    "dataset-preview",
    image_array,
    outline_mask=segmentation_labels,
)

with flowview.stage("Dataset build", node_id="dataset") as stage:
    stage.thumbnail(path)
```

Images are contrast-stretched at the second and ninety-eighth percentiles,
downsampled to at most 128 pixels on the long axis, and stored as PNG. A
segmentation mask is drawn as a one-pixel outline, never as a filled overlay.
The cache evicts oldest entries first. Call `cache.discard()` when the run
record is discarded; retain the directory when an export must remain able to
embed its thumbnails.

## Multiprocessing producers

Worker processes should emit only declared FlowView event dataclasses. The
non-blocking helper validates that an event is picklable and no larger than
64 KiB before putting it on a multiprocessing-compatible queue:

```python
from multiprocessing import Queue

from spacr.flowview import MultiprocessingFeeder, put_event_nowait
from spacr.flowview.events import StageProgress

event_queue = Queue(maxsize=2_000)
feeder = MultiprocessingFeeder(event_queue, collector).start()

# In a worker process:
put_event_nowait(event_queue, StageProgress("training", 4, 20))

# During orderly shutdown in the parent process:
feeder.stop()
```

The feeder is a daemon thread and never owns, closes, or drains the source
queue during shutdown. Invalid or oversized values are discarded. The
collector itself has a bounded 2,000-event queue with drop-oldest behavior;
if it fills, `collector.sampled` becomes true and the live panel explains
that it is sampling updates.

## Live panel

`FlowViewPanel` is a boxed PySide6 widget designed for placement below the
Classify settings. It polls the collector at 20 Hz, but skips snapshots,
layout, and painting while the collector revision is unchanged. Users can
pan, zoom, select a node for its parameters, metrics, timings, and traceback,
and export the current graph.

```python
from spacr.flowview.panel import FlowViewPanel

panel = FlowViewPanel(collector, parent=classify_screen)
settings_layout.addWidget(panel)
```

The panel owns its refresh timer and stops it when closed. Keep the collector
independent of the widget so headless and command-line runs retain the same
event and export behavior.

## Run-record lifecycle

- Treat `Collector.snapshot()` as immutable renderer input.
- Store the JSON export with the run's other reproducibility records.
- Retain thumbnail files for SVG or HTML exports that have not yet been made.
- Discard the per-run thumbnail cache when the run record itself is deleted.
- Do not use FlowView as a control surface; it reports a run but does not
  edit or re-execute pipeline stages.
