# Notes from `spacr/qt/resource_cleanup.py`

Prose lifted out of `spacr/qt/resource_cleanup.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [_cuda_stat](#_cuda_stat) (1 entry)
- [_loaded_cache_owners](#_loaded_cache_owners) (1 entry)
- [sweep_memory_budget](#sweep_memory_budget) (3 entries)
- [_clear_lru_caches](#_clear_lru_caches) (1 entry)
- [_clear_pixmap_cache](#_clear_pixmap_cache) (1 entry)
- [clear_vram](#clear_vram) (1 entry)
- [_is_a_folder](#_is_a_folder) (1 entry)
- [_readings_within_the_budget](#_readings_within_the_budget) (1 entry)
- [_on_registry_changed](#_on_registry_changed) (1 entry)
- [_budget_tick](#_budget_tick) (1 entry)
- [_request_budget_sweep](#_request_budget_sweep) (1 entry)

## Module level

### lines 130-132

```python
BUDGET_SWEEP_INTERVAL_MS = 5000
```

A sweep does bounded work on the GUI thread.  If more is required, the next five-second tick continues it; no single pass can pickle/spill hundreds of figures and turn a memory safeguard into the event-loop freeze it prevents.

### lines 583-587

```python
_CUDA_CACHE_BYTES: Optional[int] = None
```

The allocator exposes no "last kernel finished" timestamp.  Observing its reclaimable byte count on the existing five-second sweep is the honest substitute: a change, or any registered run in flight, is activity; an unchanged cache after the run is idle.  These two scalars retain no tensor and therefore cannot become another cache themselves.

### lines 1388-1393

```python
_CONFIRMATIONS: Dict[str, Tuple[str, str]] = {
```

What the confirmation says

A confirmation that asks "are you sure?" is not a confirmation: a user cannot consent to an unnamed action. Each of these names what will happen, in the order it will happen, and says what the action cannot do.

## _cuda_stat

### lines 366-368

```python
return int(getter())
```

Small test doubles and old compatible torch builds may expose only the no-argument form.  It still measures the current device rather than turning a cleanup into an import or context initialisation.

## _loaded_cache_owners

### line 459

```python
return tuple({id(owner): owner for owner in owners}.values())
```

A buggy screen must not make the same owner count twice.

## sweep_memory_budget

### lines 724-726

```python
remaining = sorted((row for row in candidates
```

Headroom is a hard floor, not another per-cache size.  Once ordinary idle/ceiling evictions have run, release the coldest remaining entries until the measured floor is restored or this bounded pass is exhausted.

### lines 747-748

```python
result = clear_vram(release_models=False)
```

CUDA allocator blocks are reclaimable state too.  This never imports torch or initialises a CUDA context; clear_vram has both guards.

### lines 753-755

```python
result = clear_vram(release_models=False)
```

A stable allocator cache obeys the same idle timeout and byte ceiling as the RAM caches.  Live allocations are excluded by ``_cuda_cached`` and a registered run pins the allocator wholesale.

## _clear_lru_caches

### lines 789-790

```python
continue
```

Not imported means not populated. Importing it to clear it would allocate rather than free.

## _clear_pixmap_cache

### lines 892-895

```python
total_used = getattr(QPixmapCache, "totalUsed", None)
```

Qt 6 removed ``totalUsed`` from the Python API. Absence of an accounting reading must not turn into absence of the cleanup: clear the cache either way, and report a byte count only when Qt supplied one. Inventing a count would violate Reclaim's measured contract.

## clear_vram

### lines 1008-1012

```python
from ..accelerator import empty_cache as release_device_memory
```

EVERY BACKEND CACHES, not only CUDA. Metal holds freed blocks in exactly the same way and answers `torch.mps.empty_cache()`; on a 4 GB card that is the difference between the next screen opening and an allocation failure. Routed through the resolver so the right call is made without a vendor branch here. See 319.

## _is_a_folder

### lines 1201-1203

```python
return False
```

No probe available and no licence to stat: say no rather than freeze. The folder returns to the report as soon as the disk check runs where it belongs.

## _readings_within_the_budget

### lines 1332-1333

```python
return dict(answers)
```

A copy, so a late answer cannot appear halfway through the report and give one folder a line the one above it was denied.

## _on_registry_changed

### lines 1635-1637

```python
_request_budget_sweep()
```

A run just finished (or the registry only shed a handle).  Its caches are no longer in use; let the event loop settle, then apply the same global policy the periodic sweep uses.

## _budget_tick

### lines 1663-1673

```python
LOG.debug(
```

DEBUG, NOT INFO. This is housekeeping the user did not ask for and cannot act on, and it fired on EVERY module open -- reported against Mask, Measure and Map Barcodes alike. It also read as nonsense when it did: "memory budget: 0.0 -> 0.0 MiB ... and 2.6 GB VRAM released" is two different accountings in one sentence, because before_mb/after_mb are HOST RSS and vram_freed is device memory. Host RSS legitimately does not move when VRAM is released, so the line was correct and unreadable at once.

Kept rather than deleted: it is genuinely useful when chasing a leak, which is what the debug level is for.

## _request_budget_sweep

### lines 1697-1698

```python
LOG.debug("could not queue the live-cache sweep", exc_info=True)
```

No event loop means the periodic integration is inapplicable.  The explicit ``sweep_memory_budget`` API remains usable by headless code.
