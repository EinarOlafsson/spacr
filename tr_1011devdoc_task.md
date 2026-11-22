# Device-rule documentation handover

Branch `ws-offload-codex-work-1011devdoc`, base `2106ce3a4a1bd4ab9576dc159b3716e335b836fc`.

The requested explanation is now in the private helper docstring:

A model that names no device runs on the CPU; the current CUDA card is never assumed.

Location: `spacr/_mask_batches.py:468`, `spacr._mask_batches._calibrate`. SHA-256 of the complete revised docstring: `860a02dea5a26622f73357c9ff0ff001fa263c1f037c27df5deda21b870e5495`.

The executable module AST is unchanged. The private-source arrival contract binds the exact previous function and revised source, and the existing 1011 test file checks both the prose and executable equivalence. The companion JSON includes exact before/after text, hashes and source locations, plus the new test docstring.

Fresh exact-base/current extraction preserves all 14961 published API bodies and all runtime source tables. This private helper is outside the published API inventory; there are no new canonical API or runtime translation identities.

## Revised private docstring

Measure one tile's and the fixed working set's GPU memory on the run's own data (item 1011).

Two probe evaluations of ``sample`` at tile batches ``t1 = 2`` and
``t2 = min(8, cap)`` are measured with the governor's own rule (reserved
growth, never below 1.5 times the allocated growth, through
``_BatchController._learn`` of ``site``); their results are discarded. A
linear fit gives ``per_tile`` and ``fixed`` (the weights the process holds,
image and flow buffers, the fixed working set), each floored at a tenth of
item 987b's constants. The budget ``B`` (declared, or free less the other
spaCR processes' announcements; less ``gpu_memory_margin_gb`` and the CUDA
context; divided by ``mask_workers_per_gpu``) then gives the tile batch
first, ``clamp(floor((B - fixed - per_field) / per_tile), 1, cap)``, and the
most fields of one resident batch, ``floor((B - fixed - tiles * per_tile) /
per_field)``. Without a CUDA device nothing is probed: ``tiles = min(cap, 8)``
and ``max_fields`` is None (the caller applies its RAM limit, capped at 50).

A model that names no device runs on the CPU; the current CUDA card is
never assumed.

:param model: the loaded segmentation model.
:param sample: the first field of the seeded order, as the real call gets it.
:param settings: the run settings (``mask_eval_batch_cap``,
    ``gpu_memory_margin_gb``, ``gpu_vram_budget_gb``, ``mask_workers_per_gpu``).
:param site: the governor call site the probes are measured on.
:param evaluate: callable ``tiles -> None`` running one evaluation of
    ``sample`` at that tile batch; ``model.eval(sample, batch_size=tiles)``
    when None.
:param per_field: bytes one field adds to a resident batch;
    ``sample.nbytes`` when None.
:param governor: the governor module (:mod:`spacr._gpu_governor`), for tests.
:returns: the calibration record: ``tiles``, ``max_fields``, ``per_tile``,
    ``fixed``, ``per_field``, ``budget`` (bytes), ``budget_source``
    (``declared``, ``free`` or ``cpu``), ``cap``, ``probes``, ``device``,
    ``total`` and the Cellpose and torch versions.

## Validation

Whole 1011 file: 86 passed with CUDA hidden; 86 passed with visibility unset and 2 GB VRAM reserved. All six required source-rule files: 157 passed.

This is the later documentation item requested after the superseded audits. It is outside the pilot scope.
