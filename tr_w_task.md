# TASK W translation and arrival-contract handover

Source branch: ws-offload-codex-work-masktrace. Implementation: a681b9a73b694d5b408df7e528221b5783650d5e. Evidence checkpoint: d97cbe6a240327b5ee1816150d6ba18d2914f858. Base: ef603c010373e769be01b839d88fcf9f7ce9a045. Evidence and handover commits do not change these source strings.

W lands after X on the train-4 base. The speed measurements are descriptive (n = 2 per mode and worker count), not a performance claim. Re-extract source/context/hash pairs if the landing changes an admitted block.

Languages: sv, de, es, zh_CN, pt, hi, ko, is, fr. Only admitted public API docstrings and admitted tr() runtime strings need source-bound arrival contracts and records. Use the established explicit AI-review label with no human or native-speaker signoff. Preserve code identifiers, placeholders and Sphinx markup.

All 35 inventoried docstrings were matched to the current AST by source line and exact text, then checked against the real canonical API extractor. Admission counts: {"canonical API": 1, "private-nested": 9, "test": 25}.

Every private, nested and test entry below is explicitly marked "no records". Do not create orphan records or widen source admission for this handover. New or changed runtime strings: none. There are no runtime-not-admitted entries.

Canonical blocks below give the exact source, context and SHA-256 accepted by the extractor. Validate each record against that source. Supersede changed admitted evidence through the established overlay process while retaining archived files. Skip docs assembly.

Extraction used only read-only public_docstrings(), translatable_blocks() and extract_static_ui_sources(). The import-only ensure_review_inputs hydration hook was suppressed because the pinned cache is unavailable on this checkout; no extractor/admission/record validator was bypassed and no documentation was assembled.

## Exact inventory

### 1. spacr/_mask_tracing.py:1 — <module>

Admission: canonical API. Records: admitted source-bound API records.

```text
Overlap public Cellpose tracing calls within one resident image batch.
```

Canonical API symbol: `spacr._mask_tracing`.

```json
[
  {
    "index": 0,
    "source": "Overlap public Cellpose tracing calls within one resident image batch.",
    "sha256": "e28a60cad17989bd11f271cd2baab4439a74911d35fbd28346a4ecfd82b18266",
    "context": "Overlap public Cellpose tracing calls within one resident image batch."
  }
]
```

### 2. spacr/_mask_tracing.py:12 — _public_trace_api

Admission: private-nested. Records: no records.

```text
Return the public API whose split-call contract has been verified.

    Cellpose 4.0.9 returns resampled flows and derives the tracing iteration
    count from the diameter. Other releases retain their combined eval call
    until their public return and scaling contracts have been checked.

    :returns: model type, original eval and public tracing function, or None.
```

### 3. spacr/_mask_tracing.py:31 — _can_overlap

Admission: private-nested. Records: no records.

```text
Whether this call has the verified independent two-dimensional contract.

    :param model: the in-process segmentation model.
    :param images: the complete resident batch supplied to eval.
    :param keywords: the unchanged public eval options.
    :returns: True only for supported CUDA models and flat image lists.
```

### 4. spacr/_mask_tracing.py:68 — _trace_mask

Admission: private-nested. Records: no records.

```text
Construct one mask with the verified public Cellpose tracing call.

    The network has already returned NumPy flows, so the tracing stream has
    no dependency on model tensors or the network stream. Its result is a
    host mask before the future completes.

    :param flow: public eval flow tuple for one image.
    :param shape: original image height and width.
    :param keywords: unchanged public eval options.
    :param device: the same device used by the combined eval path.
    :param stream: independent CUDA stream used only for tracing.
    :returns: the mask returned by public Cellpose dynamics.
```

### 5. spacr/_mask_tracing.py:100 — _overlap_images

Admission: private-nested. Records: no records.

```text
Keep one tracing future in flight while evaluating the next image.

    Work remains inside the caller-owned batch: no second batch is loaded.
    Results preserve input order, and the executor drains before any error
    reaches the governor, so a retry cannot overlap abandoned work.

    :param images: images already resident in the current batch.
    :param evaluate: public network-and-flow call for one image.
    :param trace: public tracing call taking flows and image shape.
    :param checkpoint: cancellation check before each network call.
    :returns: masks, per-image flows and styles in public list-eval order.
```

### 6. spacr/_mask_tracing.py:129 — _overlap_eval

Admission: private-nested. Records: no records.

```text
Evaluate a supported batch with independent public tracing calls.

    The governor owns the model and retries the complete call. CPU fallback,
    unsupported versions, other backends and volumetric calls keep the
    combined public eval path. Cellpose functions and module attributes are
    never replaced.

    :param model: the original Cellpose model governed by the caller.
    :param images: one complete resident batch.
    :param keywords: unchanged public eval options, including tile batch size.
    :returns: the ordinary public masks, flows and styles tuple.
```

### 7. spacr/_mask_tracing.py:150 — _overlap_eval.evaluate

Admission: private-nested. Records: no records.

```text
Return flows from the original public eval on one image.
```

### 8. spacr/_mask_tracing.py:154 — _overlap_eval.trace

Admission: private-nested. Records: no records.

```text
Trace one image on its independent stream using public dynamics.
```

### 9. spacr/_gpu_governor.py:1154 — _governed_eval

Admission: private-nested. Records: no records.

```text
Call ``model.eval(x, batch_size=..., **kwargs)`` under the governor.

    The batch size is the number of tiles Cellpose sends through the network
    at once, so halving it retries the same images with less memory. The
    CPU fallback moves the network to the CPU for that call only.

    :param model: a Cellpose model or a spaCR backend with the same ``eval``.
    :param x: the image or list of images.
    :param site: the call site's name.
    :param batch_size: the batch size the caller asked for.
    :param settings: run settings holding the governor keys, if any.
    :param evaluate: optional spaCR orchestration callable taking the original
        model, images and eval options; retries and CPU fallback retain the
        same model and call boundary.
    :param kwargs: passed to ``eval`` unchanged.
    :returns: what ``eval`` returned.
```

### 10. spacr/object.py:1277 — _mask_eval

Admission: private-nested. Records: no records.

```text
Call ``model.eval(x, **kwargs)`` under the GPU memory governor.

    A model that runs in this process (Cellpose-SAM and the in-process
    backends) is governed: on a shared card a batch that would not fit is
    shrunk, retried at half size after an out-of-memory error or moved to the
    CPU, instead of stopping the run. With memory to spare, or without a
    CUDA device, the call is made exactly as asked, so the masks are the same.
    A model with no device of its own, such as a backend served from its
    own environment, is called directly.

    :param model: a model with Cellpose's ``eval``.
    :param x: the image or images.
    :param site: the governor's name for this call site.
    :param settings: run settings holding the governor keys, if any.
    :param kwargs: passed to ``eval``; ``batch_size`` is the size asked for.
        Supported flat Cellpose 4.0.9 CUDA batches overlap public tracing on
        an independent stream within the same resident batch.
    :returns: what ``eval`` returned.
```

### 11. tests/test_mask_tracing_overlap.py:1 — <module>

Admission: test. Records: no records.

```text
Public tracing overlaps without replacing Cellpose functions or escaping the governor.
```

### 12. tests/test_mask_tracing_overlap.py:16 — test_the_next_network_call_overlaps_one_trace_and_preserves_order

Admission: test. Records: no records.

```text
The second network call must enter before the first tracing call finishes.
```

### 13. tests/test_mask_tracing_overlap.py:55 — test_failures_drain_the_pending_trace_before_a_governor_retry

Admission: test. Records: no records.

```text
No tracing thread remains active when the complete eval call fails.
```

### 14. tests/test_mask_tracing_overlap.py:95 — test_public_cellpose_functions_remain_the_original_objects

Admission: test. Records: no records.

```text
Run real public empty-flow tracing without replacing any Cellpose callable.
```

### 15. tests/test_mask_tracing_overlap.py:128 — _Model

Admission: test. Records: no records.

```text
A local model stand-in, without any changes to Cellpose modules.
```

### 16. tests/test_mask_tracing_overlap.py:150 — _Network

Admission: test. Records: no records.

```text
Record device transfers without initializing CUDA.
```

### 17. tests/test_mask_tracing_overlap.py:169 — test_unverified_calls_keep_the_combined_public_api

Admission: test. Records: no records.

```text
Only the measured flat-image contract can select overlap.
```

### 18. tests/test_mask_tracing_overlap.py:179 — test_cpu_and_single_image_calls_keep_their_original_return_value

Admission: test. Records: no records.

```text
The ordinary eval call handles unsupported inputs and CPU fallback.
```

### 19. tests/test_mask_tracing_overlap.py:188 — test_the_governor_retries_the_orchestration_with_the_same_images

Admission: test. Records: no records.

```text
An OOM still shrinks tile size while retaining the complete image batch.
```

### 20. tests/test_mask_tracing_overlap.py:206 — test_the_orchestration_observes_cpu_fallback_and_device_restoration

Admission: test. Records: no records.

```text
CPU fallback uses the same callback and restores the original device.
```

### 21. tests/test_mask_tracing_overlap.py:23 — test_the_next_network_call_overlaps_one_trace_and_preserves_order.evaluate

Admission: test. Records: no records.

```text
Release the first trace only when the second network call enters.
```

### 22. tests/test_mask_tracing_overlap.py:32 — test_the_next_network_call_overlaps_one_trace_and_preserves_order.trace

Admission: test. Records: no records.

```text
Record serialized tracing calls while the network proceeds.
```

### 23. tests/test_mask_tracing_overlap.py:43 — test_the_next_network_call_overlaps_one_trace_and_preserves_order.checkpoint

Admission: test. Records: no records.

```text
Count cancellation checks at image boundaries.
```

### 24. tests/test_mask_tracing_overlap.py:61 — test_failures_drain_the_pending_trace_before_a_governor_retry.evaluate

Admission: test. Records: no records.

```text
Fail the second network call only after the first trace starts.
```

### 25. tests/test_mask_tracing_overlap.py:71 — test_failures_drain_the_pending_trace_before_a_governor_retry.trace

Admission: test. Records: no records.

```text
Always mark completion, including when tracing raises.
```

### 26. tests/test_mask_tracing_overlap.py:82 — test_failures_drain_the_pending_trace_before_a_governor_retry.checkpoint

Admission: test. Records: no records.

```text
Cancel before the second image while a trace is pending.
```

### 27. tests/test_mask_tracing_overlap.py:105 — test_public_cellpose_functions_remain_the_original_objects.evaluate

Admission: test. Records: no records.

```text
Return empty test flows without altering the real model API.
```

### 28. tests/test_mask_tracing_overlap.py:110 — test_public_cellpose_functions_remain_the_original_objects.trace

Admission: test. Records: no records.

```text
Use the original public dynamics function on the CPU.
```

### 29. tests/test_mask_tracing_overlap.py:115 — test_public_cellpose_functions_remain_the_original_objects.checkpoint

Admission: test. Records: no records.

```text
Allow both synthetic images to complete.
```

### 30. tests/test_mask_tracing_overlap.py:131 — _Model.__init__

Admission: test. Records: no records.

```text
Record the apparent device and network moves.
```

### 31. tests/test_mask_tracing_overlap.py:143 — _Model.eval

Admission: test. Records: no records.

```text
Return the original inputs for the fallback-path assertion.
```

### 32. tests/test_mask_tracing_overlap.py:153 — _Network.__init__

Admission: test. Records: no records.

```text
Start with no device transfers.
```

### 33. tests/test_mask_tracing_overlap.py:157 — _Network.to

Admission: test. Records: no records.

```text
Record a requested transfer and retain this stand-in.
```

### 34. tests/test_mask_tracing_overlap.py:192 — test_the_governor_retries_the_orchestration_with_the_same_images.evaluate

Admission: test. Records: no records.

```text
Refuse the first tile size without changing models or images.
```

### 35. tests/test_mask_tracing_overlap.py:210 — test_the_orchestration_observes_cpu_fallback_and_device_restoration.evaluate

Admission: test. Records: no records.

```text
Fail only GPU execution, allowing the governed CPU call to finish.
```
