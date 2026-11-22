# Queue rendering translation handover

Source branch: `ws-offload-codex-work-990queue`; base `79343f5b59f62507adce96feb68b912ebfa0222f`.
Exact portable source manifest: `tr_990h_sources.json`. No locale records or catalogs were created.

The renderer follows the authoritative card-v3 recipe SHA256 `0820ad998c9e40ca3c25996fc21a45dcf71f14e3f674913b3506c39d4bd1b754`. All six supplied tiles match exact RGB arrays and raw pixel MD5s and were inspected visually. Default Funk tiles use nuclei; three explicit cell-mask controls use the same table centroid. The Funk calibration is the recipe-declared assumption 0.33 µm per source pixel.

## Public API

`spacr.ops_cell_cycle.write_label_queue` at `spacr/ops_cell_cycle.py:1294`; admitted; SHA256 `544eca9ed4a7f8c1e82c169f1d1d86fe064b36d081c5d52c1681a86d36a72be7`.
Immutable original owning-arrival SHA256 `f1772545d4223a04aa1b3c57ab6e4d3d7a533236d8af313bcb768e97bcd1eb62`.

```text
Write blind stage-labelling tiles from unmasked context and the own object mask.

A fixed 96 by 96 source window starts at the rounded table centroid minus
48; cells whose window crosses the available field context are excluded.
``radius`` is a dilation of the own nucleus mask by one Euclidean disk,
independent of that window. Funk defaults use the nucleus mask. Pixels outside
the dilation are black, while neighbouring objects inside it remain visible.

The panels show grey DNA, grey tubulin and their blue-DNA/green-tubulin
composite with a zero red channel. Each panel is enlarged threefold with
nearest neighbours, followed by a one-pixel yellow own-mask outline. Two
four-pixel grey separators give a tile of 872 by 288 pixels. A white 10 µm
scale bar requires explicit source pixel calibration; no physical size is
inferred from image dimensions.

The fixed Funk display bounds are DAPI 550–23,039 and GFP 1,084–5,002 in raw
uint16 units. ``ranges`` overrides these bounds in selected-channel order;
no crop or queue statistics are used. Opaque PNG names map exactly to cells
in the separate ``queue.csv`` manifest, which must stay outside blind inputs.
Annotation 1 to 7 follows :data:`CLASSES`.

:param queue: unique cells from :func:`build_label_queue`, in display order;
    optional ``cy, cx`` are field coordinates and ``pixel_size_um`` supplies
    source calibration. A crop with a stored bounding box keeps its origin.
:param crop_root: a ``ph_crops`` context-crop project, or a field-context
    folder with ``crop_index.csv`` (keys plus ``cy, cx``), ``fields/`` and
    ``masks/``. The supplied Funk plate-A layout is supported directly; its
    recorded calibration assumption is 0.33 µm per pixel. Other field
    indices supply ``field_path, mask_path, pixel_size_um`` explicitly.
:param folder: the Annotate project folder to create.
:param labels: the CSV to which collected answers are appended.
:param channels: up to two zero-based crop indices, DNA first and tubulin
    second; None selects the first two available planes. Other stains can
    be selected in either slot with explicit display bounds.
:param ranges: fixed ``(low, high)`` pairs in selected-channel order, or a
    mapping with ``dna`` and optional ``mt`` pairs; None uses the Funk bounds.
:param radius: nonnegative nucleus-mask dilation radius in source pixels,
    30 by default, giving a 61 by 61 Euclidean disk.
:param crop_source: optional source exposing ``get_raw(row)`` and returning
    unmasked ``(pixels (C, H, W), own_object_mask)`` through existing spaCR
    readers. Pixel calibration can be supplied as ``pixel_size_um`` or an
    existing micrometre :class:`spacr.layers.Spacing` on the source. None
    uses native OPS context crops or full fields with the nucleus mask; masked
    legacy crops cannot provide neighbour context.
:returns: ``folder``; excluded identities and reasons are kept in ``queue.json``.
:raises ValueError: for invalid controls, missing context or nucleus masks,
    missing physical calibration, or a queue with no complete windows.
```

Canonical API blocks use zero-based indices:

- `spacr.ops_cell_cycle.write_label_queue#0`; SHA256 `0c86c6180a7ba4731a29c815b4974c655d05e30dcd971a4590e889968a0b0990`; admitted: Write blind stage-labelling tiles from unmasked context and the own object mask.
- `spacr.ops_cell_cycle.write_label_queue#1`; SHA256 `6acf059472552dd2d0de86ec17dd7ea10e1aa860faf426fed24f9e2e0cbf169d`; admitted: A fixed 96 by 96 source window starts at the rounded table centroid minus 48; cells whose window crosses the available field context are excluded. ``radius`` is a dilation of the own nucleus mask by one Euclidean disk, independent of that window. Funk defaults use the nucleus mask. Pixels outside the dilation are black, while neighbouring objects inside it remain visible.
- `spacr.ops_cell_cycle.write_label_queue#2`; SHA256 `f77125489de64417b1d2fef0f2679405aae06c3a840e77876591ceecdd5bf923`; admitted: The panels show grey DNA, grey tubulin and their blue-DNA/green-tubulin composite with a zero red channel. Each panel is enlarged threefold with nearest neighbours, followed by a one-pixel yellow own-mask outline. Two four-pixel grey separators give a tile of 872 by 288 pixels. A white 10 µm scale bar requires explicit source pixel calibration; no physical size is inferred from image dimensions.
- `spacr.ops_cell_cycle.write_label_queue#3`; SHA256 `b833806148a83760d905824f371db9238cafd536293ab1b64b302c456be5e36c`; admitted: The fixed Funk display bounds are DAPI 550–23,039 and GFP 1,084–5,002 in raw uint16 units. ``ranges`` overrides these bounds in selected-channel order; no crop or queue statistics are used. Opaque PNG names map exactly to cells in the separate ``queue.csv`` manifest, which must stay outside blind inputs. Annotation 1 to 7 follows :data:`CLASSES`.
- `spacr.ops_cell_cycle.write_label_queue#4`; SHA256 `3b1c2319f6cd8c9f1a424334af776fa965d4ee4475e9bc87b3587750d1eef636`; admitted: unique cells from :func:`build_label_queue`, in display order; optional ``cy, cx`` are field coordinates and ``pixel_size_um`` supplies source calibration. A crop with a stored bounding box keeps its origin.
- `spacr.ops_cell_cycle.write_label_queue#5`; SHA256 `1278b11cdfe27df3e6d10c1f7db007f6b784bdb0ebcff6263726796cc8c667fd`; admitted: a ``ph_crops`` context-crop project, or a field-context folder with ``crop_index.csv`` (keys plus ``cy, cx``), ``fields/`` and ``masks/``. The supplied Funk plate-A layout is supported directly; its recorded calibration assumption is 0.33 µm per pixel. Other field indices supply ``field_path, mask_path, pixel_size_um`` explicitly.
- `spacr.ops_cell_cycle.write_label_queue#6`; SHA256 `5fc84619f8a1c15ea1f06c74a86091a28e9ad53f81fb229878864df34d0c0ecf`; admitted: the Annotate project folder to create.
- `spacr.ops_cell_cycle.write_label_queue#7`; SHA256 `e0c0e9d4093933b487d289d6ddc905d1e59a215786a8cf17f24f99e59045b223`; admitted: the CSV to which collected answers are appended.
- `spacr.ops_cell_cycle.write_label_queue#8`; SHA256 `ddb6f737d5be6cec3471e6a4929a744ff5550d0a11315871009c0c3ea5ff1e81`; admitted: up to two zero-based crop indices, DNA first and tubulin second; None selects the first two available planes. Other stains can be selected in either slot with explicit display bounds.
- `spacr.ops_cell_cycle.write_label_queue#9`; SHA256 `81c72ba4eee99db056d74859cf43804e39c61778334b9281da10baf4edb32e39`; admitted: fixed ``(low, high)`` pairs in selected-channel order, or a mapping with ``dna`` and optional ``mt`` pairs; None uses the Funk bounds.
- `spacr.ops_cell_cycle.write_label_queue#10`; SHA256 `9617162013752ba399cca408ad34bfecf7aef23cfa890204824bf19e7d6bba29`; admitted: nonnegative nucleus-mask dilation radius in source pixels, 30 by default, giving a 61 by 61 Euclidean disk.
- `spacr.ops_cell_cycle.write_label_queue#11`; SHA256 `a3f8d7fa11fae4c624d3f06754db77101018ef56fed2b663d13e542c97f92629`; admitted: optional source exposing ``get_raw(row)`` and returning unmasked ``(pixels (C, H, W), own_object_mask)`` through existing spaCR readers. Pixel calibration can be supplied as ``pixel_size_um`` or an existing micrometre :class:`spacr.layers.Spacing` on the source. None uses native OPS context crops or full fields with the nucleus mask; masked legacy crops cannot provide neighbour context.
- `spacr.ops_cell_cycle.write_label_queue#12`; SHA256 `6d515706ce1d45ba2ff210b7b1bf1176c1cbabda9593b8a8ee756d15ddc0ebd6`; admitted: ``folder``; excluded identities and reasons are kept in ``queue.json``.
- `spacr.ops_cell_cycle.write_label_queue#13`; SHA256 `36fb0b3cd8f9669ef154e03f6cef1e8ab30c134bf14aa4c78a61e9826004fbad`; admitted: for invalid controls, missing context or nucleus masks, missing physical calibration, or a queue with no complete windows.

## Runtime and settings sources

Each exact text, hash, source location and admission is also recorded in the JSON manifest. Repeated occurrences share one locale record.

- `spacr/ops_cell_cycle.py:1360`; SHA256 `edd85747fff2ab6ad779a7e8b3a72cf689e2601156ef5f61d78d129ebbe78c3f`; admitted: Label queue radius must be a nonnegative integer in pixels.
- `spacr/ops_cell_cycle.py:1362`; SHA256 `6617c65416e701bd7c8ebeefd0797b45e8c7a2c4ff42bfa1ef4b15ff79ca0bd1`; admitted: Select at least one cell and include each cell only once in the label queue.
- `spacr/ops_cell_cycle.py:1367`; SHA256 `80064cf245b6391817d72faa0554bcab078916069bf783c3a5ad076cd81994d8`; admitted: Select one or two distinct zero-based label queue channel indices.
- `spacr/ops_cell_cycle.py:1376`; SHA256 `61e286abd43d316af3f6987fb249e88218d8e3002e86ef90357ac558947c1de4`; admitted: Give one finite increasing display range for each selected label queue channel.
- `spacr/ops_cell_cycle.py:1409`; SHA256 `e89845a6766b9c5e2a18572b43599d13c7745c47188887fe7c77a3f3dcd9f23d`; admitted: Stage labelling needs unmasked full-field context; enable context in the crop source.
- `spacr/ops_cell_cycle.py:1523`; SHA256 `8d33765132d9377c7a9bed401724af5f63d050947717df385954f68627e05179`; admitted: No selected cells have a complete 96-pixel context window; choose cells away from field edges.
- `spacr/ops_cell_cycle.py:1371`; SHA256 `706ed6cb037e070fcaa68d035b8388fc036de83b55565177ab1ec7d1341f4c0b`; admitted: A label queue range mapping needs a dna pair and an optional mt pair.
- `spacr/ops_cell_cycle.py:1384`; SHA256 `b96e5fd338113cd33b43a46c27b1153b7961c6a1f1333383be6be08bf00a01cf`; admitted: The context crop index needs columns {columns}.
- `spacr/ops_cell_cycle.py:1386`; SHA256 `d9c537a1d22b13d39723c260bdc2ddb9cb0fd813ff85b4400a7d9ab064d01d96`; admitted: A multi-well context index needs explicit field_path and mask_path columns.
- `spacr/ops_cell_cycle.py:1448`; SHA256 `886d42981fcf5af99cf5151fbe56c4fb4913ec585367b91bf01c93e13bd8b859`; admitted: Label queue crops must contain channels, height and width.
- `spacr/ops_cell_cycle.py:1450`; SHA256 `27e7a03b00d311c2a1c5781bf2e454911d57a0300d0037fafea2ab67e5def261`; admitted: Stage labelling needs the own nucleus mask for every selected cell.
- `spacr/ops_cell_cycle.py:1453`; SHA256 `1dbbe9e245327e5fb0fd41a498b161096e663c3303072a1c007e9e9ce716c7ad`; admitted: Return only the selected nucleus mask; keep neighbouring objects in the image pixels.
- `spacr/ops_cell_cycle.py:1468`; SHA256 `fb26b49e376af8034574813cee87baac670ce9530fa100c8bfb9b33437715201`; admitted: A selected label queue channel is absent from a crop; check the crop channel order.
- `spacr/ops_cell_cycle.py:1476`; SHA256 `61e286abd43d316af3f6987fb249e88218d8e3002e86ef90357ac558947c1de4`; admitted: Give one finite increasing display range for each selected label queue channel.
- `spacr/ops_cell_cycle.py:1484`; SHA256 `1f4ae1d3c894f16ac360b3c3faff6c0a0c8a4935219393763d2020a36c62113b`; admitted: A 10 µm scale bar needs measured pixel_size_um or source Spacing in micrometres.
- `spacr/ops_cell_cycle.py:1514`; SHA256 `8d2ae3d069a3eacc92e6b2934d36a7c9fd68e43230a311ecd78066ba4dcccaf0`; admitted: no tubulin channel
- `spacr/ops_cell_cycle.py:1393`; SHA256 `9bf17683da69b99c1c83f4f7de8d8bd5e96d0429ecd596d5eeac27de3b3f6783`; admitted: no crops of plate {plate} under {root}; run ph_crops first
- `spacr/ops_cell_cycle.py:1396`; SHA256 `e4f8c1e96a3f36a0dc7716473f198c7844eda06eb33b0092b0ed5f14f9c7313a`; admitted: Legacy OPS shards contain masked crops; provide full-field or unmasked nucleus context crops.
- `spacr/ops_cell_cycle.py:1398`; SHA256 `2d48f76f77de34c31bf05a2841cb42b88c4313ed08b5ae1a1b2d9498447d78b5`; admitted: The crop index contains duplicate cells; rebuild ph_crops before labelling.
- `spacr/ops_cell_cycle.py:1425`; SHA256 `32fb81785e54828996271636b4634ba507cb54ddde768539ca6bb34a6e06524b`; admitted: Cell {cell} has no crop; run ph_crops for its well.
- `spacr/ops_cell_cycle.py:1471`; SHA256 `a8f91bad16d47e2d7d0ed406e5b969441607e0ddbb4bdcd0662a6491f00e84fb`; admitted: Funk display bounds require raw uint16 images; give explicit ranges for other pixel units.
- `spacr/ops_cell_cycle.py:1436`; SHA256 `58df9346fe0d9177f5c09708ac46f832d41f9b7e900aaf6f73c654e9b65842d9`; admitted: The context field needs a nuclei label mask for stage labelling.
- `spacr/ops_cell_cycle.py:1439`; SHA256 `e4f8c1e96a3f36a0dc7716473f198c7844eda06eb33b0092b0ed5f14f9c7313a`; admitted: Legacy OPS shards contain masked crops; provide full-field or unmasked nucleus context crops.
- `spacr/ops_cell_cycle.py:1443`; SHA256 `b3cb7733945b2dc10fe45f72a20233ad452e061b7eb857c8b45a15ced4378f0c`; admitted: Stage labelling needs context crops: set crop_margin_px and keep crop_zero_outside off.

Setting `ops_cell_cycle_label_channels`: default `None`, types `list | None`, panel `Cell-cycle stages`; API owner `spacr.ops_cell_cycle.run_cell_cycle_phase`.

- `setting_labels/ops.ops_cell_cycle_label_channels`; `spacr/ops_settings.py` declarations [303, 874, 1417, 3759]; SHA256 `844b6ddc2b66263298913034f31295912626e408b605b01a3137f597c3b84507`; admitted: Cell cycle label channels
- `setting_labels/ops_cell_cycle_label_channels`; `spacr/ops_settings.py` declarations [303, 874, 1417, 3759]; SHA256 `656c44124eadd03229be4e70ea9d4be812ec2c76de1c8bec4af543f67be6134a`; admitted: Ops cell cycle label channels
- `setting_tooltips/ops_cell_cycle_label_channels`; `spacr/ops_settings.py` declarations [303, 874, 1417, 3759]; SHA256 `14fc329af75c371cd668147b11d8a21516ca446d734ca806199cd5893f51782d`; admitted: Up to two crop channel indices for stage labelling, starting at 0. Select DNA first and tubulin second, or select another stain with explicit display bounds. Panels show grey DNA, grey tubulin and their blue-DNA/green-tubulin overlay. Funk defaults use fixed raw DAPI and GFP bounds, a 96-pixel unmasked context window and the nucleus mask dilated by 30 pixels. None selects the first two crop channels. Default None. Alpha.

## No-record inventory

The manifest lists test, private and nested test docstrings with file and line. No production private or nested function was added. These entries need no locale records.

## Exact historical bindings

Writer-only seven-line frozen hook approved v80; separate exact inherited module hook approved v84; both applied. Original arrival fixture unchanged.

- `spacr.ops_cell_cycle`: historical SHA256 `c0bebd33ff6b5bfabb34284c41287386b388160fdda350b3829c0f35b8b5b44c` → unchanged base/current SHA256 `7cac209740b296dba04d2e6b904b53a09efa25dfd4edeb155922e4f77bd42c8d`.

## Validation dependencies

The raw validation receipts live in the private task recovery folder. The untouched workflow723 expected-file-count failure remains a master-landing dependency; it is not counted as passing. Catalog rebuilds and locale-record updates remain assigned to the coordinator.

## Final validation and record ownership

Production source SHA256 `ac6b8cb2afbe772ab2357ef321e8dd2c1c8ea15c5011ba949a46a2d793deb2f2`. No production change followed the corrected renderer checks.

- `affected_renderer_source`: 34 passed in 48.54s
- `items`: 1283 passed, 10 skipped, 152 warnings, 959 subtests passed in 196.49s (0:03:16)
- `ops`: 1 failed, 1180 passed, 29 skipped, 185 warnings, 102 subtests passed in 308.13s (0:05:08)
- `settings`: 630 passed, 2 skipped, 369 subtests passed in 62.26s (0:01:02)
- `ratchets_filtered`: 5 failed, 4082 passed, 76 skipped, 3 xfailed, 69 warnings in 794.05s (0:13:14)
- Approved fixture followup: 3 passed in 79.89s (0:01:19). This is a separate receipt; overlapping cohorts are not added together.

The coordinator approved both exact fixture patches in the appended v89 answer (inbox SHA256 `9aa9e6106005ff531057e7bba130914c19e5ffb5e02a9a45f376a3e370c02c71`). The API extractor reuses the strict writer/module adapters; the tooltip roster contains exactly the new `ops_cell_cycle_label_channels` key after retiring six already-catalogued entries. All original assertions remain unchanged.

The full filtered ratchet receipt remains **5 failed, 4082 passed**, followed by **3 passed** for the approved fixture corrections. The remaining two actual failures are explicitly confirmed landing dependencies:

- `tests.test_module_coverage_ratchet::test_coverage_workflow_is_sharded_artifact_safe_and_required`
- `tests.qt.test_external_i18n_catalogs::test_empty_flowchart_caption_translates_without_changing_python_none`

Workflow723 is owned by master landing `6df970bf79`; the stale Swedish reviewed runtime source is owned by Home. Neither is counted as passing. The exact thirty catalog-dependent deferrals remain listed under `validation.exact_catalog_deferrals` in the source manifest; strict English/source checks were selected. The original unfiltered ratchet attempt was cancelled after an actual 16 GiB OOM, returned 143 without XML, and is not counted as completed. The filtered rerun used 4 workers and 24 GiB and had no OOM.

GPU-rule scope: the affected queue files render through NumPy, SciPy, PIL and shared IO; the original learner tests use CPU HGB, including the explicit CNN-to-tabular fallback case. There are no affected GPU-touching test files, so a GPU-visible cohort is not applicable; no GPU result is claimed. No rebase or pilot-source fix was applied.

The no-record inventory includes 59 test/private/nested entries, including all three new test-module headers. No locale records, catalog pins, workflows or generated publication assets are included.
