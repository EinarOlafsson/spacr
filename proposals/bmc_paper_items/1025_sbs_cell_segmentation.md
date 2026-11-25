# Item 1025: spaCR makes its own SBS cells

Status: requested by Einar 2026-11-25 ("SpaCR should make its own SBS cells"), relayed by the Home spacr master. To be built on the WS lane after L5, at P2, as an alpha feature that is off by default. It does not block L5, the A1 rerun or plate A. Item 1023 stays the default unless this beats it.

## Why

spaCR segments only nuclei in the sequencing (SBS) images. Brieflow also segments SBS cells and assigns reads to the cell mask. On plate A well A1 (report half, B1 gate, top share at least 0.9):

| Read assignment | Genotyped | Split-read agreement |
|---|---|---|
| Brieflow SBS cell masks | 68.0 % | 97.0 % |
| spaCR nucleus + 3 px (item 1023) | 67.6 % | 97.8 % |

To test cell-mask assignment fairly, spaCR needs its own SBS cells.

## What to build

1. SBS cell segmentation, in the same shared segmentation code that Measure and OPS phenotype use. One operation, one implementation: no OPS-only routine.
   - Input: an SBS cytoplasm proxy made from the registered cycle images, for example a per-pixel aggregate of the four base channels after background removal, as Brieflow does.
   - Seeds: the SBS nuclei spaCR already makes.
   - Methods: both the shared Cellpose path (seeded by nuclei) and a seeded watershed on the proxy. The paper evaluation chooses between them; neither is assumed.
2. Nucleus-cell pairing: every SBS nucleus gets exactly one cell, or has_cell = False. This uses the same routine as phenotype, spacr.measure._assign_nuclei_to_cells (item 1024 also works in that function; build on its final branch).
3. New setting `ops_read_assignment`:
   - `nucleus_reach` (default): the current behaviour, with ops_read_neighbour_reach and ops_footprint;
   - `cell_mask`: assign reads to the SBS cell mask.
4. Storage: the masks go in fov.ome.zarr like the other OPS labels. The store keeps the (well, site, label) key space.
5. A new schema number and a DEFAULT_CHANGES row that keeps the current behaviour. Ship as alpha; catalogue the symbols in the same commit; run the full ratchets.

## How it will be judged (paper side)

- Default: `cell_mask` is the third read-assignment mode in the yield-at-error-cap optimisation, with the same tiles, nuclei, decoys and split reference: tune half for selection, report half with CIs. It becomes the default only if it wins there and is confirmed on a second well (LaC024E A1, which the paper session packages). Brieflow masks exist only for plate A A1.
- Mask QC: IoU against Brieflow SBS cells on A1, plus a 20-site montage for Einar.
