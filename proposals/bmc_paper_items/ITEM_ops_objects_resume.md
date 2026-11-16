# Item 958: resume the OPS objects phase per window

- **Number:** 958 (assigned by master 2026-11-16). **Priority:** low-medium.
- **Found by:** the PERISCOPE-on-Funk card, 2026-11-16. A whole-well `run_ops` objects phase (Funk plate A A1, CPU cpsam, 841 windows) was interrupted at 650/841. The rerun recomposed and resegmented every window, losing about 18 CPU hours.
- **Gap:** `_objects` in `spacr/ops_engine.py` (compose → segment → sew over windows) has no per-window checkpoint. Item 872 resumes ph_masks, measurement and archived wells per field or well, and 951 runs site subsets; neither covers objects windows.

## Spec
1. **Per-window checkpoint:**
   - Each window's labels (and any per-window side outputs the sew step reads) are written atomically (temp file + rename), with a done marker.
   - The key is the window's top/left plus a hash of its inputs and the settings that affect segmentation: model, diameter, thresholds, device-independent settings, compose blend, source tiles and their mtime/size, and the spaCR version stamp. A changed setting or input therefore never reuses a stale window.
   - Location: `<run>/objects_windows/<well>/` (configurable).
2. **Resume:** a rerun of the objects phase skips windows whose done marker and key match, and recomputes the rest.
3. **Identity:** sewing and numbering after a resume give `ops_objects` (and every downstream table) byte-identical to an uninterrupted run. Object numbering must not depend on completion order: it is assigned in the sew step in window order, as now.
4. **Cleanup:** after the well's sew step succeeds, window files are deleted unless `ops_objects_keep_windows=True`. The done markers' summary is written into the run journal.
5. **Defaults:**
   - checkpointing is on (`ops_objects_resume=True`), cleanup on (`ops_objects_keep_windows=False`);
   - outputs of a run with no interruption are byte-identical to today, so no schema bump;
   - disk use during the phase rises by one well's window labels at most.
6. **New settings:** alpha, with 9-locale strings; the deck entry goes on the existing grouped line "Runs resume, skip missing fields…".

## Proof check (pre-set)
- **Interrupt and resume:** on a test well (synthetic, and one real Funk A1 subset of at least 30 windows), kill the phase at a random window (fixed seed), resume, and compare `ops_objects` and the downstream tables by md5 against an uninterrupted run. They must be equal.
- **Stale-key control:** change one segmentation setting between the interrupted and the resumed run. Every window must be recomputed (0 reused), and the output equals a fresh run with the new setting.
- **Cleanup:** after success no window files remain (default), and with keep=True they all remain.
- **Default identity:** an uninterrupted run with the new defaults is byte-identical to nightly before 958.
