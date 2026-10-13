# Full field-edge ripples on current nightly source

Private candidate `c797c6d33ab02b2e235082dbce722fe070ef2e47` applies the complete-boundary wave change to published nightly `75c72e858a15c67768b503470773cb5db19e2e51`. Both newer optional Section import guards remain. A real detached or deleted splitter handle now ends the fold without emitting a ghost edge or raising.

Twelve affected Qt files passed 171 tests under a 4 GiB cap using Qt 6.12 offscreen. Branch coverage directly observed every changed statement and arc in ambient, splitter, section and dock. The preferences file changed only popup-help text and was not selected for branch coverage. Earlier `owning.log.gz` retains the detached-pane failure before the follow-up repair; `owning-final.log.gz` is the passing current-source result.

At density 3 with actual 1920×1080 and 3840×2160 native engine buffers, one cold and three fixed-time steady QImage shades were measured in separate, capped CPU processes. The old two-side material used six point events; the new material uses two full-edge events. Their single-process steady medians were 60.5→53.1 ms at 1080p and 222.9→189.5 ms at 4K. The popup changed from one centre point to one four-segment perimeter; 4K cold cost was 144.7→175.8 ms, steady 17.3→18.9 ms. These scenes paint different intended pixels, and the timings say nothing about displayed frame rate, GPU performance or 24 FPS acceptance.

Normal extraction on both exact source trees found 13,232 API keys and 7,288 UI strings on each. One `field_ripple_for_widget` parameter description changed, and one popup-frequency help sentence was replaced. API and runtime catalogs need their normal owner regeneration; no generated files were edited here.

Run `python verify.py` for all stored and decompressed payload hashes. Run `python verify.py --source` in the exact private candidate checkout to check the final source blobs too.
