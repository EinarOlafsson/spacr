# Matched fixture widget census, 2026-10-07

This records the test-only ownership repair in `35d395b6e7`. The two affected
test files were run before and after that commit against the same production
`settings_model.py` blob. The fourth test is the real field-fade boundary, not
a synthetic cleanup. Both runs passed all four nodes under a 4 GiB cap with
CUDA hidden, Qt offscreen, Python 3.12.13 and PySide6/Shiboken6 6.11.2.

At the field-fade test's setup boundary, the old fixtures left 447 live
widgets, 149 of them top-level. The repaired fixtures had 449 live widgets but
only one top-level owner. At the end of the field-fade test, the old run still
had 447/149; the repaired run had 0/0. This is a widget ownership and normal
teardown observation. It does not establish the cause of a hosted native crash.
The observer only calls `QApplication.allWidgets()`; it performs no forced GC,
event pumping, widget deletion, or app-level cleanup.

`census-before.jsonl` and `census-after.jsonl` contain every setup and
teardown snapshot, including all type counts. `census_plugin.py` is the exact
scratch-only observer. The two `.log` files are the corresponding complete
pytest outputs. `receipt.json` binds all evidence and fixture/source blobs.

The selected nodes, in order, were:

```text
tests/qt/test_no_control_is_inert.py::TestTheMergeBoxesAreRead::test_the_png_list_box_reaches_the_joiner
tests/qt/test_no_control_is_inert.py::TestTheClassifyPanelHasNoSupersededControl::test_it_is_not_offered[class_folder_names]
tests/qt/test_widgets.py::test_all_settings_booleans_use_switches
tests/qt/test_field_fade.py::test_turning_it_off_restores_the_plain_field
```

Replay from a materialized repository tree at the receipt's `repo_head`, with
the before or after fixture blobs restored as specified in `receipt.json`:

```sh
env CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  PYTHONPATH=.:features/data/47_fixture_widget_census_2026-10-07 \
  SPACR_WIDGET_CENSUS=/tmp/spacr-widget-census.jsonl \
  tools/run_capped.sh 4G python -m pytest -q -p no:randomly \
  -p no:cacheprovider -p census_plugin \
  'tests/qt/test_no_control_is_inert.py::TestTheMergeBoxesAreRead::test_the_png_list_box_reaches_the_joiner' \
  'tests/qt/test_no_control_is_inert.py::TestTheClassifyPanelHasNoSupersededControl::test_it_is_not_offered[class_folder_names]' \
  tests/qt/test_widgets.py::test_all_settings_booleans_use_switches \
  tests/qt/test_field_fade.py::test_turning_it_off_restores_the_plain_field
```
