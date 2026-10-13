# Final scoped Qt selector ownership repair (2026-10-07)

Only `test_cov_r4_settings_model.py::test_a_selector_with_no_signal_is_left_unconnected_rather_than_fatal`
showed a retained Qt population. It built four synthetic selector controls
without an owner; the regression-backend control also built child widgets.
The test now gives its `SettingsWidgets` model and all four controls a
`qtbot`-owned `QWidget` parent. Its signal and widget-type assertions are
unchanged. `test_cov_r5_settings_model.py` is byte-for-byte identical to its
pre-audit source in the final commit.

The exact 11 affected nodes plus the next-file field-fade sentinel passed
12/12 before and after under an enforced 4 GiB cap, hidden CUDA and offscreen
Qt. The passive observer sampled at next-test setup and did not run GC or
process Qt events.

| Boundary | Before r4 fix | Final r4-only fix |
| --- | ---: | ---: |
| First r5 test after signal-less selector | 20 live widgets | 0 |
| Later r5 setups and field-fade sentinel | 20 each | 0 each |

The pre-fix 20 comprised three `_Voiceless` selectors and the composite
regression-backend field and children. A separate r5-only 11-test probe found
zero live widgets at every boundary even before changes. This is why the
earlier local r4+r5 experiment (`a535cb6330`, archived as `36ef41a87f`)
is historical and must not be cherry-picked. The final integration sequence
is only `cc8db83917` followed by this receipt commit.

The final 12-test process ended near 575.9 MiB RSS, like the pre-fix probe.
This is a measured widget-ownership repair, not a measured RSS reduction or
completion of uninterrupted serial-memory acceptance. The before journal
was recorded at `76b72a836b`; the final source is based on `70668186e5`.
The relevant settings-model, field-fade and two test-file blobs match the
before source at that parent, as recorded in `manifest.json`. Ruff and
`git diff --check` passed.
