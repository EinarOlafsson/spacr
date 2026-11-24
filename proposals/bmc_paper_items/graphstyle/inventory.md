# Function inventory coverage

Source: `299179d9530ac618d82f11cc823f3ad8abd1c001`.

729 Python files parsed; 32922 lexical function definitions; no parse failures.
1035 broad candidates all classified. 1607 unique reviewed entries across 213 files, including supplemental drawing/support functions and explicit exclusions.
No duplicate IDs, missing candidates or source-coordinate mismatches. Module-scope entries are marked `<module>`.

The [CSV](functions.csv) is the complete per-function table. Counts by disposition preserve each reviewer’s classification vocabulary; they are not counts of compliance failures.

| Disposition | Entries |
|---|---:|
| annotation | 14 |
| artist_adapter | 12 |
| creator | 120 |
| diagnostic | 2 |
| excluded | 53 |
| excluded_decoration | 88 |
| excluded_nonplot | 2 |
| excluded_raw_io | 16 |
| excluded_text_or_control | 39 |
| image | 7 |
| image-helper | 2 |
| image-report | 1 |
| image_or_attribution | 74 |
| image_panel | 1 |
| interactive_adjustment_chart | 3 |
| interactive_schematic | 3 |
| layout | 2 |
| plot | 98 |
| policy | 7 |
| renderer | 6 |
| statistical_creator_or_family_wrapper | 217 |
| style-support | 25 |
| style_policy | 48 |
| support | 38 |
| support_non_drawing | 317 |
| wrapper | 278 |
| writer | 81 |
| writer_or_host | 53 |

## Review families

| Review | Entries |
|---|---:|
| [core_plot_audit.md](reviews/core_plot_audit.md) | 504 |
| [general_plot_audit.md](reviews/general_plot_audit.md) | 180 |
| [gui_training_audit.md](reviews/gui_training_audit.md) | 549 |
| [ops_audit.md](reviews/ops_audit.md) | 65 |
| [assay_regression_audit.md](reviews/assay_regression_audit.md) | 134 |
| [timelapse_audit.md](reviews/timelapse_audit.md) | 57 |
| [remaining_audit.md](reviews/remaining_audit.md) | 118 |

Exposure estimates come from caller routes, not measured usage. Empty checker output is not proof of complete style compliance.
