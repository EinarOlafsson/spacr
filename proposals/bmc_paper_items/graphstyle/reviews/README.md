# Source review notes

These notes support the complete [function inventory](../functions.csv) and [main specification](../../graphstyle_audit.md). They distinguish source-only findings from the parent-run synthetic renders. The main specification resolves authority ambiguities and defines the implementation proposal.

References within individual notes to `*_rows.json`, `ops_inventory.json` or `.codex-audit/` identify intermediate local audit files. Their reviewed rows are consolidated in the function inventory above. The original `before_report.json` is published unchanged as [render measurements](../thumbnails/before_v1/measurements.json). Source-coordinate and coverage checks passed before three geometry findings were clarified to distinguish missing constraints from measured violations; those wording changes preserve every function ID and location.

- [core_plot_audit.md](core_plot_audit.md)
- [general_plot_audit.md](general_plot_audit.md)
- [gui_training_audit.md](gui_training_audit.md)
- [ops_audit.md](ops_audit.md)
- [assay_regression_audit.md](assay_regression_audit.md)
- [timelapse_audit.md](timelapse_audit.md)
- [remaining_audit.md](remaining_audit.md)
