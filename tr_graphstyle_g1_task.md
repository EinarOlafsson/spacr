# Graph style G1 translation task

This branch is still in progress; this file will be extended before FINAL.

The first isolated commit fixes error-bar/category associations in multiple-measurement bar plots. It adds no runtime caption, setting, tooltip or public API docstring. The private helper `spacr.plot.spacrGraph._create_bar_plot` is absent from the canonical public API dictionary.

Internal helper docstring, `spacr/plot.py`, `_create_bar_plot`:

- Previous: Helper method to create a bar plot with consistent bar thickness and centered error bars.
- Revised: Draw bars with consistent width and error bars matched to displayed categories.

No catalog, pin or reviewed translation record is changed here. The rest of G1 is pending.
