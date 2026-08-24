"""The Tkinter GUI, kept together and out of the way of the rest of spaCR.

spaCR's interface is the Qt application in :mod:`spacr.qt`. What lives here
is the original Tkinter one: the window classes, the widget library, and the
per-module launchers (`mask`, `measure`, `annotate`, `classify`,
`make_masks`) that ship as console scripts and still work.

WHY IT IS A SUBPACKAGE. Eleven modules totalling about twelve thousand lines
sat beside the analysis code, so `spacr/` read as though Tkinter were half
the project. Gathering them says what they are, and makes the one real
cross-dependency visible: `spacr.settings` imports `spacrToolTip` from
`gui_elements`, which is the Tk widget library reaching into settings and
settings reaching back.

NOTHING MOVED FROM A CALLER'S POINT OF VIEW. `spacr.gui`, `spacr.app_mask`
and the rest are still importable at their old paths -- the console scripts
in setup.py name them -- because each is a shim re-exporting from here. A
notebook, a script or an installed `mask` command keeps working.
"""
