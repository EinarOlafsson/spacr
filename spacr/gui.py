"""Compatibility shim: this module now lives in :mod:`spacr.legacy_tk`.

The Tkinter GUI was gathered into one subpackage so it no longer sits beside
the analysis code. This name is kept because setup.py's console scripts and
every existing notebook import it from this path.

IT IS THE SAME MODULE OBJECT, not a copy of its names. Re-exporting with
`import *` would give two modules with two copies of every attribute, and
patching one would not be seen by the other -- which is exactly what a test
that monkeypatches `spacr.gui_elements.apply_theme` needs to work. Binding
the old name in `sys.modules` makes `spacr.gui_elements` and
`spacr.legacy_tk.gui_elements` one object with one identity.
"""
import sys

from .legacy_tk import gui as _real

sys.modules[__name__] = _real
