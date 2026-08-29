"""Build-time spaCR version used by the lightweight package facade.

This literal is kept in lockstep with ``setup.py::VERSION`` by
``packaging/release.py bump``.  Keeping the installed artifact's own version
beside the package lets ``import spacr`` report ``spacr.__version__`` without
importing :mod:`importlib.metadata` and its email, pathlib, inspect and typing
stack.  The public :mod:`spacr.version` module still performs a metadata lookup
when a caller explicitly asks it to inspect the installed distribution.
"""

from __future__ import annotations

__version__ = "1.5.0.4"
