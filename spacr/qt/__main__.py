"""`python -m spacr.qt` launches the Qt GUI."""
from __future__ import annotations

import sys

from . import run


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
