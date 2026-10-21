"""Analyze staged runtime catalogs while preserving the repository sources."""
from pathlib import Path
import atexit
import runpy
import tempfile


def pre_find_module_path(pfmp_api):
    """Redirect only the locale package to an owned, projected build copy."""
    root = Path(__file__).resolve().parents[3]
    helper = runpy.run_path(str(root / "_spacr_catalog_build.py"))
    staging = tempfile.TemporaryDirectory(prefix="spacr-catalogs-")
    try:
        search = helper["stage_runtime_catalogs"](root, staging.name)
    except BaseException:
        staging.cleanup()
        raise
    atexit.register(staging.cleanup)
    pfmp_api.search_dirs = [str(search)]
