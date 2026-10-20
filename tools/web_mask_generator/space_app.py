"""Hugging Face app_file entry when this repository is the Space source."""
from pathlib import Path
import sys
import importlib
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if os.environ.get("SPACR_WEB_ZERO_GPU") == "1":
    importlib.import_module("spaces")

from tools.web_mask_generator.app import main


if __name__ == "__main__":
    from tools.web_mask_generator.provision import managed_startup
    managed_startup()
    main()
