"""Hugging Face app_file entry when this repository is the Space source."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.web_mask_generator.app import main


if __name__ == "__main__":
    main()
