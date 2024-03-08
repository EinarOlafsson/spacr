"""
Copyright © 2024 Something
"""

from importlib.metadata import PackageNotFoundError, version
import sys
from platform import python_version
import torch

try:
    version = version("spacr")
except PackageNotFoundError:
    version = "unknown"

version_str = f"""
spacr version: \t{version} 
platform:       \t{sys.platform} 
python version: \t{python_version()} 
torch version:  \t{torch.__version__}"""
