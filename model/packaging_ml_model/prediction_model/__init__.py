import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT.parent))

from prediction_model.config import config

with open(os.path.join(config.PACKAGE_ROOT,'VERSION')) as f:
    __version__=f.read().strip()