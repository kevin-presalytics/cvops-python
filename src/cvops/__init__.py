""" CVOps Python SDK"""
import sys
from pathlib import Path
from importlib.metadata import version
import cvops.config
import cvops.logging
from cvops.config import SETTINGS


sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))


cvops.logging.configure_logging(SETTINGS.log_level)


try:
    __version__ = version(__name__)
except Exception:  # pylint: disable=broad-except
    __version__ = "build"
