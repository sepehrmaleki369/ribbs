import logging
import sys

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)

fmt = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(fmt)

# (re)attach handler
if not logger.handlers:
    logger.addHandler(handler)