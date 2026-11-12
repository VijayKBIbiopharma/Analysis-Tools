"""Console logging, and only console logging.

The ECP is explicit twice over (sections A.5 and 5): logs go to the console, no
file handler. On the platform, stdout is collected into CloudWatch; a file
handler would hide the same lines inside the container and would need attached
storage the application is not supposed to rely on.

So this is the only place logging is configured, it attaches exactly one
handler, and it writes to stdout.
"""

import logging
import sys

LOGGER_NAME = "analysis_tools"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure(level: int = logging.INFO) -> logging.Logger:
    """Attach a single stdout handler. Safe to call more than once."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get(name: str = "") -> logging.Logger:
    base = logging.getLogger(LOGGER_NAME)
    return base.getChild(name) if name else base
