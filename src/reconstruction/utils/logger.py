from __future__ import annotations
import logging
import os
from typing import Optional

def configure_logging(level: int = logging.INFO, rich: Optional[bool] = True) -> None:
    """
    Configure root logger with a nice console handler. Uses 'rich' if available.
    """
    # Avoid duplicate handlers in notebooks / reruns
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    if rich:
        try:
            from rich.logging import RichHandler  # type: ignore
            handler = RichHandler(rich_tracebacks=True, show_path=False)
            handler.setLevel(level)
            logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=[handler])
            return
        except Exception:
            pass

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logging.basicConfig(level=level, handlers=[handler])

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)