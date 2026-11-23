"""
logger.py
One place to create loggers. Logs go to the console AND to logs/app.log.
Usage in any file:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Hello")
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import load_config, resolve_path

_configured = False


def _setup() -> None:
    """Create the main 'rag' logger once (so we never add handlers twice)."""
    global _configured
    if _configured:
        return

    cfg = load_config()["logging"]
    log_dir = resolve_path(cfg["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("rag")
    root.setLevel(getattr(logging, cfg["level"].upper(), logging.INFO))
    root.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file: when it reaches ~1 MB a new file starts (keeps 3 old ones)
    file_handler = RotatingFileHandler(
        log_dir / cfg["file_name"], maxBytes=1_000_000,
        backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    _setup() 
    return logging.getLogger(f"rag.{name}")