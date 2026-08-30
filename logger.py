"""
Centralized logging configuration.

Writes structured logs to both the console (minimal) and a rotating
log file (detailed) so we can trace every decision the system makes
without cluttering the user-facing terminal output.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened", extra={"order_id": "ORD-1015"})
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "support_bot.log")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with file + console handlers.

    File handler: DEBUG level, rotates at 5MB, keeps 3 backups.
    Console handler: WARNING level only (keeps terminal clean).
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Ensure the logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- File handler (detailed, for debugging) ---
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # --- Console handler (silent — only critical, all else goes to file) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    return logger
