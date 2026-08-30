"""
Centralized logging configuration.

Uses structured JSON logging for the file handler (machine-parseable,
grep-friendly, compatible with log aggregation tools like ELK/Datadog)
and a clean human-readable format for console output.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Order checked", extra={"order_id": "ORD-1015", "status": "delivered"})
"""

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "support_bot.log")
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class JSONFormatter(logging.Formatter):
    """
    Emit each log record as a single JSON line.

    Structured logs are the standard in production systems because they're:
    - Machine-parseable (grep, jq, log aggregators)
    - Searchable by field (filter by order_id, intent, error type)
    - Compatible with ELK Stack, Datadog, CloudWatch, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": os.path.relpath(record.pathname, _PROJECT_ROOT),
            "line": record.lineno,
            "function": record.funcName,
        }

        # Include any extra fields passed via log.info("msg", extra={...})
        # Skip internal LogRecord attributes
        skip = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "pathname", "filename", "module", "levelno", "levelname",
            "msecs", "thread", "threadName", "process", "processName",
            "message", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ReadableFormatter(logging.Formatter):
    """Human-readable format for console output during development."""

    COLORS = {
        "DEBUG": "\033[90m",      # grey
        "INFO": "\033[36m",       # cyan
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = f"{color}{timestamp} | {record.levelname:<7} | {record.name} | {record.getMessage()}{self.RESET}"

        if record.exc_info and record.exc_info[0] is not None:
            msg += f"\n{''.join(traceback.format_exception(*record.exc_info))}"

        return msg


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with structured JSON file output and readable console output.

    File handler: DEBUG level, JSON lines, rotates at 5MB, keeps 3 backups.
    Console handler: WARNING level only (keeps terminal clean for user interaction).
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Ensure the logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- File handler (structured JSON, for production/debugging) ---
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # --- Console handler (human-readable, minimal) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(ReadableFormatter())
    logger.addHandler(console_handler)

    return logger
