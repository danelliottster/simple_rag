"""Simple program-wide logging configuration helper.

Call `configure_logging()` from your entry-point (for example `build_rag.py`).
This keeps modules/libraries free of configuration and lets the main script control
handlers, levels and an optional log file.
"""
from logging.config import dictConfig
import logging

def configure_logging(level: str = "INFO", logfile: str = None) -> None:
    """Configure the root logger with a console handler and optional file handler.

    Args:
        level: Logging level name, e.g. 'INFO', 'DEBUG'.
        logfile: Optional path to a file to also write logs to.
    """
    handlers = ["console"]
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(name)s %(levelname)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            }
        },
        "root": {"level": level, "handlers": handlers},
    }

    if logfile:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": level,
            "filename": logfile,
            "encoding": "utf-8",
        }
        config["root"]["handlers"] = ["console", "file"]

    dictConfig(config)
