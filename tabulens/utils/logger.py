import logging
from pathlib import Path
from datetime import datetime
import os
from typing import Optional

def get_logger(
    name: str,
    log_dir: Path = Path(os.path.join(os.getcwd(), "logs")),
    log_level: Optional[dict] = None,
    verbose: bool = False,
    filehandler: bool = True
) -> logging.Logger:
    
    """
    Create a logger with console and file handlers.

    Args:
        name (str): Name of the logger.
        log_dir (Path): Directory to save log files.
        log_level (str): Logging level for console output.
        filehanler (bool): Whether to create a file handler.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_level is None:
        log_level = {"console": "WARNING", "file": "DEBUG"}
    log_level = {**{"console": "DEBUG", "file": "DEBUG"}, **log_level}

    # Ensure the logs directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    # if not logger.handlers:
    logger.setLevel(logging.DEBUG)

    # Console handler
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(log_level.get("console", "DEBUG").upper())
        ch.setFormatter(logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(ch)

    if filehandler:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = log_dir / f"run_{ts}.log"
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fh.setLevel(log_level.get("file", "DEBUG").upper())
        fh.setFormatter(logging.Formatter(
            "%(asctime)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

    return logger