import logging
from pathlib import Path


def setup_logger(log_file: str = "truss_analysis.log") -> logging.Logger:
    logger = logging.getLogger("truss_analysis")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        log_path = Path(log_file)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


logger = setup_logger()
