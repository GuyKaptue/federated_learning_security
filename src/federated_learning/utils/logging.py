"""Package logger (stdout + optional file under logs/)."""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "federated_learning", level: int = logging.INFO, logfile: str | Path | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile); fh.setFormatter(fmt); log.addHandler(fh)
    log.setLevel(level)
    return log
