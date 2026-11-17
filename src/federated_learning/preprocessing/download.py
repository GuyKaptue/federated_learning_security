"""Download + integrity verification of raw public datasets into data/raw/<key>/ (never redistributed)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from federated_learning.preprocessing.registry import SPECS, RawFile, get_spec
from federated_learning.utils.hashing import sha256_file
from federated_learning.utils.paths import RAW_DIR


def raw_dir(key: str) -> Path:
    return RAW_DIR / get_spec(key).key


def raw_path(key: str, filename: str) -> Path:
    return raw_dir(key) / filename


def fetch_file(key: str, f: RawFile, timeout: int = 60) -> Path:
    """Return the verified local path of `f`, downloading it first if missing. Refuses files with a wrong hash."""
    if f.url is None:
        raise ValueError(f"{f.filename} is bundled with a library; nothing to download")
    p = raw_path(key, f.filename)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not f.url.startswith("https://"):
            raise ValueError("only https URLs are allowed")
        with urllib.request.urlopen(f.url, timeout=timeout) as r:  # nosec B310
            p.write_bytes(r.read())
    got = sha256_file(p)
    if got != f.sha256:
        raise RuntimeError(f"{key}/{f.filename}: SHA-256 mismatch (got {got}); refusing to use an unverified file")
    return p


def download(key: str) -> list[Path]:
    spec = get_spec(key)
    return [fetch_file(spec.key, f) for f in spec.files if f.url is not None]


def download_all() -> dict[str, list[str]]:
    return {k: [str(p) for p in download(k)] for k in SPECS}
