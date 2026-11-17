"""Wire protocol of the networked federation (framing over a TCP/TLS stream).

Frame = MAGIC(4) | header_len(u32 BE) | payload_len(u32 BE) | header(JSON, UTF-8) | payload(raw little-endian float32 values)

* The header is a small JSON object with a mandatory string field "type".
* The payload carries one or more float32 vectors back to back; the header says how many values each has ("sizes").
* Length limits are enforced BEFORE any allocation, so a peer cannot make the receiver allocate unbounded memory.
This module only moves bytes; authentication and encryption are provided by the TLS layer (see tls.py).
"""
from __future__ import annotations

import json
import socket
import struct

import numpy as np

MAGIC = b"FLN1"
MAX_HEADER = 64 * 1024
DEFAULT_MAX_PAYLOAD = 256 * 1024 * 1024
_PREFIX = struct.Struct(">4sII")


class ProtocolError(Exception):
    """Malformed, oversized or truncated message."""


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ProtocolError("connection closed mid-message")
        buf += chunk
    return bytes(buf)


def send_message(sock: socket.socket, header: dict, arrays: list[np.ndarray] | None = None) -> int:
    """Send one frame; returns the number of bytes written to the stream."""
    arrays = arrays or []
    header = dict(header, sizes=[int(a.size) for a in arrays])
    hb = json.dumps(header, separators=(",", ":")).encode()
    payload = b"".join(np.ascontiguousarray(a, dtype="<f4").tobytes() for a in arrays)
    if len(hb) > MAX_HEADER:
        raise ProtocolError("header too large")
    frame = _PREFIX.pack(MAGIC, len(hb), len(payload)) + hb + payload
    sock.sendall(frame)
    return len(frame)


def recv_message(sock: socket.socket, max_payload: int = DEFAULT_MAX_PAYLOAD) -> tuple[dict, list[np.ndarray], int]:
    """Receive one frame; returns (header, arrays, bytes_read). Raises ProtocolError on any violation."""
    magic, hl, pl = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if hl > MAX_HEADER or pl > max_payload:
        raise ProtocolError("frame exceeds the configured size limits")
    try:
        header = json.loads(_recv_exact(sock, hl).decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError("header is not valid JSON") from e
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise ProtocolError("header lacks a string 'type'")
    sizes = header.get("sizes", [])
    if not isinstance(sizes, list) or any(not isinstance(s, int) or s < 0 for s in sizes) or sum(sizes) * 4 != pl:
        raise ProtocolError("sizes do not match payload length")
    raw = _recv_exact(sock, pl)
    arrays, off = [], 0
    for s in sizes:
        arrays.append(np.frombuffer(raw, dtype="<f4", count=s, offset=off).astype(np.float32)); off += 4 * s
    return header, arrays, _PREFIX.size + hl + pl
