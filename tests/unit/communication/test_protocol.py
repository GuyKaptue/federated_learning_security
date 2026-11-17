import socket
import struct

import numpy as np
import pytest

from federated_learning.communication.protocol import MAGIC, ProtocolError, recv_message, send_message


@pytest.fixture
def pair():
    a, b = socket.socketpair(); yield a, b; a.close(); b.close()


def test_roundtrip(pair):
    a, b = pair; x = np.arange(5, dtype=np.float32); y = np.ones(3, np.float32)
    n = send_message(a, {"type": "t", "k": 1}, [x, y]); h, arr, m = recv_message(b)
    assert h["type"] == "t" and h["k"] == 1 and n == m and np.array_equal(arr[0], x) and np.array_equal(arr[1], y)


def _feed(raw: bytes, **kw):
    a, b = socket.socketpair()
    try:
        a.sendall(raw)
        return recv_message(b, **kw)
    finally:
        a.close(); b.close()


def test_rejects_bad_magic_oversize_and_bad_sizes():
    with pytest.raises(ProtocolError, match="magic"):
        _feed(b"XXXX" + struct.pack(">II", 2, 0) + b"{}")
    with pytest.raises(ProtocolError, match="limits"):
        _feed(MAGIC + struct.pack(">II", 2, 10**9) + b"{}", max_payload=1000)
    hdr = b'{"type":"x","sizes":[5]}'
    with pytest.raises(ProtocolError, match="sizes"):
        _feed(MAGIC + struct.pack(">II", len(hdr), 4) + hdr + b"\0\0\0\0")


def test_truncated_and_non_json(pair):
    a, b = pair
    a.sendall(MAGIC + struct.pack(">II", 50, 0) + b"{}"); a.shutdown(socket.SHUT_WR)
    with pytest.raises(ProtocolError, match="closed"):
        recv_message(b)
    c, d = socket.socketpair()
    c.sendall(MAGIC + struct.pack(">II", 4, 0) + b"\xff\xfe\xfd\xfc")
    with pytest.raises(ProtocolError, match="JSON"):
        recv_message(d)
    c.close(); d.close()
