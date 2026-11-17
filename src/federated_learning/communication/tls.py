"""Mutual-TLS helpers: a throw-away demo PKI and strict SSL contexts.

`generate_pki` creates a private CA, a server certificate (SAN: localhost, 127.0.0.1) and one client certificate per client id
(CN = "client-<id>"). It exists for local tests and demos; in a real deployment use certificates from your own PKI and keep keys secret.
Contexts require TLS >= 1.2 and certificates on BOTH sides, so a peer without a certificate signed by the CA cannot complete a handshake.
"""
from __future__ import annotations

import datetime
import ipaddress
import os
import ssl
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


@dataclass(frozen=True)
class PKI:
    ca_cert: Path
    server_cert: Path
    server_key: Path
    client_certs: dict[int, tuple[Path, Path]]   # cid -> (cert, key)


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _write_key(path: Path, key) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


def _issue(cn, key, ca_cert, ca_key, days, san=None, client=False):
    now = datetime.datetime.now(datetime.timezone.utc)
    b = (x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(ca_cert.subject).public_key(key.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
         .not_valid_after(now + datetime.timedelta(days=days))
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True))
    if san:
        b = b.add_extension(x509.SubjectAlternativeName(san), critical=False)
    b = b.add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH if client else x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    return b.sign(ca_key, hashes.SHA256())


def generate_pki(directory: Path, client_ids: list[int], days: int = 7) -> PKI:
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(_name("fl-demo-ca")).issuer_name(_name("fl-demo-ca")).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=days))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False, key_encipherment=False,
                                       data_encipherment=False, key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    (d / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    sk = ec.generate_private_key(ec.SECP256R1())
    sc = _issue("fl-server", sk, ca, ca_key, days, san=[x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))])
    (d / "server.pem").write_bytes(sc.public_bytes(serialization.Encoding.PEM)); _write_key(d / "server.key", sk)
    clients = {}
    for cid in client_ids:
        k = ec.generate_private_key(ec.SECP256R1()); c = _issue(f"client-{cid}", k, ca, ca_key, days, client=True)
        cp, kp = d / f"client_{cid}.pem", d / f"client_{cid}.key"
        cp.write_bytes(c.public_bytes(serialization.Encoding.PEM)); _write_key(kp, k); clients[cid] = (cp, kp)
    return PKI(d / "ca.pem", d / "server.pem", d / "server.key", clients)


def server_context(ca_cert: Path, cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED               # clients MUST present a certificate signed by our CA
    ctx.load_verify_locations(cafile=str(ca_cert)); ctx.load_cert_chain(str(cert), str(key))
    return ctx


def client_context(ca_cert: Path, cert: Path, key: Path, server_hostname_check: bool = True) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED; ctx.check_hostname = server_hostname_check
    ctx.load_verify_locations(cafile=str(ca_cert)); ctx.load_cert_chain(str(cert), str(key))
    return ctx


def peer_common_name(ssock: ssl.SSLSocket) -> str | None:
    cert = ssock.getpeercert()
    for rdn in (cert or {}).get("subject", ()):
        for k, v in rdn:
            if k == "commonName":
                return v
    return None
