"""The package must import, and the networked federation must start, WITHOUT the optional HE dependencies (regression test)."""
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent('''
    import sys
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("tenseal", "phe"):
                raise ModuleNotFoundError(name)
    sys.meta_path.insert(0, Block())
    import federated_learning, federated_learning.privacy
    import federated_learning.servers.network_server, federated_learning.clients.network_client
    from federated_learning.privacy.homomorphic import CKKSAggregator
    try:
        CKKSAggregator()
    except ImportError as e:
        assert "TenSEAL" in str(e); print("clear-error")
''')


def test_import_without_he_libraries_and_clear_error_on_use():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    assert "clear-error" in r.stdout
