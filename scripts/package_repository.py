"""Repeatable packaging of the repository into build/federated_learning_security.zip with verification.

Steps: clean caches -> stage a filtered copy -> secret scan -> naming/structure checks -> manifest + SHA256SUMS -> zip ->
extract the ZIP into a fresh directory and run ruff, pytest, `python -m build` and notebook checks THERE -> write the validation report.
Raw datasets, model weights, logs, MLflow stores and the unshipped IEEEtran class are excluded (see .gitignore).
Run: python scripts/package_repository.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[1]
NAME = "federated_learning_security"
BUILD = R / "build"
STAGE = BUILD / "stage" / NAME
ZIP = BUILD / f"{NAME}.zip"
REPORT = BUILD / "zip_delivery_validation.md"

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "build", "dist", ".venv", "venv", "mlruns", "vendor", ".git"}
EXCLUDE_SUFFIX = (".pyc", ".aux", ".bbl", ".blg", ".log", ".out", ".db")
REQUIRED = ["src", "notebooks", "configs", "data/raw", "data/processed", "data/synthetic", "data/external", "models", "experiments", "reports",
            "docs", "tests/unit", "tests/integration", "tests/end_to_end", "scripts", "logs", ".github/workflows", "README.md", "LICENSE",
            "requirements.txt", "pyproject.toml", ".gitignore", "setup.py", "Makefile"]
SECRET = re.compile(r"(AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|xox[bp]-[A-Za-z0-9-]{20,})")
checks: list[tuple[str, bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    checks.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def keep(rel: Path) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_DIRS for p in parts[:-1]) or rel.name.endswith(EXCLUDE_SUFFIX) or rel.name.endswith(".egg-info"):
        return False
    if any(p.endswith(".egg-info") for p in parts):
        return False
    if parts[0] == "data" and len(parts) > 2 and parts[1] in ("raw", "processed", "external") and rel.name not in ("README.md", ".gitkeep"):
        return False
    if parts[0] == "data" and parts[1:2] == ("synthetic",) and parts[2:3] == ("generated",) and rel.name != ".gitkeep":
        return False
    if parts[0] in ("models", "logs") and rel.name not in ("README.md", ".gitkeep"):
        return False
    return True


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd, env=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)  # nosec B603


def stage() -> list[Path]:
    if BUILD.exists():
        for child in BUILD.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    STAGE.mkdir(parents=True)
    files = []
    for p in sorted(R.rglob("*")):
        rel = p.relative_to(R)
        if rel.parts[0] == "build" or p.is_dir() or not keep(rel):
            continue
        dst = STAGE / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst); files.append(rel)
    for d in ("data/raw", "data/processed", "data/external", "models", "logs", "experiments", "reports"):
        (STAGE / d).mkdir(parents=True, exist_ok=True)
    return files


def static_checks(files: list[Path]) -> None:
    for r in REQUIRED:
        ok(f"required path {r}", (STAGE / r).exists())
    bad_dirs = sorted({str(d.relative_to(STAGE)) for d in STAGE.rglob("*") if d.is_dir() and d.name != d.name.lower()})
    ok("all folder names lower-case", not bad_dirs, ", ".join(bad_dirs[:5]))
    bad_py = [str(p.relative_to(STAGE)) for p in STAGE.rglob("*.py") if not re.fullmatch(r"[a-z0-9_]+\.py", p.name)]
    ok("all Python files snake_case", not bad_py, ", ".join(bad_py[:5]))
    bad_nb = [p.name for p in STAGE.rglob("*.ipynb") if not re.fullmatch(r"[a-z0-9_]+\.ipynb", p.name)]
    ok("all notebooks snake_case", not bad_nb, ", ".join(bad_nb[:5]))
    ok("no leftover research_sprint package/dirs", not (STAGE / "src/research_sprint").exists() and not list(STAGE.rglob("research_sprint")))
    raw = [p for p in (STAGE / "data/raw").rglob("*") if p.is_file() and p.name != "README.md"]
    ok("no raw dataset file shipped", not raw, f"{len(raw)} files")
    hits = []
    for p in STAGE.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".tex", ".cfg") and p.stat().st_size < 5_000_000:
            m = SECRET.search(p.read_text(errors="ignore"))
            if m:
                hits.append(f"{p.relative_to(STAGE)}:{m.group(0)[:12]}")
    ok("secret scan (key/token patterns; author email is intentional project metadata)", not hits, ", ".join(hits[:5]))
    meta = (STAGE / "pyproject.toml").read_text()
    ok("author metadata", "Guy M. Kaptue T." in meta and "guykaptue24@gmail.com" in meta and "github.com/GuyKaptue" in meta)
    nbs = sorted((STAGE / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    ok("14 numbered notebooks present", len(nbs) == 14, f"{len(nbs)}")
    err = noout = 0
    for p in list((STAGE / "notebooks").rglob("*.ipynb")):
        nb = json.loads(p.read_text()); codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
        err += sum(1 for c in codes for o in c.get("outputs", []) if o.get("output_type") == "error")
        noout += int(not any(c.get("outputs") for c in codes))
    ok("all notebooks executed (outputs stored, 0 error outputs)", err == 0 and noout == 0, f"errors={err}, notebooks without outputs={noout}")
    figs = list((STAGE / "reports/figures").glob("*.png"))
    ok("publication figures present", len(figs) >= 14, f"{len(figs)} PNG")


def manifest(files: list[Path]) -> None:
    rows = {str(p.relative_to(STAGE)): sha(p) for p in sorted(STAGE.rglob("*")) if p.is_file()}
    (STAGE / "artifact_manifest.json").write_text(json.dumps({"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                                              "n_files": len(rows), "sha256": rows}, indent=1))
    (STAGE / "SHA256SUMS").write_text("".join(f"{h}  {k}\n" for k, h in rows.items()))


def make_zip() -> None:
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, f"{NAME}/{p.relative_to(STAGE)}")


def verify_extracted() -> None:
    with tempfile.TemporaryDirectory(prefix="fl_zip_check_") as tmp:
        with zipfile.ZipFile(ZIP) as z:
            ok("zip integrity (testzip)", z.testzip() is None, f"{len(z.namelist())} members")
            z.extractall(tmp)
        root = Path(tmp) / NAME
        bad = [k for k in (l.split("  ", 1) for l in (root / "SHA256SUMS").read_text().splitlines()) if sha(root / k[1]) != k[0]]
        ok("SHA256SUMS verify after extraction", not bad, f"{len(bad)} mismatches")
        env = {**os.environ, "PYTHONPATH": str(root / "src"), "RS_QUICK": "1", "FL_ROOT": str(root)}
        r = run([sys.executable, "-m", "ruff", "check", "."], root); ok("ruff check (extracted copy)", r.returncode == 0, r.stdout[-200:])
        r = run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], root, env)
        ok("pytest from extracted copy", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])
        r = run([sys.executable, "-m", "build", "--outdir", str(Path(tmp) / "dist")], root); ok("python -m build (sdist + wheel)", r.returncode == 0, r.stderr[-200:])
        r = run([sys.executable, "-m", "twine", "check", *map(str, (Path(tmp) / "dist").glob("*"))], root); ok("twine check", r.returncode == 0, r.stdout[-200:])
        r = run([sys.executable, "-m", "bandit", "-q", "-r", "src", "-c", "pyproject.toml"], root); ok("bandit (src)", r.returncode == 0, "")
        r = run([sys.executable, "-c", "import federated_learning as f; print(f.__version__)"], root, env); ok("package imports", r.returncode == 0, r.stdout.strip())
        r = run([sys.executable, "-m", "federated_learning.synthetic_data.pipelines", "--seed", "5", "--n-samples", "800", "--out", str(Path(tmp) / "syn")], root, env)
        ok("synthetic pipeline CLI runs from the extracted copy", r.returncode == 0 and "true" in r.stdout.lower(), r.stdout.strip()[:120])


def main() -> None:
    files = stage()
    static_checks(files)
    manifest(files)
    make_zip()
    verify_extracted()
    passed = sum(c[1] for c in checks)
    lines = [f"# ZIP delivery validation — {NAME}.zip", "", f"Generated {datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds')}Z by scripts/package_repository.py.",
             f"ZIP SHA-256: `{sha(ZIP)}` · size {ZIP.stat().st_size} bytes", "", f"**{passed}/{len(checks)} checks passed.**", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if c else 'FAIL'} | {d.replace('|', '/').replace(chr(10), ' ')} |" for n, c, d in checks]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\n{passed}/{len(checks)} checks passed; ZIP = {ZIP}")
    sys.exit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
