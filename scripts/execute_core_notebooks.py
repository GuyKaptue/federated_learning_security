"""Executes the 14 numbered notebooks in place (outputs stored in the .ipynb) and writes reports/notebook_execution_report.json."""
import json
import os
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

R = Path(__file__).resolve().parents[1]
os.environ["FL_ROOT"] = str(R)
only = sys.argv[1:]
rows = []
for p in sorted((R / "notebooks").glob("[0-9][0-9]_*.ipynb")):
    if only and not any(o in p.name for o in only):
        continue
    nb = nbf.read(p, as_version=4); t = time.time(); status, err = "PASS", ""
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(R)}}).execute()
    except Exception as e:
        status, err = "FAIL", str(e)[-800:]
    nbf.write(nb, p)
    rows.append(dict(notebook=p.name, status=status, seconds=round(time.time() - t, 1), error=err)); print(rows[-1]["notebook"], status, rows[-1]["seconds"], flush=True)
(R / "reports").mkdir(exist_ok=True)
rep = R / "reports" / "notebook_execution_report.json"
old = json.loads(rep.read_text()) if rep.exists() and only else []
json.dump([r for r in old if r["notebook"] not in {x["notebook"] for x in rows}] + rows, open(rep, "w"), indent=1)
print("passed", sum(r["status"] == "PASS" for r in rows), "of", len(rows))
