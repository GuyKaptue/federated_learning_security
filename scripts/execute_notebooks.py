"""Executes every notebook (nbclient) in place (notebooks/sprint_tasks/<story>/) and reports pass/fail + 18-section check."""
import json
import re
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

R = Path(__file__).resolve().parents[1]
rows = []
for nbp in sorted((R/"notebooks/sprint_tasks").glob("*/*.ipynb")):
    nb = nbf.read(nbp, as_version=4)
    heads = [re.match(r"## (\d+)\.", c.source).group(1) for c in nb.cells if c.cell_type == "markdown" and re.match(r"## (\d+)\.", c.source)]
    t = time.time(); status, err = "PASS", ""
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(R)}}).execute()
    except Exception as e:
        status, err = "FAIL", str(e)[-600:]
    nbf.write(nb, nbp)   # executed in place: outputs are stored in the notebook
    errs = sum(1 for c in nb.cells if c.cell_type=="code" for o in c.get("outputs", []) if o.output_type=="error")
    rows.append(dict(notebook=nbp.name, story=nbp.parent.name, status=status, sections_ok=heads == [str(i) for i in range(1, 19)], n_sections=len(heads), error_outputs=errs, seconds=round(time.time()-t,1), error=err))
    print(rows[-1]["notebook"], status, rows[-1]["seconds"], flush=True)

json.dump(rows, open(R/"reports/sprint_notebook_execution_report.json","w"), indent=1)
print("passed", sum(r["status"]=="PASS" for r in rows), "of", len(rows))
