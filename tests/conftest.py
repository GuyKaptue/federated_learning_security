import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("RS_QUICK", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
