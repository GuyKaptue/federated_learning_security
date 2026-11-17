.PHONY: network-demo install install-dev lint test test-quick security build data synthetic notebooks docs clean zip e3 experiments
PY ?= python
export PYTHONPATH := src

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps
install-dev: install
	$(PY) -m pip install -r requirements-dev.txt
lint:
	ruff check .
test:
	$(PY) -m pytest tests
test-quick:
	RS_QUICK=1 $(PY) -m pytest tests/unit
security:
	bandit -q -r src -c pyproject.toml && pip-audit -r requirements.txt
build:
	$(PY) -m build && twine check dist/*
data:
	$(PY) -c "from federated_learning.preprocessing import download_all; print(download_all())"
synthetic:
	$(PY) -m federated_learning.synthetic_data.pipelines --seed 0
notebooks:
	$(PY) scripts/generate_core_notebooks.py && $(PY) scripts/execute_core_notebooks.py
docs:
	$(PY) scripts/generate_dataset_docs.py
network-demo:
	$(PY) scripts/run_network_demo.py --clients 5 --rounds 10
experiments:
	$(PY) -m federated_learning.pipelines.experiments.run_all --task all
e3:
	$(PY) -m federated_learning.pipelines.experiments.run_all --task FL-E3-BASE
	$(PY) -m federated_learning.pipelines.experiments.run_all --task FL-E3-FGVAL
zip:
	$(PY) scripts/package_repository.py
clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info; find . -name __pycache__ -type d -prune -exec rm -rf {} +
