.PHONY: install lint test smoke reproduce clean

install:
	pip install -e ".[dev]"

lint:
	ruff check neuroaugment tests

test:
	pytest

smoke:
	python tests/integration/smoke_pretrain.py
	python tests/integration/federated_round_trip.py

reproduce:
	./repro/one_hour_repro.sh

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
