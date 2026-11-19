# Convenience targets (see README.md).
PY ?= python3

.PHONY: install test test-all prepare

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

test:            ## fast tests, no network
	$(PY) -m pytest -q -m "not network"

test-all:        ## includes tests that download from the Hugging Face hub
	$(PY) -m pytest -q

prepare:         ## tag and corrupt all tasks locally (slow)
	nlicorrupt prepare --task all --style 2021,2022
