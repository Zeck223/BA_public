PYTHON ?= python3
VENV ?= .venv-BA
VENV_PYTHON := $(VENV)/bin/python
LMS := $(VENV)/bin/lms
CONFIG ?=
MODEL ?=
TEST_SET ?=
DATA ?= grids
SEED ?=

.PHONY: setup generator simulation plot freq train evaluate eval

setup:
	$(PYTHON) -m venv "$(VENV)"
	$(VENV_PYTHON) -m pip install -e .
	@if [ ! -e configs/simulation.yaml ] && [ ! -L configs/simulation.yaml ]; then \
		cp configs/simulation.yaml.example configs/simulation.yaml; \
	fi
	@if [ ! -e configs/neural_network.yaml ] && [ ! -L configs/neural_network.yaml ]; then \
		cp configs/neural_network.yaml.example configs/neural_network.yaml; \
	fi

generator:
	$(LMS) generator

simulation:
	$(LMS) simulation --config "$(CONFIG)" $(if $(SEED),--seed "$(SEED)",)

plot:
	$(LMS) plot --config "$(CONFIG)" --type mosaic --data $(DATA)

freq:
	$(LMS) freq --config "$(CONFIG)" --all

train:
	$(LMS) train --config "$(CONFIG)" --model-save-name "$(MODEL)" --data $(DATA)

evaluate:
	$(LMS) evaluate --config "$(CONFIG)" --model-save-name "$(MODEL)" --test-set-name "$(TEST_SET)" --data $(DATA)

eval: evaluate
