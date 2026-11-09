PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin
CONFIG ?= configs/default.yaml
DICT := configs/ksl_dictionary.yaml
VOCAB ?= configs/vocabulary_v2.txt
RUN ?=
NAME ?=

.PHONY: setup models audit extract build baseline train evaluate export test dictionary references catalogue calibrate pack golden studio-import

setup:        # data pipeline only: audit, extract, references
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

setup-train:  # adds PyTorch, ONNX and scikit-learn for baseline/train/evaluate/export
	$(BIN)/pip install -e ".[dev,train]"

models:
	./scripts/download_models.sh

audit:
	$(BIN)/ashiria audit --config $(CONFIG)

extract:
	$(BIN)/ashiria extract --config $(CONFIG)

build:
	$(BIN)/ashiria build --config $(CONFIG)

baseline:
	$(BIN)/ashiria baseline --config $(CONFIG)

train:
	$(BIN)/ashiria train --config $(CONFIG) $(if $(NAME),--name $(NAME))

evaluate:
	$(BIN)/ashiria evaluate --config $(CONFIG) --run $(RUN)

export:
	$(BIN)/ashiria export --config $(CONFIG) --run $(RUN) $(if $(NAME),--name $(NAME))

test:
	$(BIN)/pytest -q

# KSL Dictionary videos: import labels, audit, extract landmarks, build reference templates.
dictionary:
	$(BIN)/ashiria import-dictionary --config $(DICT)
	$(BIN)/ashiria audit --config $(DICT)
	$(BIN)/ashiria extract --config $(DICT)
	$(BIN)/ashiria references --config $(DICT) --vocab $(VOCAB)

references:
	$(BIN)/ashiria references --config $(CONFIG) --vocab $(VOCAB)

# Content for the headset: catalogue (committed) -> calibration -> content pack (content/packs/).
catalogue:
	$(BIN)/ashiria catalogue --vocab $(VOCAB)

calibrate:
	$(BIN)/ashiria calibrate

pack:
	$(BIN)/ashiria pack

STUDIO ?= data/studio
studio-import:  # takes pulled from the headset into $(STUDIO)
	$(BIN)/ashiria import-studio $(STUDIO)

# Cross-language test vectors for the C# engine (spec/golden/).
golden:
	$(BIN)/ashiria golden
