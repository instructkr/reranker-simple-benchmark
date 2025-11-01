.PHONY: init format check requirements single-run multi-run

TYPE ?= cross-encoder
MODEL_NAME ?= sigridjineth/ko-reranker-v1.1
MODEL_CLASS ?= huggingface
DATATYPE_NAME ?= AutoRAG
NPROC ?= 8

init:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv
	. .venv/bin/activate
	uv sync

format:
	uvx ruff format

check:
	uvx ruff check --fix

requirements:
	uv export -o requirements.txt

run:
	@echo "Running with uv..."
	uv run ./retriever-simple-benchmark/main.py \
	  --type $(TYPE) \
	  --model_name "$(MODEL_NAME)" \
	  --model_class $(MODEL_CLASS) \
	  --datatype_name $(DATATYPE_NAME)

torchrun:
	@echo "Running with torchrun..."
	. .venv/bin/activate

	torchrun --nproc_per_node=$(NPROC) ./retriever-simple-benchmark/main.py \
	  --type $(TYPE) \
	  --model_name "$(MODEL_NAME)" \
	  --model_class $(MODEL_CLASS) \
	  --datatype_name $(DATATYPE_NAME)
