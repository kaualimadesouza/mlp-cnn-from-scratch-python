SHELL := /bin/bash
.PHONY: install run-mlp run-cnn clean help
.DEFAULT_GOAL := help

help:
	@printf "Comandos disponiveis:\n"
	@printf "  make install  - instala o uv, dependencias e hooks de pre-commit\n"
	@printf "  make run-mlp  - roda o treino+teste do MLP pros 5 datasets\n"
	@printf "  make run-cnn  - roda o treino+teste da CNN (Fashion MNIST)\n"
	@printf "  make clean    - apaga os arquivos gerados em saidas/\n"

install:
	@if ! command -v uv >/dev/null 2>&1; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv sync --upgrade
	@uv run pre-commit install
	@printf "  Done! Environment ready.\n"
	@printf "\n  Next steps:\n"
	@printf "    make run-mlp\n"
	@printf "    make run-cnn\n"

run-mlp:
	@printf "Rodando MLP (OR, AND, XOR, CARACTERES_REDUZIDO, CARACTERES_COMPLETO)...\n\n"
	@uv run python src/mlp/main.py

run-cnn:
	@printf "Rodando CNN (Fashion MNIST)...\n\n"
	@uv run python src/cnn/main.py

clean:
	@rm -rf saidas/*/
	@printf "  saidas/ limpo\n"
