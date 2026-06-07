SHELL := /bin/bash
.PHONY: install run-mlp run-cnn clean help
.DEFAULT_GOAL := help

help:
	@printf "$(BOLD)Comandos disponiveis:$(NC)\n"
	@printf "  $(CYAN)make install$(NC)  - instala o uv, dependencias e hooks de pre-commit\n"
	@printf "  $(CYAN)make run-mlp$(NC)  - roda o treino+teste do MLP pros 5 datasets\n"
	@printf "  $(CYAN)make run-cnn$(NC)  - roda o treino+teste da CNN (Fashion MNIST)\n"
	@printf "  $(CYAN)make clean$(NC)    - apaga os arquivos gerados em saidas/\n"

install:
	@if ! command -v uv >/dev/null 2>&1; then \
		$(call run_with_spinner,Installing uv,curl -LsSf https://astral.sh/uv/install.sh | sh); \
	fi
	@$(call run_with_spinner,Installing dependencies,uv sync --upgrade)
	@$(call run_with_spinner,Setting up pre-commit hooks,uv run pre-commit install)
	@printf "$(BOLD)$(GREEN)  ✓ Done! Environment ready.$(NC)\n"
	@printf "\n$(DIM)  Next steps:$(NC)\n"
	@printf "$(DIM)    make run-mlp$(NC)\n"
	@printf "$(DIM)    make run-cnn$(NC)\n"

run-mlp:
	@printf "$(BOLD)$(CYAN)Rodando MLP (OR, AND, XOR, CARACTERES_REDUZIDO, CARACTERES_COMPLETO)...$(NC)\n\n"
	@uv run python src/mlp/main.py

run-cnn:
	@printf "$(BOLD)$(CYAN)Rodando CNN (Fashion MNIST)...$(NC)\n\n"
	@uv run python src/cnn/main.py

clean:
	@rm -rf saidas/*/
	@printf "$(GREEN)  ✓ saidas/ limpo$(NC)\n"
