# =============================================================================
# Makefile — Common development tasks for the AI Project
# =============================================================================
# Usage: make <target>
# Run `make help` to see all available targets.

.DEFAULT_GOAL := help
.PHONY: help install install-dev train evaluate test lint format clean \
        docker-build docker-run jupyter pre-commit type-check

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
PYTHON        := python3
PIP           := pip
SRC_DIR       := src
TEST_DIR      := tests
SCRIPTS_DIR   := scripts
DOCKER_IMAGE  := ai-project
DOCKER_TAG    := latest
JUPYTER_PORT  := 8888
MLFLOW_PORT   := 5000

# Default training config
TRAIN_CONFIG  ?= configs/default_config.yaml
CHECKPOINT    ?= checkpoints/best_model.pt

# ANSI colors for pretty output
BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:  ## Show this help message
	@echo ""
	@echo "$(BOLD)AI Project — Available Make Targets$(RESET)"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install:  ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:  ## Install all dependencies including dev tools
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Dev environment ready$(RESET)"

# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------
train:  ## Run model training (set TRAIN_CONFIG to override config)
	@echo "$(BLUE)Starting training with config: $(TRAIN_CONFIG)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/train.py --config $(TRAIN_CONFIG)

evaluate:  ## Evaluate a checkpoint (set CHECKPOINT to override)
	@echo "$(BLUE)Evaluating checkpoint: $(CHECKPOINT)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py --checkpoint $(CHECKPOINT) --config $(TRAIN_CONFIG)

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
test:  ## Run the full test suite with coverage
	pytest $(TEST_DIR) -v --tb=short

test-fast:  ## Run only fast tests (exclude slow/GPU tests)
	pytest $(TEST_DIR) -v -m "not slow and not gpu" --tb=short

test-cov:  ## Run tests and open HTML coverage report
	pytest $(TEST_DIR) --cov=$(SRC_DIR)/ai_project --cov-report=html
	open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || true

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------
lint:  ## Run all linters (ruff, mypy)
	@echo "$(YELLOW)Running ruff...$(RESET)"
	ruff check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "$(YELLOW)Running mypy...$(RESET)"
	mypy $(SRC_DIR)
	@echo "$(GREEN)✓ All linters passed$(RESET)"

format:  ## Auto-format code with black and ruff
	@echo "$(YELLOW)Formatting with black...$(RESET)"
	black $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "$(YELLOW)Fixing imports with ruff...$(RESET)"
	ruff check --fix $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "$(GREEN)✓ Formatting complete$(RESET)"

type-check:  ## Run mypy type checking
	mypy $(SRC_DIR)

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

# -----------------------------------------------------------------------------
# Jupyter
# -----------------------------------------------------------------------------
jupyter:  ## Start JupyterLab server
	@echo "$(BLUE)Starting JupyterLab on port $(JUPYTER_PORT)...$(RESET)"
	jupyter lab --port=$(JUPYTER_PORT) --no-browser --ip=0.0.0.0

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build:  ## Build the Docker image
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:  ## Run the full stack with docker-compose
	docker compose -f docker/docker-compose.yml up --build

docker-down:  ## Stop all docker-compose services
	docker compose -f docker/docker-compose.yml down

docker-jupyter:  ## Open JupyterLab in the docker-compose stack
	@echo "$(BLUE)JupyterLab available at: http://localhost:$(JUPYTER_PORT)$(RESET)"
	docker compose -f docker/docker-compose.yml up jupyter

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:  ## Remove build artifacts, caches, and temp files
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	@echo "$(GREEN)✓ Clean complete$(RESET)"

clean-all: clean  ## Clean everything including data outputs
	rm -rf outputs/ wandb/ mlruns/ .hydra/
