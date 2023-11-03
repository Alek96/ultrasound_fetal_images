.ONESHELL:

CONDA_ENV_NAME := ultrasound_fetal_images_env
SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

## Show help
help:
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

## Ensure that conda environment is enabled
ensure_conda_env:
	@if [ $(CONDA_DEFAULT_ENV) != $(CONDA_ENV_NAME) ]; then \
		echo "You don't have a conda environment enabled."; \
		echo "Please enable the conda environment first!"; \
		echo "please write 'conda activate $(CONDA_ENV_NAME)'"; \
		exit 1; \
	fi

## Install project
install:
	@if [ $(CONDA_DEFAULT_ENV) != $(CONDA_ENV_NAME) ]; then \
		# Create Conda environment based on generated lock file
		conda create --name $(CONDA_ENV_NAME) --file conda-linux-64.lock; \
		# Active conda environment
		$(CONDA_ACTIVATE) $(CONDA_ENV_NAME); \
		echo "conda environment: $(CONDA_ENV_NAME)"; \
		poetry install; \
	else \
		# Update Conda packages based on generated lock file
		mamba update --file conda-linux-64.lock; \
		# Update Poetry packages based on poetry.lock
		poetry install; \
	fi


## Update project
update: ensure_conda_env
	# Re-generate Conda lock file(s) based on environment.yml
	conda-lock -k explicit --conda mamba
	# Update Conda packages based on re-generated lock file
	mamba update --file conda-linux-64.lock
	# Update Poetry packages and re-generate poetry.lock
	poetry update
	# Update pre-commit hook versions in .pre-commit-config.yaml
	pre-commit autoupdate
	# Print packages info
	poetry show --top-level

## Clean autogenerated files
clean:
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

## Clean logs
clean-logs:
	rm -rf logs/**
	rm -rf lightning_logs/**
	rm -rf tests/lightning_logs/**

## Run pre-commit hooks
format: ensure_conda_env
	pre-commit run -a

## Merge changes from main branch to your current branch
sync:
	git pull
	git pull origin main

## Run not slow tests
test: ensure_conda_env
	pytest -k "not slow"

## Run all tests
test-full: ensure_conda_env
	pytest

## Train the model
train: ensure_conda_env
	python src/train.py $(args)

## Run jupyter notebook
notebook: ensure_conda_env
	jupyter notebook --notebook-dir=./notebooks --no-browser

## Run jupyter notebook
lab: ensure_conda_env
	jupyter lab --notebook-dir=./notebooks --no-browser
