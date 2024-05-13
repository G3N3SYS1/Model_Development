# Makefile for python code
# 
# > make help
#
# The following commands can be used.
#
# init:  sets up environment and installs requirements
# install:  Installs development requirments
# format:  Formats the code with autopep8
# lint:  Runs ruff on src, exit if critical rules are broken
# clean:  Remove build and cache files
# env:  Source venv and environment files for testing
# leave:  Cleanup and deactivate venv
# test:  Run pytest
# run:  Executes the logic

VENV_PATH='env/bin/activate'
ENVIRONMENT_VARIABLE_FILE='.env'

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

help:
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)


init: ## sets up environment and installs requirements
init:
	pip install -r requirements.txt

install: ## Installs development requirments
install:
	python -m pip install --upgrade pip
	# Used for linting
	pip install ruff
	# Used for testing
	pip install pytest

lint: ## Runs ruff, exit if critical rules are broken
lint:
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	ruff check --exit-zero
	# stop the build if there are Python syntax errors or undefined names
	ruff check --statistics

format: ## Runs ruff in formatter mode
format:
	ruff format

package: ## Create package in dist
package: clean
	python3 setup/setup.py sdist bdist_wheel

clean: ## Remove build and cache files
clean:
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	rm -rf .pytest_cache
	# Remove all pycache
	find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

env: ## Source venv and environment files for testing
env:
	python3 -m venv env
	source $(VENV_PATH)
	source $(ENVIRONMENT_VARIABLE_FILE)

leave: ## Cleanup and deactivate venv
leave: clean
	deactivate

test: ## Run pytest
test:
	pytest . -p no:logging -p no:warnings
