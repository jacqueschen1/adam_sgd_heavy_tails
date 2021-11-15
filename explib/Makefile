.PHONY: help
.PHONY: install install-dev

.DEFAULT: help
help:
	@echo "install        Install the library and dependencies"
	@echo "install-dev    Install in editable mode with dev. dependencies"

install:
	@pip install -r requirements.txt

install-dev:
	@pip install -e .
