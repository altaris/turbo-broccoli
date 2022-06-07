DOCS_PATH 		= docs
SRC_PATH 		= turbo_broccoli
VENV			= ./venv

.ONESHELL:

all: format typecheck lint

.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	pdoc --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	pdoc $(SRC_PATH)

.PHONY: format
format:
	black --line-length 79 --target-version py39 $(SRC_PATH) tests/

.PHONY: lint
lint:
	pylint $(SRC_PATH)

.PHONY: test
test:
	pytest -v
	-rm out/*

.PHONY: typecheck
typecheck:
	mypy -p $(SRC_PATH)
