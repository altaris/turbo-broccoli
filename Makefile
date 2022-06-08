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
	@-mkdir -p out/test
	TB_ARTIFACT_PATH="out/test" pytest -v

.PHONY: test-clean
test-clean:
	-rm out/test/*

.PHONY: typecheck
typecheck:
	mypy -p $(SRC_PATH)
