SRC_PATH 	= turbo_broccoli
VENV_PATH	= venv
DOCS_PATH 	= docs

PDOC		= pdoc -d google --math

.ONESHELL:

all: format typecheck lint

.PHONY: clean
clean:
	-rm -r out/test/*

.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	uv run $(PDOC) --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	uv run $(PDOC) -p 8081 -n $(SRC_PATH)

.PHONY: format
format:
	uvx ruff check --select I --fix
	uvx ruff format

.PHONY: lint
lint:
	uvx ruff check

.PHONY: test
test:
	-mkdir -p out/test
	TB_ARTIFACT_PATH=out/test uv run pytest -v tests

.PHONY: typecheck
typecheck:
	uv run mypy -p $(SRC_PATH)
