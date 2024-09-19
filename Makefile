SRC_PATH 	= turbo_broccoli
VENV_PATH	= venv
DOCS_PATH 	= docs

PDOC		= pdoc -d google --math
PYTHON		= python3.11

.ONESHELL:

all: format typecheck lint

.PHONY: clean
clean:
	-rm -r out/test/*

.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PYTHON) -m $(PDOC) --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PYTHON) -m $(PDOC) -h 0.0.0.0 -p 8081 -n $(SRC_PATH)

.PHONY: format
format:
	$(PYTHON) -m isort $(SRC_PATH)
	$(PYTHON) -m black $(SRC_PATH)

.PHONY: lint
lint:
	$(PYTHON) -m pylint $(SRC_PATH)

.PHONY: test
test:
	-mkdir -p out/test
	TB_ARTIFACT_PATH=out/test pytest -v

.PHONY: typecheck
typecheck:
	$(PYTHON) -m mypy -p $(SRC_PATH)
