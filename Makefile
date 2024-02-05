DOCS_PATH 		= docs
SRC_PATH 		= turbo_broccoli
VENV			= ./venv
PDOC			= pdoc -d google

.ONESHELL:

all: format typecheck lint

.PHONY: test-clean
clean:
	-rm -r out/test/*
.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PDOC) --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PDOC) $(SRC_PATH)

.PHONY: format
format:
	black --line-length 79 --target-version py310 $(SRC_PATH) tests/

.PHONY: lint
lint:
	pylint $(SRC_PATH)

.PHONY: test
test:
	-mkdir -p out/test
	pytest -v

.PHONY: typecheck
typecheck:
	mypy -p $(SRC_PATH) --check-untyped-defs
