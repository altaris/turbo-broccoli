DOCS_PATH 		= docs
SRC_PATH 		= turbo_broccoli
TST_PATH 		= tests
VENV			= ./venv
PDOC			= pdoc -d google

ISORT			= isort --line-length 79 --python-version 310 --multi-line VERTICAL_HANGING_INDENT
BLACK			= black --line-length 79 --target-version py310

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
	$(ISORT) $(SRC_PATH)
	$(ISORT) $(TST_PATH)
	$(BLACK) $(SRC_PATH) $(TST_PATH)

.PHONY: lint
lint:
	pylint $(TST_PATH)
	pylint $(SRC_PATH)

.PHONY: test
test:
	-mkdir -p out/test
	pytest -v

.PHONY: typecheck
typecheck:
	mypy -p $(SRC_PATH) --check-untyped-defs
