Turbo Broccoli
==============

![Python 3](https://img.shields.io/badge/python-3-blue)
![License](https://img.shields.io/github/license/altaris/turbo-broccoli)
[![Code style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)
![hehe](https://img.shields.io/badge/project%20name-github-pink)

JSON (de)serialization extensions aimed at `numpy` and `tensorflow` objects.

# Usage

```py
import json
import numpy as np
import turbo_broccoli as tb

obj = {
    "an_array": np.ndarray([[1, 2], [3, 4]], dtype="float32")
}
json.dumps(obj, cls=tb.TurboBroccoliEncoder)
```
produces the following string (modulo indentation):
```
{
    "an_array": {
        "__numpy__": {
            "__type__": "ndarray",
            "__version__": 1,
            "data": "AACAPwAAAEAAAEBAAACAQA==",
            "dtype": "<f4",
            "shape": [2, 2]
        }
    }
}
```

For deserialization, simply use
```py
json.loads(json_string, cls=tb.TurboBroccoliDecoder)
```

## Supported types

* `numpy.number`
* `numpy.ndarray`

# Contributing

## Dependencies

* `python3.9` or newer;
* `requirements.txt` for runtime dependencies;
* `requirements.dev.txt` for development dependencies.
* `make` (optional);

Simply run
```sh
virtualenv venv -p python3.9
. ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

## Documentation

Simply run
```sh
make docs
```
This will generate the HTML doc of the project, and the index file should be at
`docs/index.html`. To have it directly in your browser, run
```sh
make docs-browser
```

## Code quality

Don't forget to run
```sh
make
```
to format the code following [black](https://pypi.org/project/black/),
typecheck it using [mypy](http://mypy-lang.org/), and check it against coding
standards using [pylint](https://pylint.org/).

## Unit tests

Run
```sh
make test
```
to have [pytest](https://docs.pytest.org/) run the unit tests in `tests/`.

# Credits

This project takes inspiration from
[Crimson-Crow/json-numpy](https://github.com/Crimson-Crow/json-numpy).
