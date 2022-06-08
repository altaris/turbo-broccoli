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

# Environment variables

Some behaviors of Turbo Broccoli can be tweaked by setting specific environment
variables. If you want to modify these parameters programatically, do not do so
by modifying `os.environ`. Rather, use the methods of
`turbo_broccoli.environment`.

* `TB_ARTIFACT_PATH` (default: `./`): During serialization, Turbo Broccoli may
  create artifacts to which the JSON object will point to. The artifacts will
  be stored in `TB_ARTIFACT_PATH`. For example, if `arr` is a big numpy array,

    ```py
    obj = {"an_array": arr}
    json.dumps(obj, cls=tb.TurboBroccoliEncoder)
    ```

  will generate the following string (modulo indentation and id)

    ```
    {
        "an_array": {
            "__numpy__": {
                "__type__": "ndarray",
                "__version__": 2,
                "id": "70692d08-c4cf-4231-b3f0-0969ea552d5a"
            }
        }
    }
    ```

  and a `70692d08-c4cf-4231-b3f0-0969ea552d5a.npy` file has been created in
  `TB_ARTIFACT_PATH`.

* `TB_KERAS_FORMAT` (default: `tf`, valid values are `json`, `h5`, and `tf`):
  The serialization format for keras models. If `h5` or `tf` is used, an
  artifact following said format will be created in `TB_ARTIFACT_PATH`. If
  `json` is used, the model will be contained in the JSON document (anthough
  the weights may be in artifacts if they are too large).

* `TB_NUMPY_MAX_NBYTES` (default: `8000`): The maximum byte size of an numpy
  array beyond which serialization will produce an artifact instead of storing
  the array in the JSON document. 8000 bytes should be enough for an array of
  1000 `float64`s to be stored in-document.

## Supported types

* `bytes`
* `numpy.number`
* `numpy.ndarray` with numerical dtype
* `tensorflow.Tensor` with numerical dtype, but not `tensorflow.RaggedTensor`

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
