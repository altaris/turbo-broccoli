# Turbo Broccoli 🥦

[![PyPI](https://img.shields.io/pypi/v/turbo-broccoli)](https://pypi.org/project/turbo-broccoli/)
![License](https://img.shields.io/github/license/altaris/turbo-broccoli)
[![Code
style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)
![hehe](https://img.shields.io/badge/project%20name%20by-github-pink)
[![Documentation](https://badgen.net/badge/documentation/here/green)](https://https://altaris.github.io/turbo-broccoli/turbo_broccoli.html)

JSON (de)serialization extensions, originally aimed at `numpy` and `tensorflow`
objects, but now supporting a wide range of objects.

# Installation

```sh
pip install turbo-broccoli
```

# Usage

```py
import json
import numpy as np
import turbo_broccoli as tb

obj = {
    "an_array": np.array([[1, 2], [3, 4]], dtype="float32")
}
json.dumps(obj, cls=tb.TurboBroccoliEncoder)

# or even simpler:
tb.to_json(obj)
```

produces the following string (modulo indentation):

```json
{
  "an_array": {
    "__type__": "numpy.ndarray",
    "__version__": 4,
    "data": {
      "__type__": "bytes",
      "__version__": 2,
      "data": "QAAAAAAAAAB7ImRhd..."
    }
  }
}

```

For deserialization, simply use

```py
json.loads(json_string, cls=tb.TurboBroccoliDecoder)

# or even simpler:
tb.from_json(json_string)
```

## Supported types

### Basic types

- [`bytes`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/bytes.html#to_json)

- [`dict` with non `str` keys](https://altaris.github.io/turbo-broccoli/turbo_broccoli/dict.html#to_json)

- [Collections](https://altaris.github.io/turbo-broccoli/turbo_broccoli/collections.html#to_json):
  `collections.deque`, `collections.namedtuple`

- [Dataclasses](https://altaris.github.io/turbo-broccoli/turbo_broccoli/dataclass.html#to_json): serialization is straightforward:
  ```py
  @dataclass
  class C:
      a: int
      b: str

  doc = json.dumps({"c": C(a=1, b="Hello")}, cls=tb.TurboBroccoliEncoder)
  ```
  For deserialization, first register the class:
  ```py
  tb.register_dataclass_type(C)
  json.loads(doc, cls=tb.TurboBroccoliDecoder)
  ```

### [Generic objects](https://altaris.github.io/turbo-broccoli/turbo_broccoli/generic.html#to_json)

**serialization only**. A generic object is an object that
has the `__turbo_broccoli__` attribute. This attribute is expected to be a list
of attributes whose values will be serialized. For example,
```py
class C:
    __turbo_broccoli__ = ["a", "b"]
    a: int
    b: int
    c: int

x = C()
x.a, x.b, x.c = 42, 43, 44
json.dumps(x, cls=tb.TurboBroccoliEncoder)
```
produces the following string (modulo indentation):
```json
{
  "a": 42,
  "b": 43,
}
```

Registered attributes can of course have any type supported by Turbo Broccoli,
such as numpy arrays. Registered attributes can be `@property` methods.

### [Keras](https://altaris.github.io/turbo-broccoli/turbo_broccoli/keras.html#to_json)

- [`keras.Model`](https://keras.io/api/models/model/);

- standard subclasses of [`keras.layers.Layer`](https://keras.io/api/layers/),
  [`keras.losses.Loss`](https://keras.io/api/losses/),
  [`keras.metrics.Metric`](https://keras.io/api/metrics/), and
  [`keras.optimizers.Optimizer`](https://keras.io/api/optimizers/).

### [Numpy](https://altaris.github.io/turbo-broccoli/turbo_broccoli/numpy.html#to_json)

`numpy.number`, `numpy.ndarray` with numerical dtype, and `numpy.dtype`.

### [Pandas](https://altaris.github.io/turbo-broccoli/turbo_broccoli/pandas.html#to_json)

`pandas.DataFrame` and `pandas.Series`, but with the following limitations:

- the following dtypes are not supported: `complex`, `object`, `timedelta`

- the column / series names must be strings and not numbers. The following
  is not acceptable:
  ```py
  df = pd.DataFrame([[1, 2], [3, 4]])
  ```
  because
  ```py
  print([c for c in df.columns])
  # [0, 1]
  print([type(c) for c in df.columns])
  # [int, int]
  ```

### [Tensorflow](https://altaris.github.io/turbo-broccoli/turbo_broccoli/tensorflow.html#to_json)

`tensorflow.Tensor` with numerical dtype, but not `tensorflow.RaggedTensor`.

### [Pytorch](https://altaris.github.io/turbo-broccoli/turbo_broccoli/pytorch.html#to_json)

- `torch.Tensor`, **Warning**: loaded tensors are automatically placed on the
  CPU and gradients are lost;

- `torch.nn.Module`, don't forget to register your
  module type using
  [`turbo_broccoli.register_pytorch_module_type`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#register_pytorch_module_type):
  ```py
  # Serialization
  class MyModule(torch.nn.Module):
    ...

  module = MyModule()  # Must be instantiable without arguments
  doc = json.dumps(x, cls=tb.TurboBroccoliEncoder)

  # Deserialization
  tb.register_pytorch_module_type(MyModule)
  module = json.loads(doc, cls=tb.TurboBroccoliDecoder)
  ```
  **Warning**: It is not possible to register and deserialize [standard pytorch
  module containers](https://pytorch.org/docs/stable/nn.html#containers)
  directly. Wrap them in your own custom module class.

### [Scipy](https://altaris.github.io/turbo-broccoli/turbo_broccoli/scipy.html#to_json)

Just `scipy.sparse.csr_matrix`. ^^"

### [Scikit-learn](https://altaris.github.io/turbo-broccoli/turbo_broccoli/sklearn.html#to_json)

`sklearn` estimators (i.e. that descent from
[`sklean.base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)).
To make sure which class is supported, take a look at the [unit
tests](https://github.com/altaris/turbo-broccoli/blob/main/tests/test_sklearn.py)
Doesn't work with:

- All CV classes because the `score_` attribute is a dict indexed with
  `np.int64`, which `json.JSONEncoder._iterencode_dict` rejects.

- All estimator classes that have mandatory arguments: `ClassifierChain`,
  `ColumnTransformer`, `FeatureUnion`, `GridSearchCV`,
  `MultiOutputClassifier`, `MultiOutputRegressor`, `OneVsOneClassifier`,
  `OneVsRestClassifier`, `OutputCodeClassifier`, `Pipeline`,
  `RandomizedSearchCV`, `RegressorChain`, `RFE`, `RFECV`, `SelectFromModel`,
  `SelfTrainingClassifier`, `SequentialFeatureSelector`, `SparseCoder`,
  `StackingClassifier`, `StackingRegressor`, `VotingClassifier`,
  `VotingRegressor`.

- Everything that is parametrized by an arbitrary object/callable/estimator:
  `FunctionTransformer`, `TransformedTargetRegressor`.

- Other classes that have non JSON-serializable attributes:

    | Class                       | Non-serializable attr.    |
    | --------------------------- | ------------------------- |
    | `Birch`                     | `_CFNode`                 |
    | `GaussianProcessRegressor`  | `Sum`                     |
    | `GaussianProcessClassifier` | `Product`                 |
    | `Perceptron`                | `Hinge`                   |
    | `SGDClassifier`             | `Hinge`                   |
    | `SGDOneClassSVM`            | `Hinge`                   |
    | `PoissonRegressor`          | `HalfPoissonLoss`         |
    | `GammaRegressor`            | `HalfGammaLoss`           |
    | `TweedieRegressor`          | `HalfTweedieLossIdentity` |
    | `SplineTransformer`         | `BSpline`                 |

- Some classes have `AttributeErrors`?

    | Class                         | Attribute      |
    | ----------------------------- | -------------- |
    | `IsotonicRegression`          | `f_`           |
    | `KernelPCA`                   | `_centerer`    |
    | `KNeighborsClassifier`        | `_y`           |
    | `KNeighborsRegressor`         | `_y`           |
    | `KNeighborsTransformer`       | `_tree`        |
    | `LabelPropagation`            | `X_`           |
    | `LabelSpreading`              | `X_`           |
    | `LocalOutlierFactor`          | `_lrd`         |
    | `MissingIndicator`            | `_precomputed` |
    | `NuSVC`                       | `_sparse`      |
    | `NuSVR`                       | `_sparse`      |
    | `OneClassSVM`                 | `_sparse`      |
    | `PowerTransformer`            | `_scaler`      |
    | `RadiusNeighborsClassifier`   | `_tree`        |
    | `RadiusNeighborsRegressor`    | `_tree`        |
    | `RadiusNeighborsTransformer`  | `_tree`        |
    | `SVC`                         | `_sparse`      |
    | `SVR`                         | `_sparse`      |

- Other errors:

  - `FastICA`: I'm not sure why...

  - `BaggingClassifier`: `IndexError: only integers, slices (:), ellipsis
    (...), numpy.newaxis (None) and integer or boolean arrays are valid
    indices`.

  - `GradientBoostingClassifier`, `GradientBoostingRegressor`,
    `RandomTreesEmbedding`, `KBinsDiscretizer`: `Exception:
    dtype object is not covered`.

  - `HistGradientBoostingClassifier`: Problems with deserialization of
    `_BinMapper` object?

  - `PassiveAggressiveClassifier`: some unknown label type error...

  - `BisectingKMeans`: `TypeError: Object of type function is not JSON serializable`

### [Bokeh](https://altaris.github.io/turbo-broccoli/turbo_broccoli/bokeh.html#to_json)

Bokeh [figures](https://docs.bokeh.org/en/latest) and
[models](https://docs.bokeh.org/en/latest/docs/reference/models.html).

## [Secrets](https://altaris.github.io/turbo-broccoli/turbo_broccoli/secret.html#to_json)

Basic Python types can be wrapped in their corresponding secret type according
to the following table

| Python type | Secret type                         |
| ----------- | ----------------------------------- |
| `dict`      | `turbo_broccoli.secret.SecretDict`  |
| `float`     | `turbo_broccoli.secret.SecretFloat` |
| `int`       | `turbo_broccoli.secret.SecretInt`   |
| `list`      | `turbo_broccoli.secret.SecretList`  |
| `str`       | `turbo_broccoli.secret.SecretStr`   |

The secret value can be recovered with the `get_secret_value` method. At
serialization, the this value will be encrypted. For example,
```py
# See https://pynacl.readthedocs.io/en/latest/secret/#key
import nacl.secret
import nacl.utils

key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)

from turbo_broccoli.secret import SecretStr
from turbo_broccoli.environment import set_shared_key

set_shared_key(key)

x = {
    "user": "alice",
    "password": SecretStr("dolphin")
}
json.dumps(x, cls=tb.TurboBroccoliEncoder)
```
produces the following string (modulo indentation and modulo the encrypted
content):
```json
{
  "user": "alice",
  "password": {
    "__type__": "secret",
    "__version__": 2,
    "data": {
      "__type__": "bytes",
      "__version__": 2,
      "data": "Wk+42fDb8MoZT..."
    }
  }
}
```

Deserialization decrypts the secrets, but they stay wrapped inside the secret
types above. If the wrong key is provided, an exception is raised. If no key is
provided, the secret values are replaced by a
`turbo_broccoli.secret.LockedSecret`. Internally, Turbo Broccoli uses
[`pynacl`](https://pynacl.readthedocs.io/en/latest/)'s
[`SecretBox`](https://pynacl.readthedocs.io/en/latest/secret/#nacl.secret.SecretBox).
**Warning**: In the case of `SecretDict` and `SecretList`, the values contained
within must be JSON-serializable **without** Turbo Broccoli. See also the
`TB_SHARED_KEY` environment variable below.

## Environment variables

Some behaviors of Turbo Broccoli can be tweaked by setting specific environment
variables. If you want to modify these parameters programatically, do not do so
by modifying `os.environ`. Rather, use the methods of
`turbo_broccoli.environment`.

- `TB_ARTIFACT_PATH` (default: `./`; see also
  [`turbo_broccoli.set_artifact_path`]((https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#set_artifact_path)),
  [`turbo_broccoli.environment.get_artifact_path`]((https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#get_artifact_path))):
  During serialization, Turbo Broccoli may create artifacts to which the JSON
  object will point to. The artifacts will be stored in `TB_ARTIFACT_PATH`. For
  example, if `arr` is a big numpy array,
  ```py
  obj = {"an_array": arr}
  json.dumps(obj, cls=tb.TurboBroccoliEncoder)
  ```
  will generate the following string (modulo indentation and `id`)
  ```json
  {
    "an_array": {
      "__type__": "numpy.ndarray",
      "__version__": 3,
      "id": "70692d08-c4cf-4231-b3f0-0969ea552d5a"
    }
  }
  ```
  and a `70692d08-c4cf-4231-b3f0-0969ea552d5a` file has been created in
  `TB_ARTIFACT_PATH`.

- `TB_KERAS_FORMAT` (default: `tf`, valid values are `json`, `h5`, and `tf`;
  see also
  [`turbo_broccoli.set_keras_format`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#set_keras_format),
  [`turbo_broccoli.environment.get_keras_format`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#get_keras_format)):
  The serialization format for keras models. If `h5` or `tf` is used, an
  artifact following said format will be created in `TB_ARTIFACT_PATH`. If
  `json` is used, the model will be contained in the JSON document (anthough
  the weights may be in artifacts if they are too large).

- `TB_MAX_NBYTES` (default: `8000`, see also
  [`turbo_broccoli.set_max_nbytes`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#set_max_nbytes),
  [`turbo_broccoli.environment.get_max_nbytes`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#get_max_nbytes)):
  The maximum byte size of an numpy array or pandas object beyond which
  serialization will produce an artifact instead of storing it in the JSON
  document. This does not limit the size of the overall JSON document though.
  8000 bytes should be enough for a numpy array of 1000 `float64`s to be stored
  in-document.

- `TB_NODECODE` (default: empty; see also
  [`turbo_broccoli.set_nodecode`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#set_nodecode),
  [`turbo_broccoli.environment.is_nodecode`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#is_nodecode)):
  Comma-separated list of types to not deserialize, for example
  `bytes,numpy.ndarray`. Excludable types are:

  - `bokeh`, `bokeh.buffer`, `bokeh.generic`,

  - `bytes`,

  - `dict` (this will only disable [Turbo Broccoli's custom
    serialization](https://altaris.github.io/turbo-broccoli/turbo_broccoli/dict.html#to_json)
    of `dict`s with non `str` keys),

  - `collections`, `collections.deque`, `collections.namedtuple`,
    `collections.set`,

  - `dataclass`, `dataclass.<dataclass_name>` (case sensitive),

  - `generic`,

  - `keras`, `keras.model`, `keras.layer`, `keras.loss`, `keras.metric`,
    `keras.optimizer`,

  - `numpy`, `numpy.ndarray`, `numpy.number`, `numpy.dtype`,
    `numpy.random_state`,

  - `pandas`, `pandas.dataframe`, `pandas.series`, **Warning**: excluding
    `pandas.dataframe` will crash any deserialization of `pandas.series`,

  - `pytorch`, `pytorch.tensor`, `pytorch.module`,

  - `scipy`, `scipy.csr_matrix`,

  - `secret`,

  - `sklearn`, `sklearn.estimator`, `sklearn.estimator.<estimator name>` (case
    sensitive, see the list of supported sklearn estimators),

  - `tensorflow`, `tensorflow.sparse_tensor`, `tensorflow.tensor`,
    `tensorflow.variable`.

- `TB_SHARED_KEY` (default: empty; see also
  [`turbo_broccoli.set_shared_key`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#set_shared_key),
  [`turbo_broccoli.environment.get_shared_key`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#get_shared_key)):
  Secret key used to encrypt secrets. The encryption uses [`pynacl`'s
  `SecretBox`](https://pynacl.readthedocs.io/en/latest/secret/#nacl.secret.SecretBox).
  An exception is raised when attempting to serialize a secret type while no
  key is set.

## Guarded calls

This is so cool. Check out
[`turbo_broccoli.GuardedBlockHandler`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/guard.html#GuardedBlockHandler),
[`turbo_broccoli.guarded_call`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/guard.html#guarded_call),
and
[`turbo_broccoli.produces_document`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/guard.html#produces_document).

## CLI

Turbo Broccoli has a few utilities that can be accessed from the CLI.

- `list-artifacts`: Prints all the artifacts filenames or file paths that are
  referenced by a given json file.

- `rm`: Removes a json file and all the artifacts it references.

# Contributing

## Dependencies

- `python3.9` or newer;

- `requirements.txt` for runtime dependencies;

- `requirements.dev.txt` for development dependencies.

- `make` (optional);

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
