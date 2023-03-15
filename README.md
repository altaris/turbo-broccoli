# Turbo Broccoli 🥦

[![PyPI](https://img.shields.io/pypi/v/turbo-broccoli)](https://pypi.org/project/turbo-broccoli/)
![License](https://img.shields.io/github/license/altaris/turbo-broccoli)
[![Code
style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)
![hehe](https://img.shields.io/badge/project%20name%20by-github-pink)
[![Documentation](https://badgen.net/badge/documentation/here/green)](https://https://altaris.github.io/turbo-broccoli/turbo_broccoli.html)

JSON (de)serialization extensions, originally aimed at `numpy` and `tensorflow`
objects.

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
    "__numpy__": {
      "__type__": "ndarray",
      "__version__": 3,
      "data": {
        "__bytes__": {
          "__version__": 1,
          "data": "PAAAAA..."
        }
      }
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

## Guarded calls

Consider an expensive function `f` that returns a TurboBroccoli/JSON-izable
`dict`. Wrapping/decorating `f` using `produces_document` essentially saves the
result at a specified path and when possible, loads it instead of calling `f`.
For example:

```py
_f = produces_document(f, "out/result.json")
_f(*args, **kwargs)
```

will only call `f` if the `out/result.json` does not exist, and otherwise,
loads and returns `out/result.json`. However, if `out/result.json` exists and
was produced by calling `_f(*args, **kwargs)`, then

```py
_f(*args2, **kwargs2)
```

will still return the same result. If you want to keep a different file for
each `args`/`kwargs`, set `check_args` to `True` as in

```py
_f = produces_document(f, "out/result.json")
_f(*args, **kwargs)
```

In this case, the arguments must be TurboBroccoli/JSON-izable, i.e. the
document

```
{
    "args": args,
    "kwargs": kwargs,
}
```

must be TurboBroccoli/JSON-izable. The resulting file is no longer
`out/result.json` but rather `out/result.json/<hash>` where `hash` is the MD5
hash of the serialization of the `args`/`kwargs` document above.

## Supported types

- `bytes`

- `collections.deque`, `collections.namedtuple`

- Dataclasses. Serialization is straightforward:

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

- _Generic object_, **serialization only**. A generic object is an object that
  has the `__turbo_broccoli__` attribute. This attribute is expected to be a
  list of attributes whose values will be serialized. For example,

  ```py
  class C:
      __turbo_broccoli__ = ["a"]
      a: int
      b: int

  x = C()
  x.a, x.b = 42, 43
  json.dumps(x, cls=tb.TurboBroccoliEncoder)
  ```

  produces the following string (modulo indentation):

  ```json
  {
    "__generic__": {
      "__version__": 1,
      "data": {
        "a": 42
      }
    }
  }
  ```

  Registered attributes can of course have any type supported by Turbo
  Broccoli, such as numpy arrays. Registered attributes can be `@property`
  methods.

- [`keras.Model`](https://keras.io/api/models/model/); standard subclasses of
  [`keras.layers.Layer`](https://keras.io/api/layers/),
  [`keras.losses.Loss`](https://keras.io/api/losses/),
  [`keras.metrics.Metric`](https://keras.io/api/metrics/), and
  [`keras.optimizers.Optimizer`](https://keras.io/api/optimizers/)

- `numpy.number`, `numpy.ndarray` with numerical dtype

- `pandas.DataFrame` and `pandas.Series`, but with the following limitations:

  1. the following dtypes are not supported: `complex`, `object`, `timedelta`
  2. the column / series names must be strings and not numbers. The following
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

- `tensorflow.Tensor` with numerical dtype, but not `tensorflow.RaggedTensor`

- `torch.Tensor`, **WARNING**: loaded tensors are automatically placed on the
  CPU and gradients are lost; `torch.nn.Module`, don't forget to register your
  module type using
  [`turbo_broccoli.register_pytorch_module_type`]((https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#register_pytorch_module_type)):
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
  **WARNING**: It is not possible to register and deserialize [standard pytorch
  module containers](https://pytorch.org/docs/stable/nn.html#containers)
  directly. Wrap them in your own custom module class.

- `scipy.sparse.csr_matrix`

- **EXPERIMENTAL** `sklearn` estimators (i.e. that descent from
  [`sklean.base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)).
  To make sure which class is supported, take a look at the [unit
  tests](https://github.com/altaris/turbo-broccoli/blob/main/tests/test_sklearn.py)
  Doesn't work with:
  * All CV classes because the `score_` attribute is a dict indexed with
    `np.int64`, which `json.JSONEncoder._iterencode_dict` rejects.
  * All estimator classes that have mandatory arguments: `ClassifierChain`,
    `ColumnTransformer`, `FeatureUnion`, `GridSearchCV`,
    `MultiOutputClassifier`, `MultiOutputRegressor`, `OneVsOneClassifier`,
    `OneVsRestClassifier`, `OutputCodeClassifier`, `Pipeline`,
    `RandomizedSearchCV`, `RegressorChain`, `RFE`, `RFECV`, `SelectFromModel`,
    `SelfTrainingClassifier`, `SequentialFeatureSelector`, `SparseCoder`,
    `StackingClassifier`, `StackingRegressor`, `VotingClassifier`,
    `VotingRegressor`.
  * Everything that is parametrized by an arbitrary object/callable/estimator:
    `FunctionTransformer`, `TransformedTargetRegressor`.
  * Everything that stores a random state (in the form of a `RandomState`
    object): `BisectingKMeans`, `MiniBatchDictionaryLearning`,
    `LatentDirichletAllocation`, `NeighborhoodComponentsAnalysis`,
    `MLPClassifier`, `MLPRegressor`, `SparseRandomProjection`,
    `GaussianRandomProjection`.
  * Everything with trees and forest since `Tree` objects are not JSON
    serializable: `ExtraTreesClassifier`, `ExtraTreesRegressor`,
    `RandomForestClassifier`, `RandomForestRegressor`, `RandomTreesEmbedding`,
    `IsolationForest`, `AdaBoostClassifier`, `AdaBoostRegressor`,
    `DecisionTreeClassifier`, `DecisionTreeRegressor`.
  * Other classes that have non JSON-serializable attributes:
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
    | `KernelDensity`             | `KDTree`                  |
    | `SplineTransformer`         | `BSpline`                 |
  * Some classes have AttributeErrors?
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
  * Other errors:
    * `FastICA`: I'm not sure why...
    * `BaggingClassifier`: `IndexError: only integers, slices (:), ellipsis
      (...), numpy.newaxis (None) and integer or boolean arrays are valid
      indices`.
    * `GradientBoostingClassifier`: `Exception: dtype object is not covered`.
    * `GradientBoostingRegressor`: `Exception: dtype object is not covered`.
    * `HistGradientBoostingClassifier`: Problems with deserialization of
      `_BinMapper` object?
    * `PassiveAggressiveClassifier`: some unknown label type error...
    * `KBinsDiscretizer`: `Exception: dtype object is not covered`.
    * `KBinsDiscretizer`: `Exception: dtype object is not covered`.


## Secrets

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
    "__secret__": {
      "__version__": 1,
      "data": {
        "__bytes__": {
          "__version__": 1,
          "data": "qPSsruu..."
        }
      }
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
**WARNING**: In the case of `SecretDict` and `SecretList`, the values contained
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

  will generate the following string (modulo indentation and id)

  ```json
  {
      "an_array": {
          "__numpy__": {
              "__type__": "ndarray",
              "__version__": 3,
              "id": "70692d08-c4cf-4231-b3f0-0969ea552d5a"
          }
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

  - `bytes`,
  - `dataclass.<dataclass_name>` (case sensitive),
  - `collections.deque`, `collections.namedtuple`,
  - `keras.model`, `keras.layer`, `keras.loss`, `keras.metric`,
    `keras.optimizer`,
  - `numpy.ndarray`, `numpy.number`,
  - `pandas.dataframe`, `pandas.series`, **WARNING: excluding
    `pandas.dataframe` will crash any deserialization of `pandas.series`**
  - `tensorflow.sparse_tensor`, `tensorflow.tensor`, `tensorflow.variable`.
    **WARNING**: excluding `numpy.ndarray` will may crash deserialization of
    Tensorflow and Pandas types.

- `TB_SHARED_KEY` (default: empty; see also
  [`turbo_broccoli.set_shared_key`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#set_shared_key),
  [`turbo_broccoli.environment.get_shared_key`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/environment.html#get_shared_key)):
  Secret key used to encrypt secrets. The encryption uses [`pynacl`'s
  `SecretBox`](https://pynacl.readthedocs.io/en/latest/secret/#nacl.secret.SecretBox).
  An exception is raised when attempting to serialize a secret type while no
  key is set.

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
