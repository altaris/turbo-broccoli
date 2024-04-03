# Turbo Broccoli 🥦

[![Repository](https://img.shields.io/badge/repo-github-pink)](https://github.com/altaris/turbo-broccoli)
[![PyPI](https://img.shields.io/pypi/v/turbo-broccoli)](https://pypi.org/project/turbo-broccoli/)
![License](https://img.shields.io/github/license/altaris/turbo-broccoli)
[![Code
style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)
![hehe](https://img.shields.io/badge/project%20name%20by-github-pink)
[![Documentation](https://badgen.net/badge/documentation/here/green)](https://altaris.github.io/turbo-broccoli/turbo_broccoli.html)

JSON (de)serialization extensions, originally aimed at `numpy` and `tensorflow`
objects, but now supporting a wide range of objects.

## Installation

```sh
pip install turbo-broccoli
```

## Usage

### To/from string

```py
import numpy as np
import turbo_broccoli as tb

obj = {"an_array": np.array([[1, 2], [3, 4]], dtype="float32")}
tb.to_json(obj)
```

produces the following string (modulo indentation and the value of
`$.an_array.data.data`):

```json
{
  "an_array": {
    "__type__": "numpy.ndarray",
    "__version__": 5,
    "data": {
      "__type__": "bytes",
      "__version__": 3,
      "data": "QAAAAAAAAAB7ImRhd..."
    }
  }
}
```

For deserialization, simply use

```py
tb.from_json(json_string)
```

### To/from file

Simply replace
[`turbo_broccoli.to_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#to_json)
and
[`turbo_broccoli.from_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#from_json)
with
[`turbo_broccoli.save_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#save_json)
and
[`turbo_broccoli.load_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#load_json):

```py
import numpy as np
import turbo_broccoli as tb

obj = {"an_array": np.array([[1, 2], [3, 4]], dtype="float32")}
tb.save_json(obj, "foo/bar/foobar.json")

...

obj = tb.load_json("foo/bar/foobar.json")
```

### [Contexts](https://altaris.github.io/turbo-broccoli/turbo_broccoli/context.html#Context)

The behaviour of
[`turbo_broccoli.to_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#to_json)
and
[`turbo_broccoli.from_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#from_json)
can be tweaked by using
[contexts](https://altaris.github.io/turbo-broccoli/turbo_broccoli/context.html#Context).
For example, to set a encryption/decryption key for [secret
types](https://cedric.hothanh.fr/turbo-broccoli/turbo_broccoli/custom/secret.html):

```py
import nacl.secret
import nacl.utils
import turbo_broccoli as tb

key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
ctx = tb.Context(nacl_shared_key=key)
obj = {"user": "alice", "password": tb.SecretStr("dolphin")}
doc = tb.to_json(obj, ctx)

...

obj = tb.from_json(doc, ctx)
```

The behaviour of
[`turbo_broccoli.save_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#save_json)
and
[`turbo_broccoli.load_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#load_json)
can be tweaked in a similar manner. For convenience, the argument of the
context can be passed directly to the method instead of creating a context
object manually:

```py
import nacl.secret
import nacl.utils
import turbo_broccoli as tb

key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
obj = {"user": "alice", "password": tb.SecretStr("dolphin")}
tb.save_json(obj, "foo/bar/foobar.json", nacl_shared_key=key)
```

See [the
documentation](https://altaris.github.io/turbo_broccoli/context.html#Context).

### Artifacts

If an object inside `obj` is too large to be embedded inside the JSON file
(e.g. a large numpy array), then an *artifact* file is created:

```py
import numpy as np
import turbo_broccoli as tb

obj = {"an_array": np.random.rand(1000, 1000)}
tb.save_json(obj, "foo/bar/foobar.json")
```

produces the JSON file

```json
{
  "an_array": {
    "__type__": "numpy.ndarray",
    "__version__": 5,
    "data": {
      "__type__": "bytes",
      "__version__": 3,
      "id": "1e6dff28-5e26-44df-9e7a-75bc726ce9aa"
    }
  }
}
```

and a file `foo/bar/foobar.1e6dff28-5e26-44df-9e7a-75bc726ce9aa.tb` containing
the array data. The artifact directory can be explicitely specified by setting
it in the [serialization
context](https://altaris.github.io/turbo-broccoli/turbo_broccoli/context.html#Context)
or by setting the `TB_ARTIFACT_PATH` environment variable (see below.). The
code for loading the JSON file does not change:

```py
obj = tb.load_json("foo/bar/foobar.json")
```

If using
[`turbo_broccoli.to_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#to_json),
since there is no output file path specified, the artifacts are storied in a
temporary directory instead:

```py
import numpy as np
import turbo_broccoli as tb

obj = {"an_array": np.random.rand(1000, 1000)}
doc = tb.to_json(obj)
# An artifact has been created somewhere in e.g. /tmp
```

Since no information about this directory is stored in the output JSON string,
it is not possible to load `doc` using
[`turbo_broccoli.from_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#load_json).
If deserialization is necessary, instantiate a [context](https://altaris.github.io/turbo-broccoli/turbo_broccoli/context.html#Context):

```py
import numpy as np
import turbo_broccoli as tb

ctx = tb.Context()
obj = {"an_array": np.random.rand(1000, 1000)}
doc = tb.to_json(obj, ctx)
# An artifact has been created in ctx.artifact_path

...

obj = tb.from_json(doc, ctx)
```

## Environment variables

Some behaviors of Turbo Broccoli can be tweaked by setting specific environment
variables. If you want to modify these parameters programatically, do not do so
by modifying `os.environ`. Rather, use a
[`turbo_broccoli.Context`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/context.html#Context).

- `TB_ARTIFACT_PATH` (default: output JSON file's parent directory): During
  serialization, Turbo Broccoli may create artifacts to which the JSON object
  will point to. The artifacts will be stored in `TB_ARTIFACT_PATH` if
  specified.

- `TB_KERAS_FORMAT` (default: `tf`, valid values are `keras`, `tf`, and `h5`):
  The serialization format for keras models. If `h5` or `tf` is used, an
  artifact following said format will be created in `TB_ARTIFACT_PATH`. If
  `json` is used, the model will be contained in the JSON document (anthough
  the weights may be in artifacts if they are too large).

- `TB_MAX_NBYTES` (default: `8000`):
  The maximum byte size of a python object beyond which serialization will
  produce an artifact instead of storing it in the JSON document. This does not
  limit the size of the overall JSON document though. 8000 bytes should be
  enough for a numpy array of 1000 `float64`s to be stored in-document.

- `TB_NODECODE` (default: empty):
  Comma-separated list of types to not deserialize, for example
  `bytes,numpy.ndarray`. Excludable types are:

  - `bokeh`, `bokeh.buffer`, `bokeh.generic`,

  - `bytes`, **Warning** excluding `bytes` will also exclude `bokeh`,
    `numpy.ndarray`, `pytorch.module`, `pytorch.tensor`, `secret`,
    `tensorflow.tensor`,

  - `collections`, `collections.deque`, `collections.namedtuple`,
    `collections.set`,

  - `dataclass`, `dataclass.<dataclass_name>` (case sensitive),

  - `datetime`, `datetime.datetime`, `datetime.time`, `datetime.timedelta`,

  - `dict` (this only prevents decoding dicts with non-string keys),

  - `embedded`, `embedded.dict`, `embedded.list`,

  - `generic`,

  - `keras`, `keras.model`, `keras.layer`, `keras.loss`, `keras.metric`,
    `keras.optimizer`,

  - `numpy`, `numpy.ndarray`, `numpy.number`, `numpy.dtype`,
    `numpy.random_state`,

  - `pandas`, `pandas.dataframe`, `pandas.series`, **Warning**: excluding
    `pandas.dataframe` will also exclude `pandas.series`,

  - `pytorch`, `pytorch.tensor`, `pytorch.module`, `pytorch.concatdataset`,
    `pytorch.stackdataset`, `pytorch.subset`, `pytorch.tensordataset`

  - `scipy`, `scipy.csr_matrix`,

  - `secret`,

  - `sklearn`, `sklearn.estimator`, `sklearn.estimator.<estimator name>` (case
    sensitive, see the list of supported sklearn estimators below),

  - `tensorflow`, `tensorflow.sparse_tensor`, `tensorflow.tensor`,
    `tensorflow.variable`.

  - `uuid`

- `TB_SHARED_KEY` (default: empty):
  Secret key used to encrypt/decrypt secrets. The encryption uses [`pynacl`'s
  `SecretBox`](https://pynacl.readthedocs.io/en/latest/secret/#nacl.secret.SecretBox).
  An exception is raised when attempting to serialize a secret type while no
  key is set.

## Guarded blocks

A
[`turbo_broccoli.GuardedBlockHandler`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/guard.html#GuardedBlockHandler)
"guards" a block of code, meaning it prevents it from being executed if it has
been in the past. Check out [the
documentation](https://altaris.github.io/turbo-broccoli/turbo_broccoli/guard.html)
for some examples.

## Guarded-parallel executors

A mix of
[`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
and
[`turbo_broccoli.GuardedBlockHandler`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/guard.html#GuardedBlockHandler):
a
[`turbo_broccoli.Parallel`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/parallel.html#Parallel)
object can be used to execute jobs in parallel, but those whose results have
already been obtained in the past are skipped. See [the
documentation](https://altaris.github.io/turbo-broccoli/turbo_broccoli/parallel.html)
for some examples.

## Supported types

### Basic types

- [`bytes`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/bytes.html#to_json)

- [Collections](https://altaris.github.io/turbo-broccoli/turbo_broccoli/collections.html#to_json):
  `collections.deque`, `collections.namedtuple`

- [Dataclasses](https://altaris.github.io/turbo-broccoli/turbo_broccoli/dataclass.html#to_json):
  serialization is straightforward:

  ```py
  @dataclass
  class C:
      a: int
      b: str

  doc = tb.to_json({"c": C(a=1, b="Hello")})
  ```

  For deserialization, first register the class:

  ```py
  ctx = tb.Context(dataclass_types=[C])
  tb.from_json(doc, ctx)
  ```

- [`datetime.datetime`, `datetime.time`,
  `datetime.timedelta`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/datetime.html#to_json)

- Non JSON-able dicts, i.e. dicts whose keys are not all `str`, `int`, `float`,
  `bool` or `None`

- [UUID objects](https://docs.python.org/3/library/uuid.html)

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
tb.to_json(x)
```

produces the following string:

```json
{"a": 42,"b": 43,}
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

- the column / series names cannot be ints or int-strings. The following are
  not acceptable:

  ```py
  df = pd.DataFrame([[1, 2], [3, 4]])
  df = pd.DataFrame([[1, 2], [3, 4]], columns=["0", "1"])
  ```

### [Tensorflow](https://altaris.github.io/turbo-broccoli/turbo_broccoli/tensorflow.html#to_json)

`tensorflow.Tensor` with numerical dtype, but not `tensorflow.RaggedTensor`.

### [Pytorch](https://altaris.github.io/turbo-broccoli/turbo_broccoli/pytorch.html#to_json)

- `torch.Tensor`, **Warning**: loaded tensors are automatically placed on the
  CPU and gradients are lost;

- `torch.nn.Module`, don't forget to register your module type using a
  [`turbo_broccoli.Context`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/context.html#Context):

  ```py
  # Serialization
  class MyModule(torch.nn.Module):
    ...

  module = MyModule()  # Must be instantiable without arguments
  doc = tb.to_json({"module": module})

  # Deserialization
  ctx = tb.Context(pytorch_module_types=[MyModule])
  module = tb.from_json(doc, ctx)
  ```

  **Warning**: It is not possible to register and deserialize [standard pytorch
  module containers](https://pytorch.org/docs/stable/nn.html#containers)
  directly. Wrap them in your own custom module class. For following is not
  acceptable

  ```py
  import turbo_broccoli as tb
  import torch

  module = torch.nn.Sequential(
      torch.nn.Linear(4, 2),
      torch.nn.ReLU(),
      torch.nn.Linear(2, 1),
      torch.nn.ReLU(),
  )
  obj = {"module": module}
  doc = tb.to_json(obj)  # works, but...
  tb.from_json(a, ctx)  # does't work
  ```

  but the following works:

  ```py
  class MyModule(torch.nn.Module):
    module: torch.nn.Sequential  # Wrapped sequential

    def __init__(self):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
            torch.nn.ReLU(),
        )

    ...

  module = MyModule()  # Must be instantiable without arguments
  doc = tb.to_json({"module": module})

  ctx = tb.Context(pytorch_module_types=[MyModule])
  module = tb.from_json(doc, ctx)
  ```

- `torch.utils.data.ConcatDataset`, `torch.utils.data.StackDataset`,
  `torch.utils.data.Subset`, `torch.utils.data.TensorDataset`, as long as the
  nested structure of datasets ultimately lead to
  `torch.utils.data.TensorDataset`s (e.g. a subset of a stack of subsets of
  tensor datasets is supported)

### [Scipy](https://altaris.github.io/turbo-broccoli/turbo_broccoli/scipy.html#to_json)

Just `scipy.sparse.csr_matrix`. ^^"

### [Scikit-learn](https://altaris.github.io/turbo-broccoli/turbo_broccoli/sklearn.html#to_json)

`sklearn` estimators (i.e. that inherit from
[`sklean.base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)).
Supported estimators are: `AdaBoostClassifier`, `AdaBoostRegressor`,
`AdditiveChi2Sampler`, `AffinityPropagation`, `AgglomerativeClustering`,
`ARDRegression`, `BayesianGaussianMixture`, `BayesianRidge`, `BernoulliNB`,
`BernoulliRBM`, `Binarizer`, `CategoricalNB`, `CCA`, `ClassifierChain`,
`ComplementNB`, `DBSCAN`, `DecisionTreeClassifier`, `DecisionTreeRegressor`,
`DictionaryLearning`, `ElasticNet`, `EllipticEnvelope`, `EmpiricalCovariance`,
`ExtraTreeClassifier`, `ExtraTreeRegressor`, `ExtraTreesClassifier`,
`ExtraTreesRegressor`, `FactorAnalysis`, `FeatureUnion`, `GaussianMixture`,
`GaussianNB`, `GaussianRandomProjection`, `GraphicalLasso`, `HuberRegressor`,
`IncrementalPCA`, `IsolationForest`, `Isomap`, `KernelCenterer`,
`KernelDensity`, `KernelPCA`, `KernelRidge`, `KMeans`, `KNeighborsClassifier`,
`KNeighborsRegressor`, `KNNImputer`, `LabelBinarizer`, `LabelEncoder`,
`LabelPropagation`, `LabelSpreading`, `Lars`, `Lasso`, `LassoLars`,
`LassoLarsIC`, `LatentDirichletAllocation`, `LedoitWolf`,
`LinearDiscriminantAnalysis`, `LinearRegression`, `LinearSVC`, `LinearSVR`,
`LocallyLinearEmbedding`, `LocalOutlierFactor`, `LogisticRegression`,
`MaxAbsScaler`, `MDS`, `MeanShift`, `MinCovDet`, `MiniBatchDictionaryLearning`,
`MiniBatchKMeans`, `MiniBatchSparsePCA`, `MinMaxScaler`, `MissingIndicator`,
`MLPClassifier`, `MLPRegressor`, `MultiLabelBinarizer`, `MultinomialNB`,
`MultiOutputClassifier`, `MultiOutputRegressor`, `MultiTaskElasticNet`,
`MultiTaskLasso`, `NearestCentroid`, `NearestNeighbors`,
`NeighborhoodComponentsAnalysis`, `NMF`, `Normalizer`, `NuSVC`, `NuSVR`,
`Nystroem`, `OAS`, `OneClassSVM`, `OneVsOneClassifier`, `OneVsRestClassifier`,
`OPTICS`, `OrthogonalMatchingPursuit`, `PassiveAggressiveRegressor`, `PCA`,
`Pipeline`, `PLSCanonical`, `PLSRegression`, `PLSSVD`, `PolynomialCountSketch`,
`PolynomialFeatures`, `PowerTransformer`, `QuadraticDiscriminantAnalysis`,
`QuantileRegressor`, `QuantileTransformer`, `RadiusNeighborsClassifier`,
`RadiusNeighborsRegressor`, `RandomForestClassifier`, `RandomForestRegressor`,
`RANSACRegressor`, `RBFSampler`, `RegressorChain`, `RFE`, `RFECV`, `Ridge`,
`RidgeClassifier`, `RobustScaler`, `SelectFromModel`, `SelfTrainingClassifier`,
`SGDRegressor`, `ShrunkCovariance`, `SimpleImputer`, `SkewedChi2Sampler`,
`SparsePCA`, `SparseRandomProjection`, `SpectralBiclustering`,
`SpectralClustering`, `SpectralCoclustering`, `SpectralEmbedding`,
`StackingClassifier`, `StackingRegressor`, `StandardScaler`, `SVC`, `SVC`,
`SVR`, `SVR`, `TheilSenRegressor`, `TruncatedSVD`, `TSNE`, `VarianceThreshold`,
`VotingClassifier`, `VotingRegressor`. Doesn't work with:

- All CV classes because the `score_` attribute is a dict indexed with
  `np.int64`, which `json.JSONEncoder._iterencode_dict` rejects.

- Everything that is parametrized by an arbitrary object/callable/estimator:
  `FunctionTransformer`, `TransformedTargetRegressor`.

- Other classes that have non JSON-serializable attributes:

    | Class                       | Non-serializable attr.    |
    | --------------------------- | ------------------------- |
    | `Birch`                     | `_CFNode`                 |
    | `BisectingKMeans`           | `function`                |
    | `ColumnTransformer`         | `slice`                   |
    | `GammaRegressor`            | `HalfGammaLoss`           |
    | `GaussianProcessClassifier` | `Product`                 |
    | `GaussianProcessRegressor`  | `Sum`                     |
    | `IsotonicRegression`        | `interp1d`                |
    | `OutputCodeClassifier`      | `_ConstantPredictor`      |
    | `Perceptron`                | `Hinge`                   |
    | `PoissonRegressor`          | `HalfPoissonLoss`         |
    | `SGDClassifier`             | `Hinge`                   |
    | `SGDOneClassSVM`            | `Hinge`                   |
    | `SplineTransformer`         | `BSpline`                 |
    | `TweedieRegressor`          | `HalfTweedieLossIdentity` |

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

  - `SequentialFeatureSelector`: Problem with the unit test itself ^^"

  - `KNeighborsTransformer`: A serialized-deserialized instance seems to
    `fit_transform` an array to a sparse matrix whereas the original object
    returns an array?

  - `RadiusNeighborsTransformer`: Inverse problem from `KNeighborsTransformer`.

### [Bokeh](https://altaris.github.io/turbo-broccoli/turbo_broccoli/bokeh.html#to_json)

Bokeh [figures](https://docs.bokeh.org/en/latest) and
[models](https://docs.bokeh.org/en/latest/docs/reference/models.html).

### [Secrets](https://altaris.github.io/turbo-broccoli/turbo_broccoli/secret.html#to_json)

Basic Python types can be wrapped in their corresponding secret type according
to the following table

| Python type | Secret type                  |
| ----------- | ---------------------------- |
| `dict`      | `turbo_broccoli.SecretDict`  |
| `float`     | `turbo_broccoli.SecretFloat` |
| `int`       | `turbo_broccoli.SecretInt`   |
| `list`      | `turbo_broccoli.SecretList`  |
| `str`       | `turbo_broccoli.SecretStr`   |

The secret value can be recovered with the `get_secret_value` method. At
serialization, the this value will be encrypted. For example,

```py
## See https://pynacl.readthedocs.io/en/latest/secret/#key
import nacl.secret
import nacl.utils
import turbo_broccoli as tb

key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
ctx = tb.Context(nacl_shared_key=key)
obj = {"user": "alice", "password": tb.SecretStr("dolphin")}
tb.to_json(obj, ctx)
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
      "__version__": 3,
      "data": "gbRXF3hq9Q9hIQ9Xz+WdGKYP5meJ4eTmlFt0r0Ov3PV64065plk6RqsFUcynSOqHzA=="
    }
  }
}
```

Deserialization decrypts the secrets, but they stay wrapped inside the secret
types above. If the wrong key is provided, an exception is raised. If no key is
provided, the secret values are replaced by a
`turbo_broccoli.LockedSecret`. Internally, Turbo Broccoli uses
[`pynacl`](https://pynacl.readthedocs.io/en/latest/)'s
[`SecretBox`](https://pynacl.readthedocs.io/en/latest/secret/#nacl.secret.SecretBox).
**Warning**: In the case of `SecretDict` and `SecretList`, the values contained
within must be JSON-serializable **without** Turbo Broccoli. The following is
not acceptable:

```py
import nacl.secret
import nacl.utils
import numpy as np
import turbo_broccoli as tb

key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
ctx = tb.Context(nacl_shared_key=key)
obj = {"data": tb.SecretList([np.array([1, 2, 3])])}
tb.to_json(obj, ctx)
```

See also the `TB_SHARED_KEY` environment variable below.

### Embedded dict/lists

Sometimes, it may be useful to store part of a document in its own file and
have referrenced in the main file. This is possible using
[`EmbeddedDict`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/custom/embedded.html)
and
[`EmbeddedList`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/custom/embedded.html).
For example,

```py
from turbo_broccoli import save_json, EmbeddedDict

data = {"a": 1, "b": EmbeddedDict({"c": 2, "d": 3})}
save_json(data, "data.json")
```

will result in a `data.json` file containing

```json
{
  "a": 1,
  "b": {
    "__type__": "embedded.dict",
    "__version__": 1,
    "id": "4ea0b3f3-f3e4-42bd-9db9-1e4e0b9f4fae"
  }
}
```

(modulo indentation and the id), and an artefact file
`data.4ea0b3f3-f3e4-42bd-9db9-1e4e0b9f4fae.json` containing

```json
{"c": 2, "d": 3}
```

## Contributing

### Dependencies

- `python3.10` or newer;

- `requirements.txt` for runtime dependencies;

- `requirements.dev.txt` for development dependencies.

- `make` (optional);

Simply run

```sh
virtualenv venv -p python3.10
. ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

### Documentation

Simply run

```sh
make docs
```

This will generate the HTML doc of the project, and the index file should be at
`docs/index.html`. To have it directly in your browser, run

```sh
make docs-browser
```

### Code quality

Don't forget to run

```sh
make
```

to format the code following [black](https://pypi.org/project/black/),
typecheck it using [mypy](http://mypy-lang.org/), and check it against coding
standards using [pylint](https://pylint.org/).

### Unit tests

Run

```sh
make test
```

to have [pytest](https://docs.pytest.org/) run the unit tests in `tests/`.

## Credits

This project takes inspiration from
[Crimson-Crow/json-numpy](https://github.com/Crimson-Crow/json-numpy).
