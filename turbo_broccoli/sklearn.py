"""Scikit-learn estimators"""

import re
from typing import Any, Callable, Tuple
from uuid import uuid4

# Sklearn recommends joblib rather than direct pickle
# https://scikit-learn.org/stable/model_persistence.html#python-specific-serialization
import joblib
from sklearn import (
    calibration,
    cluster,
    compose,
    covariance,
    cross_decomposition,
    datasets,
    decomposition,
    discriminant_analysis,
    dummy,
    ensemble,
    exceptions,
    feature_extraction,
    feature_selection,
    gaussian_process,
    impute,
    inspection,
    isotonic,
    kernel_approximation,
    kernel_ridge,
    linear_model,
    manifold,
    metrics,
    mixture,
    model_selection,
    multiclass,
    multioutput,
    naive_bayes,
    neighbors,
    neural_network,
    pipeline,
    preprocessing,
    random_projection,
    semi_supervised,
    svm,
    tree,
)
from sklearn.base import BaseEstimator
from sklearn.tree._tree import Tree

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)

from .environment import get_artifact_path

_SKLEARN_SUBMODULES = [
    # calibration,
    cluster,
    covariance,
    cross_decomposition,
    datasets,
    decomposition,
    # dummy,
    ensemble,
    exceptions,
    # experimental,
    # externals,
    feature_extraction,
    feature_selection,
    gaussian_process,
    inspection,
    isotonic,
    # kernel_approximation,
    # kernel_ridge,
    linear_model,
    manifold,
    metrics,
    mixture,
    model_selection,
    multiclass,
    multioutput,
    naive_bayes,
    neighbors,
    neural_network,
    pipeline,
    preprocessing,
    random_projection,
    semi_supervised,
    svm,
    tree,
    discriminant_analysis,
    impute,
    compose,
]

_SKLEARN_TREE_ATTRIBUTES = [
    "capacity",
    "children_left",
    "children_right",
    "feature",
    "impurity",
    "max_depth",
    "max_n_classes",
    "n_classes",
    "n_features",
    "n_leaves",
    "n_node_samples",
    "n_outputs",
    "node_count",
    "threshold",
    "value",
    "weighted_n_node_samples",
]

_SUPPORTED_PICKLABLE_TYPES = [
    tree._tree.Tree,
    neighbors.KDTree,
]
"""sklearn types that shall be pickled"""


def _all_base_estimators() -> dict[str, type]:
    """
    Returns (hopefully) all classes of sklearn that inherit from
    `BaseEstimator`
    """
    result = []
    for s in _SKLEARN_SUBMODULES:
        if not hasattr(s, "__all__"):
            continue
        s_all = getattr(s, "__all__")
        if not isinstance(s_all, list):
            continue
        for k in s_all:
            cls = getattr(s, k)
            if isinstance(cls, type) and issubclass(cls, BaseEstimator):
                result.append(cls)
    # Some sklearn submodules don't have __all__
    result += [
        calibration.CalibratedClassifierCV,
        dummy.DummyClassifier,
        dummy.DummyRegressor,
        kernel_approximation.PolynomialCountSketch,
        kernel_approximation.RBFSampler,
        kernel_approximation.SkewedChi2Sampler,
        kernel_approximation.AdditiveChi2Sampler,
        kernel_approximation.Nystroem,
        kernel_ridge.KernelRidge,
    ]
    return {cls.__name__: cls for cls in result}


def _sklearn_estimator_to_json(obj: BaseEstimator) -> dict:
    r = re.compile(r"\w[\w_]*[^_]_")
    return {
        "__type__": "sklearn.estimator." + obj.__class__.__name__,
        "__version__": 2,
        "params": obj.get_params(deep=False),
        "attrs": {k: v for k, v in obj.__dict__.items() if r.match(k)},
    }


def _sklearn_to_raw(obj: Any) -> dict:
    """
    Pickles an otherwise unserializable sklearn object. Actually uses the
    `joblib.dump`.

    TODO:
        Don't dump to file if the object is small enough. Unfortunately
        `joblib` can't dump to a string.
    """
    name = str(uuid4())
    joblib.dump(obj, get_artifact_path() / name)
    return {
        "__type__": "sklearn.raw",
        "__version__": 2,
        "data": name,
    }


def _sklearn_tree_to_json(obj: Tree) -> dict:
    return {
        "__type__": "sklearn.tree",
        "__version__": 2,
        **{a: getattr(obj, a) for a in _SKLEARN_TREE_ATTRIBUTES},
    }


def _json_raw_to_sklearn(dct: dict) -> Any:
    DECODERS = {
        # 1: _json_raw_to_sklearn_v1,  # Use turbo_broccoli v3
        2: _json_raw_to_sklearn_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_raw_to_sklearn_v2(dct: dict) -> Any:
    return joblib.load(get_artifact_path() / dct["data"])


def _json_to_sklearn_estimator(dct: dict) -> BaseEstimator:
    DECODERS = {
        # 1: _json_to_sklearn_estimator_v1,  # Use turbo_broccoli v3
        2: _json_to_sklearn_estimator_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_sklearn_estimator_v2(dct: dict) -> BaseEstimator:
    bes = _all_base_estimators()
    cls = bes[dct["__type__"].split(".")[-1]]
    obj = cls(**dct["params"])
    for k, v in dct["attrs"].items():
        setattr(obj, k, v)
    return obj


# pylint: disable=missing-function-docstring
def from_json(dct: dict) -> BaseEstimator:
    raise_if_nodecode("sklearn")
    DECODERS = {  # Except sklearn estimators
        "sklearn.raw": _json_raw_to_sklearn,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        if type_name.startswith("sklearn.estimator."):
            return _json_to_sklearn_estimator(dct)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: BaseEstimator) -> dict:
    """
    Serializes a sklearn estimator into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    * if the object is an estimator:

            {
                "__type__": "sklearn.estimator.<CLASS NAME>",
                "__version__": 2,
                "params": <dict returned by get_params(deep=False)>,
                "attrs": {...}
            }

      where the `attrs` dict contains all the attributes of the estimator as
      specified in the sklearn API documentation.

    * otherwise:

            {
                "__type__": "sklearn.raw",
                "__version__": 2,
                "data": <uuid4>
            }

      where the UUID4 value points to an pickle file artifact.
    """

    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (t, _sklearn_to_raw) for t in _SUPPORTED_PICKLABLE_TYPES
    ]

    ENCODERS += [
        (BaseEstimator, _sklearn_estimator_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
