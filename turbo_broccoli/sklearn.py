"""Scikit-learn estimators"""
__docformat__ = "google"

import re
from typing import Any, Callable, Dict, List, Tuple

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


def _all_base_estimators() -> Dict[str, type]:
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
    """Converts a sklearn estimator into a JSON document."""
    r = re.compile(r"\w[\w_]*[^_]_")
    return {
        "__type__": "estimator",
        "cls": obj.__class__.__name__,
        "__version__": 1,
        "params": obj.get_params(deep=False),
        "attrs": {k: v for k, v in obj.__dict__.items() if r.match(k)},
    }


def _sklearn_tree_to_json(obj: Tree) -> dict:
    """Converts a sklearn Tree object into a JSON document."""
    return {
        "__type__": "tree",
        "__version__": 1,
        **{a: getattr(obj, a) for a in _SKLEARN_TREE_ATTRIBUTES},
    }


def _json_to_sklearn_estimator(dct: dict) -> BaseEstimator:
    """
    Converts a JSON document to a sklearn estimator. See `to_json` for the
    specification `dct` is expected to follow. Note that the key
    `__sklearn__` should not be present.
    """
    DECODERS = {
        1: _json_to_sklearn_estimator_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_sklearn_estimator_v1(dct: dict) -> BaseEstimator:
    """
    Converts a JSON document to a sklearn estimator following the v1
    specification.
    """
    bes = _all_base_estimators()
    cls = bes[dct["cls"]]
    obj = cls(**dct["params"])
    for k, v in dct["attrs"].items():
        setattr(obj, k, v)
    return obj


def _json_to_sklearn_tree(dct: dict) -> BaseEstimator:
    """
    Converts a JSON document to a sklearn tree object. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__sklearn__`
    should not be present.
    """
    DECODERS = {
        1: _json_to_sklearn_tree_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_sklearn_tree_v1(dct: dict) -> BaseEstimator:
    """
    Converts a JSON document to a sklearn tree following the v1 specification.

    See also:
        * https://github.com/scikit-learn/scikit-learn/blob/9aaed498795f68e5956ea762fef9c440ca9eb239/sklearn/tree/_tree.pxd
        * https://github.com/scikit-learn/scikit-learn/blob/9aaed498795f68e5956ea762fef9c440ca9eb239/sklearn/tree/_classes.py#L349
    """
    obj = Tree(dct["n_features"], dct["n_classes"], dct["n_outputs"])
    for k in _SKLEARN_TREE_ATTRIBUTES:
        setattr(obj, k, dct[k])
    return obj


def from_json(dct: dict) -> BaseEstimator:
    """
    Deserializes a dict into a sklearn estimator. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__sklearn__`.
    """
    DECODERS = {
        "estimator": _json_to_sklearn_estimator,
        "tree": _json_to_sklearn_tree,
    }
    try:
        return DECODERS[dct["__sklearn__"]["__type__"]](dct["__sklearn__"])
    except KeyError as exc:
        raise TypeError("Not a valid sklearn document") from exc


def to_json(obj: BaseEstimator) -> dict:
    """
    Serializes a sklearn estimator into JSON by cases. See the README for the
    precise list of supported types.

    The return dict has the following structure

        {
            "__sklearn__": {
                "__type__": "estimator",
                "__version__": 1,
                "cls": <class name>,
                "params": <dict returned by get_params(deep=False)>,
                "attrs": {...}
            },
        }

    where the `attrs` dict contains all the attributes of the estimator as
    specified in the sklearn API documentation.
    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (BaseEstimator, _sklearn_estimator_to_json),
        (Tree, _sklearn_tree_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__sklearn__": f(obj)}
    raise TypeError("Not a supported sklearn type")
