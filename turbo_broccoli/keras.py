"""keras (de)serialization utilities."""
__docformat__ = "google"

from functools import partial
from typing import Any, Callable, List, Tuple

from tensorflow import keras

KERAS_LAYERS = {
    "Activation": keras.layers.Activation,
    "ActivityRegularization": keras.layers.ActivityRegularization,
    "Add": keras.layers.Add,
    "AdditiveAttention": keras.layers.AdditiveAttention,
    "AlphaDropout": keras.layers.AlphaDropout,
    "Attention": keras.layers.Attention,
    "Average": keras.layers.Average,
    "AveragePooling1D": keras.layers.AveragePooling1D,
    "AveragePooling2D": keras.layers.AveragePooling2D,
    "AveragePooling3D": keras.layers.AveragePooling3D,
    "RNN": keras.layers.RNN,
    "BatchNormalization": keras.layers.BatchNormalization,
    "Bidirectional": keras.layers.Bidirectional,
    "Concatenate": keras.layers.Concatenate,
    "Conv1D": keras.layers.Conv1D,
    "Conv1DTranspose": keras.layers.Conv1DTranspose,
    "Conv2D": keras.layers.Conv2D,
    "Conv2DTranspose": keras.layers.Conv2DTranspose,
    "Conv3D": keras.layers.Conv3D,
    "Conv3DTranspose": keras.layers.Conv3DTranspose,
    "ConvLSTM1D": keras.layers.ConvLSTM1D,
    "ConvLSTM2D": keras.layers.ConvLSTM2D,
    "ConvLSTM3D": keras.layers.ConvLSTM3D,
    "Cropping1D": keras.layers.Cropping1D,
    "Cropping2D": keras.layers.Cropping2D,
    "Cropping3D": keras.layers.Cropping3D,
    "Dense": keras.layers.Dense,
    "DepthwiseConv2D": keras.layers.DepthwiseConv2D,
    "Dot": keras.layers.Dot,
    "Dropout": keras.layers.Dropout,
    "ELU": keras.layers.ELU,
    "Embedding": keras.layers.Embedding,
    "Flatten": keras.layers.Flatten,
    "GaussianDropout": keras.layers.GaussianDropout,
    "GaussianNoise": keras.layers.GaussianNoise,
    "GlobalAveragePooling1D": keras.layers.GlobalAveragePooling1D,
    "GlobalAveragePooling2D": keras.layers.GlobalAveragePooling2D,
    "GlobalAveragePooling3D": keras.layers.GlobalAveragePooling3D,
    "GlobalMaxPooling1D": keras.layers.GlobalMaxPooling1D,
    "GlobalMaxPooling2D": keras.layers.GlobalMaxPooling2D,
    "GlobalMaxPooling3D": keras.layers.GlobalMaxPooling3D,
    "GRU": keras.layers.GRU,
    "Lambda": keras.layers.Lambda,
    "LayerNormalization": keras.layers.LayerNormalization,
    "LeakyReLU": keras.layers.LeakyReLU,
    "LocallyConnected1D": keras.layers.LocallyConnected1D,
    "LocallyConnected2D": keras.layers.LocallyConnected2D,
    "LSTM": keras.layers.LSTM,
    "Masking": keras.layers.Masking,
    "Maximum": keras.layers.Maximum,
    "MaxPooling1D": keras.layers.MaxPooling1D,
    "MaxPooling2D": keras.layers.MaxPooling2D,
    "MaxPooling3D": keras.layers.MaxPooling3D,
    "Minimum": keras.layers.Minimum,
    "MultiHeadAttention": keras.layers.MultiHeadAttention,
    "Multiply": keras.layers.Multiply,
    "Permute": keras.layers.Permute,
    "PReLU": keras.layers.PReLU,
    "ReLU": keras.layers.ReLU,
    "RepeatVector": keras.layers.RepeatVector,
    "Reshape": keras.layers.Reshape,
    "SeparableConv1D": keras.layers.SeparableConv1D,
    "SeparableConv2D": keras.layers.SeparableConv2D,
    "SimpleRNN": keras.layers.SimpleRNN,
    "Softmax": keras.layers.Softmax,
    "SpatialDropout1D": keras.layers.SpatialDropout1D,
    "SpatialDropout2D": keras.layers.SpatialDropout2D,
    "SpatialDropout3D": keras.layers.SpatialDropout3D,
    "Subtract": keras.layers.Subtract,
    "ThresholdedReLU": keras.layers.ThresholdedReLU,
    "TimeDistributed": keras.layers.TimeDistributed,
    "UnitNormalization": keras.layers.UnitNormalization,
    "UpSampling1D": keras.layers.UpSampling1D,
    "UpSampling2D": keras.layers.UpSampling2D,
    "UpSampling3D": keras.layers.UpSampling3D,
    "ZeroPadding1D": keras.layers.ZeroPadding1D,
    "ZeroPadding2D": keras.layers.ZeroPadding2D,
    "ZeroPadding3D": keras.layers.ZeroPadding3D,
}

KERAS_LOSSES = {
    "BinaryCrossentropy": keras.losses.BinaryCrossentropy,
    "CategoricalCrossentropy": keras.losses.CategoricalCrossentropy,
    "CategoricalHinge": keras.losses.CategoricalHinge,
    "CosineSimilarity": keras.losses.CosineSimilarity,
    "Hinge": keras.losses.Hinge,
    "Huber": keras.losses.Huber,
    "KLDivergence": keras.losses.KLDivergence,
    "LogCosh": keras.losses.LogCosh,
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError,
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError,
    "MeanSquaredError": keras.losses.MeanSquaredError,
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError,
    "Poisson": keras.losses.Poisson,
    "SparseCategoricalCrossentropy": keras.losses.SparseCategoricalCrossentropy,
    "SquaredHinge": keras.losses.SquaredHinge,
}

KERAS_METRICS = {
    "Accuracy": keras.metrics.Accuracy,
    "BinaryAccuracy": keras.metrics.BinaryAccuracy,
    "CategoricalAccuracy": keras.metrics.CategoricalAccuracy,
    "SparseCategoricalAccuracy": keras.metrics.SparseCategoricalAccuracy,
    "TopKCategoricalAccuracy": keras.metrics.TopKCategoricalAccuracy,
    "SparseTopKCategoricalAccuracy": keras.metrics.SparseTopKCategoricalAccuracy,
    "BinaryCrossentropy": keras.metrics.BinaryCrossentropy,
    "CategoricalCrossentropy": keras.metrics.CategoricalCrossentropy,
    "SparseCategoricalCrossentropy": keras.metrics.SparseCategoricalCrossentropy,
    "KLDivergence": keras.metrics.KLDivergence,
    "Poisson": keras.metrics.Poisson,
    "MeanSquaredError": keras.metrics.MeanSquaredError,
    "RootMeanSquaredError": keras.metrics.RootMeanSquaredError,
    "MeanAbsoluteError": keras.metrics.MeanAbsoluteError,
    "MeanAbsolutePercentageError": keras.metrics.MeanAbsolutePercentageError,
    "MeanSquaredLogarithmicError": keras.metrics.MeanSquaredLogarithmicError,
    "CosineSimilarity": keras.metrics.CosineSimilarity,
    "LogCoshError": keras.metrics.LogCoshError,
    "AUC": keras.metrics.AUC,
    "Precision": keras.metrics.Precision,
    "Recall": keras.metrics.Recall,
    "TruePositives": keras.metrics.TruePositives,
    "TrueNegatives": keras.metrics.TrueNegatives,
    "FalsePositives": keras.metrics.FalsePositives,
    "FalseNegatives": keras.metrics.FalseNegatives,
    "PrecisionAtRecall": keras.metrics.PrecisionAtRecall,
    "SensitivityAtSpecificity": keras.metrics.SensitivityAtSpecificity,
    "SpecificityAtSensitivity": keras.metrics.SpecificityAtSensitivity,
    "MeanIoU": keras.metrics.MeanIoU,
    "Hinge": keras.metrics.Hinge,
    "SquaredHinge": keras.metrics.SquaredHinge,
    "CategoricalHinge": keras.metrics.CategoricalHinge,
}

KERAS_OPTIMIZERS = {
    "Adadelta": keras.optimizers.Adadelta,
    "Adagrad": keras.optimizers.Adagrad,
    "Adam": keras.optimizers.Adam,
    "Adamax": keras.optimizers.Adamax,
    "Ftrl": keras.optimizers.Ftrl,
    "Nadam": keras.optimizers.Nadam,
    "RMSprop": keras.optimizers.RMSprop,
    "SGD": keras.optimizers.SGD,
}


def _json_to_layer(dct: dict) -> Any:
    """Converts a JSON document to a serializable keras object."""
    DECODERS = {
        1: _json_to_layer_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_layer_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a keras layer object following the v1
    specification.
    """
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_LAYERS,
    )


def _json_to_loss(dct: dict) -> Any:
    """Converts a JSON document to a serializable keras object."""
    DECODERS = {
        1: _json_to_loss_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_loss_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a keras loss object following the v1
    specification.
    """
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_LOSSES,
    )


def _json_to_metric(dct: dict) -> Any:
    """Converts a JSON document to a serializable keras object."""
    DECODERS = {
        1: _json_to_metric_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_metric_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a keras metric object following the v1
    specification.
    """
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_METRICS,
    )


def _json_to_model(dct: dict) -> Any:
    """Converts a JSON document to a serializable keras object."""
    DECODERS = {
        1: _json_to_model_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_model_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a keras model object following the v1
    specification.
    """
    model = keras.models.model_from_config(dct["model"])
    model.set_weights(dct["weights"])
    kwargs = {"metrics": dct["metrics"]}
    for k in ["loss", "optimizer"]:
        if dct.get(k) is not None:
            kwargs[k] = dct[k]
    model.compile(**kwargs)
    return model


def _json_to_optimizer(dct: dict) -> Any:
    """Converts a JSON document to a serializable keras object."""
    DECODERS = {
        1: _json_to_optimizer_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_optimizer_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a keras optimizer object following the v1
    specification.
    """
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_OPTIMIZERS,
    )


def _generic_keras_to_json(obj: Any, type_: str) -> dict:
    """Serializes a keras object using `keras.utils.serialize_keras_object`"""
    return {
        "__type__": type_,
        "__version__": 1,
        "data": keras.utils.serialize_keras_object(obj),
    }


def _model_to_json(model: keras.Model) -> dict:
    """Serializes a keras model"""
    return {
        "__type__": "model",
        "__version__": 1,
        "loss": getattr(model, "loss", None),
        "metrics": getattr(model, "metrics", []),
        "model": keras.utils.serialize_keras_object(model),
        "optimizer": getattr(model, "optimizer", None),
        "weights": model.weights,
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a Keras object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__keras__`.
    """
    DECODERS = {
        "model": _json_to_model,  # must be first!
        "layer": _json_to_layer,
        "loss": _json_to_loss,
        "metric": _json_to_metric,
        "optimizer": _json_to_optimizer,
    }
    try:
        return DECODERS[dct["__keras__"]["__type__"]](dct["__keras__"])
    except KeyError as exc:
        raise TypeError("Not a valid Keras document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a tensorflow object into JSON by cases. See the README for the
    precise list of supported types.

    The return dict has the following structure

        {
            "__keras__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`. Most keras object will simply be
    serialized using `keras.utils.serialize_keras_object`. Here are the exceptions:

    * `keras.Model` (the model must have weights)

            {
                "__keras__": {
                    "__type__": "model",
                    "__version__": 1,
                    "loss": {...} or null,
                    "metrics": [...],
                    "model": {...},
                    "optimizer": {...} or null,
                    "weights": [...],
                },
            }

    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (keras.Model, _model_to_json),  # must be first
        (keras.layers.Layer, partial(_generic_keras_to_json, type_="layer")),
        (keras.losses.Loss, partial(_generic_keras_to_json, type_="loss")),
        (
            keras.metrics.Metric,
            partial(_generic_keras_to_json, type_="metric"),
        ),
        (
            keras.optimizers.Optimizer,
            partial(_generic_keras_to_json, type_="optimizer"),
        ),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__keras__": f(obj)}
    raise TypeError("Not a supported Keras type")