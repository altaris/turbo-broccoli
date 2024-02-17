"""keras (de)serialization utilities."""

from functools import partial
from typing import Any, Callable, Tuple

from tensorflow import keras  # pylint: disable=no-name-in-module

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported

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
    "AUC": keras.metrics.AUC,
    "BinaryAccuracy": keras.metrics.BinaryAccuracy,
    "BinaryCrossentropy": keras.metrics.BinaryCrossentropy,
    "CategoricalAccuracy": keras.metrics.CategoricalAccuracy,
    "CategoricalCrossentropy": keras.metrics.CategoricalCrossentropy,
    "CategoricalHinge": keras.metrics.CategoricalHinge,
    "CosineSimilarity": keras.metrics.CosineSimilarity,
    "FalseNegatives": keras.metrics.FalseNegatives,
    "FalsePositives": keras.metrics.FalsePositives,
    "Hinge": keras.metrics.Hinge,
    "KLDivergence": keras.metrics.KLDivergence,
    "LogCoshError": keras.metrics.LogCoshError,
    "Mean": keras.metrics.Mean,
    "MeanAbsoluteError": keras.metrics.MeanAbsoluteError,
    "MeanAbsolutePercentageError": keras.metrics.MeanAbsolutePercentageError,
    "MeanIoU": keras.metrics.MeanIoU,
    "MeanSquaredError": keras.metrics.MeanSquaredError,
    "MeanSquaredLogarithmicError": keras.metrics.MeanSquaredLogarithmicError,
    "Poisson": keras.metrics.Poisson,
    "Precision": keras.metrics.Precision,
    "PrecisionAtRecall": keras.metrics.PrecisionAtRecall,
    "Recall": keras.metrics.Recall,
    "RootMeanSquaredError": keras.metrics.RootMeanSquaredError,
    "SensitivityAtSpecificity": keras.metrics.SensitivityAtSpecificity,
    "SparseCategoricalAccuracy": keras.metrics.SparseCategoricalAccuracy,
    "SparseCategoricalCrossentropy": keras.metrics.SparseCategoricalCrossentropy,
    "SparseTopKCategoricalAccuracy": keras.metrics.SparseTopKCategoricalAccuracy,
    "SpecificityAtSensitivity": keras.metrics.SpecificityAtSensitivity,
    "SquaredHinge": keras.metrics.SquaredHinge,
    "TopKCategoricalAccuracy": keras.metrics.TopKCategoricalAccuracy,
    "TrueNegatives": keras.metrics.TrueNegatives,
    "TruePositives": keras.metrics.TruePositives,
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

KERAS_LEGACY_OPTIMIZERS = {
    "Adadelta": keras.optimizers.legacy.Adadelta,
    "Adagrad": keras.optimizers.legacy.Adagrad,
    "Adam": keras.optimizers.legacy.Adam,
    "Adamax": keras.optimizers.legacy.Adamax,
    "Ftrl": keras.optimizers.legacy.Ftrl,
    "Nadam": keras.optimizers.legacy.Nadam,
    "Optimizer": keras.optimizers.legacy.Optimizer,
    "RMSprop": keras.optimizers.legacy.RMSprop,
    "SGD": keras.optimizers.legacy.SGD,
}


def _json_to_layer(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        2: _json_to_layer_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_layer_v2(dct: dict, ctx: Context) -> Any:
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_LAYERS,
    )


def _json_to_loss(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        2: _json_to_loss_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_loss_v2(dct: dict, ctx: Context) -> Any:
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_LOSSES,
    )


def _json_to_metric(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        2: _json_to_metric_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_metric_v2(dct: dict, ctx: Context) -> Any:
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_METRICS,
    )


def _json_to_model(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        5: _json_to_model_v5,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_model_v5(dct: dict, ctx: Context) -> Any:
    if "model" in dct:
        model = keras.models.model_from_config(dct["model"])
        model.set_weights(dct["weights"])
        kwargs = {"metrics": dct["metrics"]}
        for k in ["loss", "optimizer"]:
            if dct.get(k) is not None:
                kwargs[k] = dct[k]
        model.compile(**kwargs)
        return model
    path = (
        ctx.id_to_artifact_path(dct["id"], extension="keras")
        if ctx.keras_format == "keras"
        else ctx.id_to_artifact_path(dct["id"])
    )
    return keras.models.load_model(path)


def _json_to_optimizer(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        2: _json_to_optimizer_v2,
        3: _json_to_optimizer_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_optimizer_v2(dct: dict, ctx: Context) -> Any:
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=KERAS_OPTIMIZERS,
    )


def _json_to_optimizer_v3(dct: dict, ctx: Context) -> Any:
    return keras.utils.deserialize_keras_object(
        dct["data"],
        module_objects=(
            KERAS_LEGACY_OPTIMIZERS if dct["legacy"] else KERAS_OPTIMIZERS
        ),
    )


def _generic_to_json(
    obj: Any,
    ctx: Context,
    *,
    type_: str,
) -> dict:
    return {
        "__type__": "keras." + type_,
        "__version__": 2,
        "data": keras.utils.serialize_keras_object(obj),
    }


def _model_to_json(model: keras.Model, ctx: Context) -> dict:
    if ctx.keras_format == "json":
        return {
            "__type__": "keras.model",
            "__version__": 5,
            "loss": getattr(model, "loss", None),
            "metrics": getattr(model, "metrics", []),
            "model": keras.utils.serialize_keras_object(model),
            "optimizer": getattr(model, "optimizer", None),
            "weights": model.weights,
        }
    if ctx.keras_format == "keras":
        path, name = ctx.new_artifact_path(extension="keras")
    else:
        path, name = ctx.new_artifact_path()
    model.save(path, save_format=ctx.keras_format)
    return {
        "__type__": "keras.model",
        "__version__": 5,
        "format": ctx.keras_format,
        "id": name,
    }


def _optimizer_to_json(obj: Any, ctx: Context) -> dict:
    return {
        "__type__": "keras.optimizer",
        "__version__": 3,
        "data": keras.utils.serialize_keras_object(obj),
        "legacy": isinstance(obj, keras.optimizers.legacy.Optimizer),
    }


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "keras.model": _json_to_model,  # must be first!
        "keras.layer": _json_to_layer,
        "keras.loss": _json_to_loss,
        "keras.metric": _json_to_metric,
        "keras.optimizer": _json_to_optimizer,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a tensorflow object into JSON by cases. See the README for the
    precise list of supported types. Most keras object will simply be
    serialized using `keras.utils.serialize_keras_object`. Here are the
    exceptions:

    - `keras.Model` (the model must have weights). If `TB_KERAS_FORMAT` is
      `json`, the document will look like

        ```json
        {

            "__type__": "keras.model",
            "__version__": 5,
            "loss": {...} or null,
            "metrics": [...],
            "model": {...},
            "optimizer": {...} or null,
            "weights": [...],
        }
        ```

      if `TB_KERAS_FORMAT` is `h5` or `tf`, the document will look like

        ```json
        {

            "__type__": "keras.model",
            "__version__": 5,
            "format": <str>,
            "id": <uuid4>
        }
        ```

      where `id` points to an artifact. Note that if the keras saving format is
      `keras`, the artifact will have the `.keras` extension instead of the
      usual `.tb`. Tensorflow/keras [forces this
      behaviour](https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model).

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (keras.Model, _model_to_json),  # must be first
        (keras.metrics.Metric, partial(_generic_to_json, type_="metric")),
        (keras.layers.Layer, partial(_generic_to_json, type_="layer")),
        (keras.losses.Loss, partial(_generic_to_json, type_="loss")),
        (keras.optimizers.Optimizer, _optimizer_to_json),
        (keras.optimizers.legacy.Optimizer, _optimizer_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
