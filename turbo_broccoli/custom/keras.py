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
    "AvgPool1D": keras.layers.AvgPool1D,
    "AvgPool2D": keras.layers.AvgPool2D,
    "AvgPool3D": keras.layers.AvgPool3D,
    "BatchNormalization": keras.layers.BatchNormalization,
    "Bidirectional": keras.layers.Bidirectional,
    "CategoryEncoding": keras.layers.CategoryEncoding,
    "CenterCrop": keras.layers.CenterCrop,
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
    "Convolution1D": keras.layers.Convolution1D,
    "Convolution1DTranspose": keras.layers.Convolution1DTranspose,
    "Convolution2D": keras.layers.Convolution2D,
    "Convolution2DTranspose": keras.layers.Convolution2DTranspose,
    "Convolution3D": keras.layers.Convolution3D,
    "Convolution3DTranspose": keras.layers.Convolution3DTranspose,
    "Cropping1D": keras.layers.Cropping1D,
    "Cropping2D": keras.layers.Cropping2D,
    "Cropping3D": keras.layers.Cropping3D,
    "Dense": keras.layers.Dense,
    "DepthwiseConv1D": keras.layers.DepthwiseConv1D,
    "DepthwiseConv2D": keras.layers.DepthwiseConv2D,
    "Discretization": keras.layers.Discretization,
    "Dot": keras.layers.Dot,
    "Dropout": keras.layers.Dropout,
    "ELU": keras.layers.ELU,
    "EinsumDense": keras.layers.EinsumDense,
    "Embedding": keras.layers.Embedding,
    "Flatten": keras.layers.Flatten,
    "GRU": keras.layers.GRU,
    "GRUCell": keras.layers.GRUCell,
    "GaussianDropout": keras.layers.GaussianDropout,
    "GaussianNoise": keras.layers.GaussianNoise,
    "GlobalAveragePooling1D": keras.layers.GlobalAveragePooling1D,
    "GlobalAveragePooling2D": keras.layers.GlobalAveragePooling2D,
    "GlobalAveragePooling3D": keras.layers.GlobalAveragePooling3D,
    "GlobalAvgPool1D": keras.layers.GlobalAvgPool1D,
    "GlobalAvgPool2D": keras.layers.GlobalAvgPool2D,
    "GlobalAvgPool3D": keras.layers.GlobalAvgPool3D,
    "GlobalMaxPool1D": keras.layers.GlobalMaxPool1D,
    "GlobalMaxPool2D": keras.layers.GlobalMaxPool2D,
    "GlobalMaxPool3D": keras.layers.GlobalMaxPool3D,
    "GlobalMaxPooling1D": keras.layers.GlobalMaxPooling1D,
    "GlobalMaxPooling2D": keras.layers.GlobalMaxPooling2D,
    "GlobalMaxPooling3D": keras.layers.GlobalMaxPooling3D,
    "GroupNormalization": keras.layers.GroupNormalization,
    "GroupQueryAttention": keras.layers.GroupQueryAttention,
    "HashedCrossing": keras.layers.HashedCrossing,
    "Hashing": keras.layers.Hashing,
    "Identity": keras.layers.Identity,
    "Input": keras.layers.Input,
    "InputLayer": keras.layers.InputLayer,
    "InputSpec": keras.layers.InputSpec,
    "IntegerLookup": keras.layers.IntegerLookup,
    "LSTM": keras.layers.LSTM,
    "LSTMCell": keras.layers.LSTMCell,
    "Lambda": keras.layers.Lambda,
    "Layer": keras.layers.Layer,
    "LayerNormalization": keras.layers.LayerNormalization,
    "LeakyReLU": keras.layers.LeakyReLU,
    "Masking": keras.layers.Masking,
    "MaxPool1D": keras.layers.MaxPool1D,
    "MaxPool2D": keras.layers.MaxPool2D,
    "MaxPool3D": keras.layers.MaxPool3D,
    "MaxPooling1D": keras.layers.MaxPooling1D,
    "MaxPooling2D": keras.layers.MaxPooling2D,
    "MaxPooling3D": keras.layers.MaxPooling3D,
    "Maximum": keras.layers.Maximum,
    "MelSpectrogram": keras.layers.MelSpectrogram,
    "Minimum": keras.layers.Minimum,
    "MultiHeadAttention": keras.layers.MultiHeadAttention,
    "Multiply": keras.layers.Multiply,
    "Normalization": keras.layers.Normalization,
    "PReLU": keras.layers.PReLU,
    "Permute": keras.layers.Permute,
    "RNN": keras.layers.RNN,
    "RandomBrightness": keras.layers.RandomBrightness,
    "RandomContrast": keras.layers.RandomContrast,
    "RandomCrop": keras.layers.RandomCrop,
    "RandomFlip": keras.layers.RandomFlip,
    "RandomHeight": keras.layers.RandomHeight,
    "RandomRotation": keras.layers.RandomRotation,
    "RandomTranslation": keras.layers.RandomTranslation,
    "RandomWidth": keras.layers.RandomWidth,
    "RandomZoom": keras.layers.RandomZoom,
    "ReLU": keras.layers.ReLU,
    "RepeatVector": keras.layers.RepeatVector,
    "Rescaling": keras.layers.Rescaling,
    "Reshape": keras.layers.Reshape,
    "Resizing": keras.layers.Resizing,
    "SeparableConv1D": keras.layers.SeparableConv1D,
    "SeparableConv2D": keras.layers.SeparableConv2D,
    "SeparableConvolution1D": keras.layers.SeparableConvolution1D,
    "SeparableConvolution2D": keras.layers.SeparableConvolution2D,
    "SimpleRNN": keras.layers.SimpleRNN,
    "SimpleRNNCell": keras.layers.SimpleRNNCell,
    "Softmax": keras.layers.Softmax,
    "SpatialDropout1D": keras.layers.SpatialDropout1D,
    "SpatialDropout2D": keras.layers.SpatialDropout2D,
    "SpatialDropout3D": keras.layers.SpatialDropout3D,
    "SpectralNormalization": keras.layers.SpectralNormalization,
    "StackedRNNCells": keras.layers.StackedRNNCells,
    "StringLookup": keras.layers.StringLookup,
    "Subtract": keras.layers.Subtract,
    "TFSMLayer": keras.layers.TFSMLayer,
    "TextVectorization": keras.layers.TextVectorization,
    "ThresholdedReLU": keras.layers.ThresholdedReLU,
    "TimeDistributed": keras.layers.TimeDistributed,
    "TorchModuleWrapper": keras.layers.TorchModuleWrapper,
    "UnitNormalization": keras.layers.UnitNormalization,
    "UpSampling1D": keras.layers.UpSampling1D,
    "UpSampling2D": keras.layers.UpSampling2D,
    "UpSampling3D": keras.layers.UpSampling3D,
    "Wrapper": keras.layers.Wrapper,
    "ZeroPadding1D": keras.layers.ZeroPadding1D,
    "ZeroPadding2D": keras.layers.ZeroPadding2D,
    "ZeroPadding3D": keras.layers.ZeroPadding3D,
}

KERAS_LOSSES = {
    "BinaryCrossentropy": keras.losses.BinaryCrossentropy,
    "BinaryFocalCrossentropy": keras.losses.BinaryFocalCrossentropy,
    "CTC": keras.losses.CTC,
    "CategoricalCrossentropy": keras.losses.CategoricalCrossentropy,
    "CategoricalFocalCrossentropy": keras.losses.CategoricalFocalCrossentropy,
    "CategoricalHinge": keras.losses.CategoricalHinge,
    "CosineSimilarity": keras.losses.CosineSimilarity,
    "Hinge": keras.losses.Hinge,
    "Huber": keras.losses.Huber,
    "KLD": keras.losses.KLD,
    "KLDivergence": keras.losses.KLDivergence,
    "LogCosh": keras.losses.LogCosh,
    "Loss": keras.losses.Loss,
    "MAE": keras.losses.MAE,
    "MAPE": keras.losses.MAPE,
    "MSE": keras.losses.MSE,
    "MSLE": keras.losses.MSLE,
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError,
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError,
    "MeanSquaredError": keras.losses.MeanSquaredError,
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError,
    "Poisson": keras.losses.Poisson,
    "Reduction": keras.losses.Reduction,
    "SparseCategoricalCrossentropy": keras.losses.SparseCategoricalCrossentropy,
    "SquaredHinge": keras.losses.SquaredHinge,
}

KERAS_METRICS = {
    "AUC": keras.metrics.AUC,
    "Accuracy": keras.metrics.Accuracy,
    "BinaryAccuracy": keras.metrics.BinaryAccuracy,
    "BinaryCrossentropy": keras.metrics.BinaryCrossentropy,
    "BinaryIoU": keras.metrics.BinaryIoU,
    "CategoricalAccuracy": keras.metrics.CategoricalAccuracy,
    "CategoricalCrossentropy": keras.metrics.CategoricalCrossentropy,
    "CategoricalHinge": keras.metrics.CategoricalHinge,
    "CosineSimilarity": keras.metrics.CosineSimilarity,
    "F1Score": keras.metrics.F1Score,
    "FBetaScore": keras.metrics.FBetaScore,
    "FalseNegatives": keras.metrics.FalseNegatives,
    "FalsePositives": keras.metrics.FalsePositives,
    "Hinge": keras.metrics.Hinge,
    "IoU": keras.metrics.IoU,
    "KLDivergence": keras.metrics.KLDivergence,
    "LogCoshError": keras.metrics.LogCoshError,
    "Mean": keras.metrics.Mean,
    "MeanAbsoluteError": keras.metrics.MeanAbsoluteError,
    "MeanAbsolutePercentageError": keras.metrics.MeanAbsolutePercentageError,
    "MeanIoU": keras.metrics.MeanIoU,
    "MeanMetricWrapper": keras.metrics.MeanMetricWrapper,
    "MeanSquaredError": keras.metrics.MeanSquaredError,
    "MeanSquaredLogarithmicError": keras.metrics.MeanSquaredLogarithmicError,
    "Metric": keras.metrics.Metric,
    "OneHotIoU": keras.metrics.OneHotIoU,
    "OneHotMeanIoU": keras.metrics.OneHotMeanIoU,
    "Poisson": keras.metrics.Poisson,
    "Precision": keras.metrics.Precision,
    "PrecisionAtRecall": keras.metrics.PrecisionAtRecall,
    "R2Score": keras.metrics.R2Score,
    "Recall": keras.metrics.Recall,
    "RecallAtPrecision": keras.metrics.RecallAtPrecision,
    "RootMeanSquaredError": keras.metrics.RootMeanSquaredError,
    "SensitivityAtSpecificity": keras.metrics.SensitivityAtSpecificity,
    "SparseCategoricalAccuracy": keras.metrics.SparseCategoricalAccuracy,
    "SparseCategoricalCrossentropy": keras.metrics.SparseCategoricalCrossentropy,
    "SparseTopKCategoricalAccuracy": keras.metrics.SparseTopKCategoricalAccuracy,
    "SpecificityAtSensitivity": keras.metrics.SpecificityAtSensitivity,
    "SquaredHinge": keras.metrics.SquaredHinge,
    "Sum": keras.metrics.Sum,
    "TopKCategoricalAccuracy": keras.metrics.TopKCategoricalAccuracy,
    "TrueNegatives": keras.metrics.TrueNegatives,
    "TruePositives": keras.metrics.TruePositives,
}

KERAS_OPTIMIZERS = {
    "Adadelta": keras.optimizers.Adadelta,
    "Adafactor": keras.optimizers.Adafactor,
    "Adagrad": keras.optimizers.Adagrad,
    "Adam": keras.optimizers.Adam,
    "AdamW": keras.optimizers.AdamW,
    "Adamax": keras.optimizers.Adamax,
    "Ftrl": keras.optimizers.Ftrl,
    "Lion": keras.optimizers.Lion,
    "LossScaleOptimizer": keras.optimizers.LossScaleOptimizer,
    "Nadam": keras.optimizers.Nadam,
    "Optimizer": keras.optimizers.Optimizer,
    "RMSprop": keras.optimizers.RMSprop,
    "SGD": keras.optimizers.SGD,
}

KERAS_LEGACY_OPTIMIZERS = {
    "Adagrad": keras.optimizers.legacy.Adagrad,
    "Adam": keras.optimizers.legacy.Adam,
    "Ftrl": keras.optimizers.legacy.Ftrl,
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

        ```py
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

        ```py
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
