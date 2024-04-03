"""User provided encoder/decoder methods."""

from typing import Any, Callable, TypeAlias

from .context import Context

Encoder: TypeAlias = Callable[[Any, Context], dict]
Decoder: TypeAlias = Callable[[dict, Context], Any]
ClassOrClasses: TypeAlias = (
    type | str | list[type | str] | tuple[type | str] | set[type | str]
)

ENCODERS: dict[str, Encoder] = {}
DECODERS: dict[str, Decoder] = {}


def register_encoder(encoder: Encoder | None, class_or_tuple: ClassOrClasses):
    """
    Register a custom encoder for the given type(s). An encoder is a function
    that takes an object and a `Context` and returns a dict with the following
    structure:

    ```py
    {
        "__type__": "user.<a_type_name>",
        ...
    }
    ```

    `a_type_name` should be a name used when registering the corresponding
    decoder.

    The dict can be/contain types that are not readily JSON serializable as
    long as they can be serialized by TurboBroccoli or by other user-provided
    encoders.

    Args:
        encoder (Encoder | None): If `None`, the encoder is removed.
        class_or_tuple (ClassOrClasses): See `ClassOrClasses`
    """
    if not isinstance(class_or_tuple, (list, tuple, set)):
        class_or_tuple = [class_or_tuple]
    if len(class_or_tuple) == 0:
        raise ValueError(
            "At least one class must be provided to register an encoder"
        )
    for cls in class_or_tuple:
        name = cls if isinstance(cls, str) else cls.__name__
        if encoder is None and name in ENCODERS:
            del ENCODERS[name]
        elif encoder is not None:
            ENCODERS[name] = encoder


def register_decoder(decoder: Decoder | None, class_or_tuple: ClassOrClasses):
    """
    Register a custom encoder for the given type(s). An decoder is a function
    that takes a dict and a `Context` and returns an object. The dict will have
    the form specified in `register_encoder`. It contains data that have
    already been deserialized by TurboBroccoli or by other user-provided
    decoders.

    Args:
        decoder (Decoder | None): If `None`, the decoder is removed.
        class_or_tuple (ClassOrClasses): See `ClassOrClasses`
    """
    if not isinstance(class_or_tuple, (list, tuple, set)):
        class_or_tuple = [class_or_tuple]
    if len(class_or_tuple) == 0:
        raise ValueError(
            "At least one class must be provided to register a decoder"
        )
    for cls in class_or_tuple:
        name = cls if isinstance(cls, str) else cls.__name__
        if decoder is None and name in DECODERS:
            del DECODERS[name]
        elif decoder is not None:
            DECODERS[name] = decoder
