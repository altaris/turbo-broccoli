"""pandas (de)serialization utilities."""
__docformat__ = "google"

import json
from typing import Any, Callable, List, Tuple
from uuid import uuid4

import pandas as pd
from turbo_broccoli.environment import (
    get_artifact_path,
    get_max_nbytes,
    get_pandas_format,
)


def _dataframe_to_json(df: pd.DataFrame) -> dict:
    """Converts a pandas dataframe into a JSON document."""
    # Sometimes column names are int, so cannot be used as keys in a JSON
    # document. Eventhough int column names are not supported, this is
    # future-proofing.
    dtypes = [(k, v.name) for k, v in df.dtypes.items()]
    if df.memory_usage(deep=True).sum() <= get_max_nbytes():
        return {
            "__type__": "dataframe",
            "__version__": 1,
            "data": json.loads(df.to_json(date_format="iso", date_unit="ns")),
            "dtypes": dtypes,
        }
    fmt = get_pandas_format()
    name = str(uuid4())
    path = get_artifact_path() / name
    if fmt in ["h5", "hdf"]:
        df.to_hdf(path, "main")
    else:
        getattr(df, f"to_{fmt}")(path)
    return {
        "__type__": "dataframe",
        "__version__": 1,
        "dtypes": dtypes,
        "id": name,
        "format": fmt,
    }


def _json_to_dataframe(dct: dict) -> pd.DataFrame:
    """
    Converts a JSON document to a pandas dataframe. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__pandas__`
    should not be present.
    """
    DECODERS = {
        1: _json_to_dataframe_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_dataframe_v1(dct: dict) -> pd.DataFrame:
    """
    Converts a JSON document to a pandas dataframe following the v1
    specification.
    """
    if "data" in dct:
        df = pd.read_json(json.dumps(dct["data"]))
    else:
        fmt = dct["format"]
        path = get_artifact_path() / dct["id"]
        if fmt in ["h5", "hdf"]:
            df = pd.read_hdf(path, "main")
        else:
            df = getattr(pd, f"read_{fmt}")(path)
    # Rename columns with non-string names
    # df.rename({str(d[0]): d[0] for d in dct["dtypes"]}, inplace=True)
    df = df.astype(
        {
            str(a): b
            for a, b in dct["dtypes"]
            if not str(b).startswith("datetime")
        }
    )
    for a, _ in filter(lambda x: x[1].startswith("datetime"), dct["dtypes"]):
        df[a] = pd.to_datetime(df[a]).dt.tz_localize(None)
    return df


def _json_to_series(dct: dict) -> pd.Series:
    """
    Converts a JSON document to a pandas series. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__pandas__`
    should not be present.
    """
    DECODERS = {
        1: _json_to_series_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_series_v1(dct: dict) -> pd.Series:
    """
    Converts a JSON document to a pandas series following the v1 specification.
    """
    return dct["data"][dct["name"]]


def _series_to_json(ser: pd.Series) -> dict:
    """Converts a pandas series into a JSON document."""
    name = ser.name if ser.name is not None else "main"
    return {
        "__type__": "series",
        "__version__": 1,
        "data": ser.to_frame(name=name),
        "name": name,
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a pandas object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__pandas__`.
    """
    DECODERS = {
        "dataframe": _json_to_dataframe,
        "series": _json_to_series,
    }
    try:
        return DECODERS[dct["__pandas__"]["__type__"]](dct["__pandas__"])
    except KeyError as exc:
        raise TypeError("Not a valid pandas document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a pandas object into JSON by cases. See the README for the
    precise list of supported types.

    The return dict has the following structure

        {
            "__pandas__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.

    * `pandas.DataFrame`: A dataframe is processed differently depending on its
      size and on the `TB_MAX_NBYTES` environment variable. If the dataframe is
      small, i.e. at most `TB_MAX_NBYTES` bytes, then it is directly stored in
      the resulting JSON document as

            {
                "__pandas__": {
                    "__type__": "dataframe",
                    "__version__": 1,
                    "data": {...},
                    "dtypes": [
                        [col1, dtype1],
                        [col2, dtype2],
                        ...
                    ],
                }
            }

      where `{...}` is the result of `pandas.DataFrame.to_json` (in `dict`
      form). On the other hand, the dataframe is too but, then its content is
      stored in an artifact, whose format follows the `TB_PANDAS_FORMAT`
      environment (HDF5 by default). Said file is saved to the
      path specified by the `TB_ARTIFACT_PATH` environment variable with a
      random UUID4 as filename. The resulting JSON document looks like

            {
                "__pandas__": {
                    "__type__": "dataframe",
                    "__version__": 1,
                    "dtypes": [
                        [col1, dtype1],
                        [col2, dtype2],
                        ...
                    ],
                    "id": <UUID4 str>,
                    "format": <str>
                }
            }

    * `pandas.Series`: A series will be converted to a dataframe before being
      serialized. The final document will look like this

            {
                "__pandas__": {
                    "__type__": "series",
                    "__version__": 1,
                    "data": {...},
                    "name": <str>,
                }
            }

      where `{...}` is the document of the dataframe'd series. So for example

            {
                "__pandas__": {
                    "__type__": "series",
                    "__version__": 1,
                    "data": {
                        "__type__": "dataframe",
                        "__version__": 1,
                        "id": <UUID4 str>,
                        "format": <str>,
                    },
                    "name": <str>,
                }
            }

      if the series is large.

    Warning:
        Series and column names must be strings!

    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (pd.DataFrame, _dataframe_to_json),
        (pd.Series, _series_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__pandas__": f(obj)}
    raise TypeError("Not a supported pandas type")
