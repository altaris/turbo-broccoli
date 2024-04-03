"""pandas (de)serialization utilities."""

import json
from io import StringIO
from typing import Any, Callable, Tuple

import pandas as pd

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _dataframe_to_json(df: pd.DataFrame, ctx: Context) -> dict:
    dtypes = [[str(k), v.name] for k, v in df.dtypes.items()]
    if df.memory_usage(deep=True).sum() <= ctx.min_artifact_size:
        return {
            "__type__": "pandas.dataframe",
            "__version__": 2,
            "data": json.loads(df.to_json(date_format="iso", date_unit="ns")),
            "dtypes": dtypes,
        }
    fmt = ctx.pandas_format
    path, name = ctx.new_artifact_path()
    getattr(df, f"to_{fmt}")(path, **ctx.pandas_kwargs)
    return {
        "__type__": "pandas.dataframe",
        "__version__": 2,
        "dtypes": dtypes,
        "id": name,
        "format": fmt,
    }


def _json_to_dataframe(dct: dict, ctx: Context) -> pd.DataFrame:
    DECODERS = {
        2: _json_to_dataframe_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_dataframe_v2(dct: dict, ctx: Context) -> pd.DataFrame:
    if "data" in dct:
        df = pd.read_json(StringIO(json.dumps(dct["data"])))
    else:
        fmt = dct["format"]
        path = ctx.id_to_artifact_path(dct["id"])
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


def _json_to_series(dct: dict, ctx: Context) -> pd.Series:
    ctx.raise_if_nodecode("pandas.dataframe")
    DECODERS = {
        2: _json_to_series_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_series_v2(dct: dict, ctx: Context) -> pd.Series:
    return dct["data"][dct["name"]]


def _series_to_json(ser: pd.Series, ctx: Context) -> dict:
    name = ser.name if ser.name is not None else "main"
    return {
        "__type__": "pandas.series",
        "__version__": 2,
        "data": ser.to_frame(name=name),
        "name": name,
    }


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "pandas.dataframe": _json_to_dataframe,
        "pandas.series": _json_to_series,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a pandas object into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - `pandas.DataFrame`: A dataframe is processed differently depending on its
      size and on the `TB_MAX_NBYTES` environment variable. If the dataframe is
      small, i.e. at most `TB_MAX_NBYTES` bytes, then it is directly stored in
      the resulting JSON document as

        ```py
        {
            "__type__": "pandas.dataframe",
            "__version__": 2,
            "data": {...},
            "dtypes": [
                [col1, dtype1],
                [col2, dtype2],
                ...
            ],
        }
        ```

      where `{...}` is the result of `pandas.DataFrame.to_json` (in `dict`
      form). On the other hand, the dataframe is too large, then its content is
      stored in an artifact, whose format follows the `TB_PANDAS_FORMAT`
      environment (CSV by default). The resulting JSON document looks like

        ```py
        {
            "__type__": "pandas.dataframe",
            "__version__": 2,
            "dtypes": [
                [col1, dtype1],
                [col2, dtype2],
                ...
            ],
            "id": <UUID4 str>,
            "format": <str>
        }
        ```

    - `pandas.Series`: A series will be converted to a dataframe before being
      serialized. The final document will look like this

        ```py
        {
            "__type__": "pandas.series",
            "__version__": 2,
            "data": {...},
            "name": <str>,
        }
        ```

      where `{...}` is the document of the dataframe'd series, see above.

    Warning:
        Series and column names must be strings!

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (pd.DataFrame, _dataframe_to_json),
        (pd.Series, _series_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
