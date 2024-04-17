"""NetworkX graph serialization and deserialization."""

from typing import Any, Callable, Tuple

import networkx as nx

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _graph_to_json(obj: nx.Graph, ctx: Context) -> dict:
    return {
        "__type__": "networkx.graph",
        "__version__": 1,
        "data": nx.adjacency_data(obj),
    }


def _json_to_graph(dct: dict, ctx: Context) -> nx.Graph:
    decoders = {1: _json_to_graph_v1}
    return decoders[dct["__version__"]](dct, ctx)


def _json_to_graph_v1(dct: dict, ctx: Context) -> nx.Graph:
    return nx.adjacency_graph(dct["data"])


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> nx.Graph:
    decoders = {
        "networkx.graph": _json_to_graph,
    }
    try:
        type_name = dct["__type__"]
        return decoders[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: nx.Graph, ctx: Context) -> dict:
    """
    Serializes a graph into JSON by cases. The return dict has the following
    structure

    ```py
    {
        "__type__": "networkx.graph",
        "__version__": 1,
        "data": {...}
    }
    ```

    where the `{...}` is produced by
    [`networkx.adjacency_data`](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.adjacency_data.html#adjacency-data).
    """
    encoders: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (nx.Graph, _graph_to_json),
    ]
    for t, f in encoders:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
