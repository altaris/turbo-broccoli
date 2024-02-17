"""
A context object holds information about the (de)serialization process, such as
the current position in the document, output paths, etc.
"""

import tempfile
from os import environ as ENV
from pathlib import Path
from typing import Literal
from uuid import uuid4

from turbo_broccoli.exceptions import TypeIsNodecode


def _list_of_types_to_dict(lot: list[type]) -> dict[str, type]:
    """
    Converts a list of types `[T1, T2, ...]` to a dict that looks like `{"T1":
    T1, "T2": T2, ...}`.
    """
    return {t.__name__: t for t in lot}


# pylint: disable=too-many-instance-attributes
class Context:
    """
    (De)Serialization context, which is an object that contains various
    information and parameters about the ongoing operation. If you want your
    (de)serialization to behave a certain way, create a context object and pass
    it to
    [`turbo_broccoli.to_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#to_json)
    or
    [`turbo_broccoli.from_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#from_json).
    For convenience,
    [`turbo_broccoli.save_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#save_json)
    and
    [`turbo_broccoli.load_json`](https://altaris.github.io/turbo-broccoli/turbo_broccoli/turbo_broccoli.html#load_json)
    take the context parameter's as kwargs.
    """

    artifact_path: Path
    dataclass_types: dict[str, type]
    file_path: Path | None
    json_path: str
    keras_format: str
    min_artifact_size: int = 8000
    nacl_shared_key: bytes | None
    nodecode_types: list[str]
    pandas_format: str
    pandas_kwargs: dict
    pytorch_module_types: dict[str, type]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        file_path: str | Path | None = None,
        artifact_path: str | Path | None = None,
        min_artifact_size: int | None = None,
        nodecode_types: list[str] | None = None,
        keras_format: Literal["keras", "tf", "h5"] | None = None,
        pandas_format: (
            Literal[
                "csv",
                "excel",
                "feather",
                "html",
                "json",
                "latex",
                "orc",
                "parquet",
                "pickle",
                "sql",
                "stata",
                "xml",
            ]
            | None
        ) = None,
        pandas_kwargs: dict | None = None,
        nacl_shared_key: bytes | None = None,
        dataclass_types: dict[str, type] | list[type] | None = None,
        pytorch_module_types: dict[str, type] | list[type] | None = None,
        json_path: str = "$",
    ) -> None:
        """
        Args:
            file_path (str | Path | None, optional): Output JSON file path.
            artifact_path (str | Path | None, optional): Artifact path.
                Defaults to the parent directory of `file_path`, or a new
                temporary directory if `file_path` is `None`.
            min_artifact_size (int, optional): Byte strings (and everything
                that serialize to byte strings such as numpy arrays) larget
                than this will be stored in artifact rather than be embedded in
                the output JSON string/file.
            nodecode_types (list[str], optional): List of type names which
                shall be deserialized to `None` rather than their true value.
                See
                [`TB_NODECODE`](https://altaris.github.io/turbo-broccoli/turbo_broccoli.html#environment-variables)
            keras_format ("keras", "tf", "h5", optional): Format for Keras
                artifacts
            pandas_format ("csv", "excel", "feather", "html", "json", "latex",
                "orc", "parquet", "pickle", "sql", "stata", "xml", optional):
                Format for pandas artifacts
            pandas_kwargs (dict, optional): kwargs to forward to the pandas
                `to_*` and `read_*` function. For example, if
                `pandas_format="parquet"`, then the content of `pandas.kwargs`
                will be forwarded to
                [`pandas.DataFrame.to_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)
                and
                [`pandas.read_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html)
            nacl_shared_key (bytes, optional): PyNaCl shared key. See also
                [PyNaCl's
                documentation](https://pynacl.readthedocs.io/en/latest/secret/#key)
            dataclass_types (dict[str, type] | list[type], optional): List of
                dataclass types for deserialization. See the
                [README](https://altaris.github.io/turbo-broccoli/turbo_broccoli.html#supported-types).
            pytorch_module_types (dict[str, type] | list[type], optional): List
                of pytorch module types for deserialization. See the
                [README](https://altaris.github.io/turbo-broccoli/turbo_broccoli.html#supported-types).
            json_path (str, optional): Current JSONpath. Don't use.
        """
        self.json_path = json_path
        self.file_path = (
            Path(file_path) if isinstance(file_path, str) else file_path
        )
        if artifact_path is None:
            if p := ENV.get("TB_ARTIFACT_PATH"):
                self.artifact_path = Path(p)
            else:
                self.artifact_path = (
                    self.file_path.parent
                    if self.file_path is not None
                    else Path(tempfile.mkdtemp())
                )
        else:
            self.artifact_path = Path(artifact_path)
        self.min_artifact_size = (
            min_artifact_size
            if min_artifact_size is not None
            else int(ENV.get("TB_MAX_NBYTES", 8000))
        )
        self.nodecode_types = nodecode_types or ENV.get(
            "TB_NODECODE", ""
        ).split(",")
        self.keras_format = keras_format or str(
            ENV.get("TB_KERAS_FORMAT", "tf")
        )
        self.pandas_format = pandas_format or str(
            ENV.get("TB_PANDAS_FORMAT", "csv")
        )
        self.pandas_kwargs = pandas_kwargs or {}
        if isinstance(nacl_shared_key, bytes):
            self.nacl_shared_key = nacl_shared_key
        elif "TB_SHARED_KEY" in ENV:
            self.nacl_shared_key = str(ENV["TB_SHARED_KEY"]).encode("utf-8")
        else:
            self.nacl_shared_key = None
        self.dataclass_types = (
            _list_of_types_to_dict(dataclass_types)
            if isinstance(dataclass_types, list)
            else (dataclass_types or {})
        )
        self.pytorch_module_types = (
            _list_of_types_to_dict(pytorch_module_types)
            if isinstance(pytorch_module_types, list)
            else (pytorch_module_types or {})
        )

    def __repr__(self) -> str:
        fp, ap = str(self.file_path), str(self.artifact_path)
        return (
            f"Context(file_path={fp}, artifact_path={ap}, "
            f"json_path={self.json_path})"
        )

    def __truediv__(self, x: str | int) -> "Context":
        """
        Returns a copy of the current context but where the `json_path`
        attribute is `self.json_path + "." + str(x)`. Use this when you're
        going down the document.
        """
        kwargs = self.__dict__.copy()
        kwargs["json_path"] = self.json_path + "." + str(x)
        return Context(**kwargs)

    def id_to_artifact_path(self, art_id: str, extension: str = "tb") -> Path:
        """
        Takes an artifact id (which is an UUID4 string) and returns the
        absolute path to the corresponding artifact file.
        """
        art_fn = art_id + "." + extension
        if self.file_path is not None:
            art_fn = self.file_path.stem + "." + art_fn
        return self.artifact_path / art_fn

    def new_artifact_path(self, extension: str = "tb") -> tuple[Path, str]:
        """Returns the path to a new artifact alongside the artifact's ID"""
        art_id = str(uuid4())
        return self.id_to_artifact_path(art_id, extension), art_id

    def raise_if_nodecode(self, type_name: str) -> None:
        """
        Raises a `TypeIsNodecode` exception if `type_name` or any prefix is set
        to not be decoded in this context (see `nodecode_types` constructor
        argument).

        For example, if `type_name` is `a.b.c`, then this method raises
        `TypeIsNodecode` if either `a`, `a.b`, or `a.b.c` is set as a nodecode
        type.
        """
        parts = type_name.split(".")
        for i in range(1, len(parts) + 1):
            t = ".".join(parts[:i])
            if t in self.nodecode_types:
                raise TypeIsNodecode(t)
