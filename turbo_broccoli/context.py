"""
A context object holds information about the (de)serialization process, such as
the current position in the document, output paths, etc.
"""

from os import environ as ENV
import tempfile
from pathlib import Path
from typing import Literal
from uuid import uuid4

from turbo_broccoli.utils import TypeIsNodecode


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
    it to `turbo_broccoli.from_json` or `turbo_broccoli.to_json`. For
    convenience, `turbo_broccoli.save_json` and `turbo_broccoli.load_json` take
    the context parameter's as kwargs.
    """

    json_path: str
    file_path: Path | None
    artifact_path: Path
    min_artifact_size: int = 8000
    nodecode_types: list[str]
    keras_format: str
    pandas_format: str
    pandas_kwargs: dict
    nacl_shared_key: bytes | None
    dataclass_types: dict[str, type]
    pytorch_module_types: dict[str, type]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        json_path: str = "$",
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
    ) -> None:
        """
        Args:
            json_path (str, optional): Current JSONpath
            file_path (str | Path | None, optional): Output JSON file path
            artifact_path (str | Path | None, optional): Artifact path.
                Defaults to the parent directory of `file_path`, or a new
                temporary directory if `file_path` is `None`.
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
        self.min_artifact_size = min_artifact_size or int(
            ENV.get("TB_MAX_NBYTES", 8000)
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

    def __truediv__(self, x: str | int) -> "Context":
        """
        Returns a copy of the current context but where the `json_path`
        attribute is `self.json_path + "." + str(x)`. Use this when you're
        going down the document.
        """
        return Context(  # TODO: define an __all__ variable
            json_path=self.json_path + "." + str(x),
            file_path=self.file_path,
            artifact_path=self.artifact_path,
            min_artifact_size=self.min_artifact_size,
            nodecode_types=self.nodecode_types,
            keras_format=self.keras_format,  # type: ignore
            pandas_format=self.pandas_format,  # type: ignore
            pandas_kwargs=self.pandas_kwargs,
            nacl_shared_key=self.nacl_shared_key,
            dataclass_types=self.dataclass_types,
            pytorch_module_types=self.pytorch_module_types,
        )

    def new_artifact_path(self) -> tuple[Path, str]:
        """Returns the path to a new artifact alongside the artifact's ID"""
        name = str(uuid4())
        return self.artifact_path / (name + ".tb"), name

    def raise_if_nodecode(self, type_name: str) -> None:
        """
        Raises a `TypeIsNodecode` exception if `type_name` is set to not be
        decoded in this context (see `nodecode_types` constructor argument).
        """
        if type_name in self.nodecode_types:
            raise TypeIsNodecode(type_name)
