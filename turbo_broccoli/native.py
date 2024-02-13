"""
"Native" saving utilities:
* `turbo_broccoli.native.save` takes a serializable/dumpable object and
  a path, and uses the file extension to choose the correct way to save the
  object;
* `turbo_broccoli.native.load` does the opposite.
"""

# pylint: disable=unused-argument
# pylint: disable=import-outside-toplevel

from functools import partial
from pathlib import Path
from typing import Any, Callable

try:
    import safetensors  # pylint: disable=unused-import

    HAS_SAFETENSORS = True
except ModuleNotFoundError:
    HAS_SAFETENSORS = False

from turbo_broccoli.custom import (
    HAS_KERAS,
    HAS_NUMPY,
    HAS_PANDAS,
    HAS_PYTORCH,
    HAS_TENSORFLOW,
)
from turbo_broccoli.turbo_broccoli import load_json, save_json


def _is_dict_of(obj: Any, value_type: type, key_type: type = str) -> bool:
    """Returns true if `obj` is a `dict[key_type, value_type]`"""
    return (
        isinstance(obj, dict)
        and all(isinstance(k, key_type) for k in obj.keys())
        and all(isinstance(v, value_type) for v in obj.values())
    )


def _load_csv(path: str | Path, **kwargs) -> Any:
    if not HAS_PANDAS:
        _raise_package_not_installed("pandas", "csv")
    import pandas as pd

    df = pd.read_csv(path, **kwargs)
    if "Unnamed: 0" in df.columns:
        df.drop(["Unnamed: 0"], axis=1, inplace=True)
    return df


def _load_keras(path: str | Path, **kwargs) -> Any:
    if not HAS_KERAS:
        _raise_package_not_installed("keras", "keras")

    import keras

    return keras.saving.load_model(path, **kwargs)


def _load_np(path: str | Path, **kwargs) -> Any:
    if not HAS_NUMPY:
        _raise_package_not_installed("numpy", ".npy/.npz")
    import numpy as np

    return np.load(path, **kwargs)


def _load_pq(path: str | Path, **kwargs) -> Any:
    if not HAS_PANDAS:
        _raise_package_not_installed("pandas", ".parquet/.pq")
    import pandas as pd

    df = pd.read_parquet(path, **kwargs)
    return df


def _load_pt(path: str | Path, **kwargs) -> Any:
    if not HAS_PYTORCH:
        _raise_package_not_installed("torch", "pt")
    import torch

    return torch.load(path, **kwargs)


def _load_st(path: str | Path, **kwargs) -> Any:
    if not HAS_SAFETENSORS:
        _raise_package_not_installed("safetensors", ".safetensors/.st")
    from safetensors import numpy as st

    return st.load_file(path, **kwargs)


def _raise_package_not_installed(package_name: str, extension: str):
    """
    Raises a `RuntimeError` with a templated error message

    Args:
        package_name (str): e.g. "numpy"
        extension (str): e.g. "npy"
    """
    if extension[0] != ".":
        extension = "." + extension
    raise RuntimeError(
        f"Cannot create or load `{extension}` file because {package_name} is "
        f"not installed. You can install {package_name} by running "
        f"python3 -m pip install {package_name}"
    )


def _raise_wrong_type(path: str | Path, obj_needs_to_be_a: str):
    """
    Raises a `TypeError` with a templated error message

    Args:
        path (str | Path): Path where the file should have been saved
        extension (str): "pandas DataFrame or Series"
    """
    raise TypeError(
        f"Could not save object to '{path}': object needs to be a "
        + obj_needs_to_be_a
    )


def _save_csv(obj: Any, path: str | Path, **kwargs) -> None:
    if not HAS_PANDAS:
        _raise_package_not_installed("pandas", "csv")
    import pandas as pd

    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        _raise_wrong_type(path, "pandas DataFrame or Series")
    obj.to_csv(path, **kwargs)


def _save_keras(obj: Any, path: str | Path, **kwargs) -> None:
    if not HAS_KERAS:
        _raise_package_not_installed("keras", "keras")

    import keras

    if not isinstance(obj, keras.Model):
        _raise_wrong_type(path, "keras model")
    keras.saving.save_model(obj, path, **kwargs)


def _save_npy(obj: Any, path: str | Path, **kwargs) -> None:
    if not HAS_NUMPY:
        _raise_package_not_installed("numpy", "npy")
    import numpy as np

    if not isinstance(obj, np.ndarray):
        _raise_wrong_type(path, "numpy array")
    np.save(str(path), obj, **kwargs)


def _save_npz(obj: Any, path: str | Path, **kwargs) -> None:
    if not HAS_NUMPY:
        _raise_package_not_installed("numpy", "npz")
    import numpy as np

    if not _is_dict_of(obj, np.ndarray):
        _raise_wrong_type(path, "dict of numpy arrays")
    np.savez(str(path), **obj, **kwargs)


def _save_pq(obj: Any, path: str | Path, **kwargs) -> None:
    if not HAS_PANDAS:
        _raise_package_not_installed("pandas", ".parquet/.pq")
    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        _raise_wrong_type(path, "pandas DataFrame")
    obj.to_parquet(path, **kwargs)


def _save_pt(obj: Any, path: str | Path, **kwargs) -> None:
    if not HAS_PYTORCH:
        _raise_package_not_installed("torch", "pt")
    import torch

    if not (isinstance(obj, torch.Tensor) or _is_dict_of(obj, torch.Tensor)):
        _raise_wrong_type(path, "torch tensor or a dict of torch tensors")
    torch.save(obj, path, **kwargs)


def _save_st(obj: Any, path: str | Path, **kwargs) -> None:
    if not HAS_SAFETENSORS:
        _raise_package_not_installed("safetensors", ".safetensors/.st")
    import safetensors  # pylint: disable=redefined-outer-name

    if HAS_NUMPY:
        import numpy as np

        if _is_dict_of(obj, np.ndarray):
            safetensors.numpy.save_file(obj, str(path), **kwargs)
            return

    if HAS_TENSORFLOW:
        import tensorflow as tf

        if _is_dict_of(obj, tf.Tensor):
            safetensors.tensorflow.save_file(obj, str(path), **kwargs)
            return

    if HAS_PYTORCH:
        import torch

        if _is_dict_of(obj, torch.Tensor):
            safetensors.torch.save_file(obj, str(path), **kwargs)
            return

    raise _raise_wrong_type(
        path,
        "dict of numpy arrays, a dict of tensorflow tensors, or a dict of "
        "pytorch tensors",
    )


def load(path: str | Path, **kwargs) -> Any:
    """
    Loads an object from a file using format-specific (or "native") methods.
    See `turbo_broccoli.native.save` for the list of supported file extensions.

    Warning:
        Safetensors files (`.st` or `.safetensors`) will be loaded as dicts of
        numpy arrays even of the object was originally a dict of e.g. torch
        tensors.
    """
    extension = Path(path).suffix
    methods: dict[str, Callable[[str | Path], Any]] = {
        ".csv": _load_csv,
        ".h5": _load_keras,
        ".keras": _load_keras,
        ".npy": _load_np,
        ".npz": _load_np,
        ".parquet": _load_pq,
        ".pq": _load_pq,
        ".pt": _load_pt,
        ".st": _load_st,
        ".tf": _load_keras,
    }
    method: Callable = methods.get(extension, load_json)
    return method(path, **kwargs)


def save(obj: Any, path: str | Path, **kwargs) -> None:
    """
    Saves an object using the file extension of `path` to determine the
    serialization/dumping method:

    * `.csv`:
      [`pandas.DataFrame.to_csv`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
      or
      [`pandas.Series.to_csv`](https://pandas.pydata.org/docs/reference/api/pandas.Series.to_csv.html)
    * `.h5`:
      [`tf.keras.saving.save_model`](https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model)
      with `save_format="h5"`
    * `.keras`:
      [`tf.keras.saving.save_model`](https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model)
      with `save_format="keras"`
    * `.npy`:
        [`numpy.save`](https://numpy.org/doc/stable/reference/generated/numpy.save.html)
    * `.npz`:
        [`numpy.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
    * `.pq`, `.parquet`:
      [`pandas.DataFrame.to_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)
    * `.pt`:
      [`torch.save`](https://pytorch.org/docs/stable/generated/torch.save.html)
    * `.safetensors`, `.st`: (for numpy arrays, pytorch tensors and tensorflow tensors)
      [safetensors](https://huggingface.co/docs/safetensors/index)
    * `.tf`:
      [`tf.keras.saving.save_model`](https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model)
      with `save_format="tf"`
    * `.json` and anything else: just forwarded to `turbo_broccoli.save_json`

    Args:
        obj (Any):
        path (str | Path):
        kwargs: Passed to the serialization method
    """
    extension = Path(path).suffix
    methods: dict[str, Callable[[Any, str | Path], None]] = {
        ".csv": _save_csv,
        ".h5": partial(_save_keras, save_format="h5"),
        ".keras": partial(_save_keras, save_format="keras"),
        ".npy": _save_npy,
        ".npz": _save_npz,
        ".parquet": _save_pq,
        ".pq": _save_pq,
        ".pt": _save_pt,
        ".st": _save_st,
        ".tf": partial(_save_keras, save_format="tf"),
    }
    method = methods.get(extension, save_json)
    method(obj, path, **kwargs)
