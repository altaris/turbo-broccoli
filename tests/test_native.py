# pylint: disable=import-outside-toplevel
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
"""turbo_broccoli.native test suite"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from turbo_broccoli.native import save as native_save

TEST_PATH = Path("out") / "test"


def test_save_native_csv():
    import pandas as pd

    ser1 = pd.Series([1, 2, 3], name="asdf")
    native_save(ser1, TEST_PATH / "save_native_csv_1.csv")
    ser2 = pd.read_csv(TEST_PATH / "save_native_csv_1.csv")["asdf"]
    assert (ser1 == ser2).all()

    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]})
    native_save(df1, TEST_PATH / "save_native_csv_2.csv")
    df2 = pd.read_csv(TEST_PATH / "save_native_csv_2.csv")
    if "Unnamed: 0" in df2.columns:
        df2.drop(["Unnamed: 0"], axis=1, inplace=True)
    assert (df1 == df2).all().all()


def test_save_native_npy():
    import numpy as np

    state = np.random.RandomState(seed=0)
    x = state.rand(5, 5)
    native_save(x, TEST_PATH / "test_save_native_npy_1.npy")
    y = np.load(TEST_PATH / "test_save_native_npy_1.npy")
    np.testing.assert_array_equal(x, y)


def test_save_native_npz():
    import numpy as np

    state = np.random.RandomState(seed=0)
    x = {
        "a": state.rand(5, 5),
        "b": state.rand(10, 10),
        "c": state.rand(15, 15),
    }
    native_save(x, TEST_PATH / "test_save_native_npz_1.npz")
    y = np.load(TEST_PATH / "test_save_native_npz_1.npz")
    # assert isinstance(y, dict)
    assert sorted(list(x.keys())) == sorted(list(y.keys()))
    for k, v in x.items():
        np.testing.assert_array_equal(v, y[k])


def test_save_native_pq():
    import pandas as pd

    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]})
    native_save(df1, TEST_PATH / "save_native_pq_1.pq")
    df2 = pd.read_parquet(TEST_PATH / "save_native_pq_1.pq")
    if "Unnamed: 0" in df2.columns:
        df2.drop(["Unnamed: 0"], axis=1, inplace=True)
    assert (df1 == df2).all().all()


def test_save_native_pt():
    import torch

    gen = torch.random.manual_seed(0)

    a = torch.rand((5, 5), generator=gen)
    native_save(a, TEST_PATH / "test_save_native_pt_1.pt")
    b = torch.load(TEST_PATH / "test_save_native_pt_1.pt")
    torch.testing.assert_close(a, b)

    x = {
        "a": torch.rand((5, 5), generator=gen),
        "b": torch.rand((10, 10), generator=gen),
        "c": torch.rand((15, 15), generator=gen),
    }
    native_save(x, TEST_PATH / "test_save_native_pt_2.pt")
    y = torch.load(TEST_PATH / "test_save_native_pt_2.pt")
    assert isinstance(y, dict)
    assert sorted(list(x.keys())) == sorted(list(y.keys()))
    for k, v in x.items():
        torch.testing.assert_close(v, y[k])


def test_save_native_st_np():
    import numpy as np
    from safetensors import numpy as st

    state = np.random.RandomState(seed=0)
    x = {
        "a": state.rand(5, 5),
        "b": state.rand(10, 10),
        "c": state.rand(15, 15),
    }
    native_save(x, TEST_PATH / "test_save_native_st_np_1.st")
    y = st.load_file(TEST_PATH / "test_save_native_st_np_1.st")
    assert isinstance(y, dict)
    assert sorted(list(x.keys())) == sorted(list(y.keys()))
    for k, v in x.items():
        np.testing.assert_array_equal(v, y[k])


def test_save_native_st_tf():
    import tensorflow as tf
    from safetensors import tensorflow as st

    gen = tf.random.Generator.from_seed(0)
    x = {
        "a": gen.uniform((5, 5)),
        "b": gen.uniform((10, 10)),
        "c": gen.uniform((15, 15)),
    }
    native_save(x, TEST_PATH / "test_save_native_st_tf.st")
    y = st.load_file(TEST_PATH / "test_save_native_st_tf.st")
    assert isinstance(y, dict)
    assert sorted(list(x.keys())) == sorted(list(y.keys()))
    for k, v in x.items():
        tf.debugging.assert_equal(v, y[k])


def test_save_native_st_torch():
    import torch
    from safetensors import torch as st

    gen = torch.random.manual_seed(0)
    x = {
        "a": torch.rand((5, 5), generator=gen),
        "b": torch.rand((10, 10), generator=gen),
        "c": torch.rand((15, 15), generator=gen),
    }
    native_save(x, TEST_PATH / "test_save_native_st_torch_1.st")
    y = st.load_file(TEST_PATH / "test_save_native_st_torch_1.st")
    assert isinstance(y, dict)
    assert sorted(list(x.keys())) == sorted(list(y.keys()))
    for k, v in x.items():
        torch.testing.assert_close(v, y[k])
