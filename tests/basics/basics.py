import yaml

import pandas as pd
import numpy as np
import pytest

def construct_pandas_index(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.Index:
    return pd.RangeIndex(**loader.construct_mapping(node))

def construct_pandas_series(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.Series:
    return pd.Series(**loader.construct_mapping(node, deep=True))

def construct_pandas_dataframe(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.DataFrame:
    return pd.DataFrame(**loader.construct_mapping(node, deep=True))

def construct_numpy_array(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> np.array:
    return np.array(**loader.construct_mapping(node, deep=True))

def assertNone(a):
    assert a is None

def assertEq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-8)
    else:
        assert a == b