"""
Numeric inputs shared by evaluator and metric APIs.

Array shapes and numeric contents are checked by the input validators. Lists
and tuples may be nested to represent survival curves and competing-risk CIFs.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch

Numeric = float | int | bool
NumericArrayLike = (
    list[Any] | tuple[Any, ...] | np.ndarray | pd.Series | pd.DataFrame | torch.Tensor
)
