from typing import Union

import numpy as np
import pandas as pd
import torch

Numeric = Union[float, int, bool]
NumericArrayLike = Union[
    list[Numeric], tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor
]
