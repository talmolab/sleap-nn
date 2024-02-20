"""Hungarian matching function(s)."""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Wrapper for Hungarian matching algorithm in scipy."""

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))