import numpy as np


# =============================================================================
# Base function
# =============================================================================
def calc_attr_belief(X, n):
    """
    Parameters
    ----------
    X : array-like
        输入数据

    n : int
        需要切成n个片段

    Returns
    -------
    result : A
       属性参考值.
    """
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    A = []
    for i in range(np.shape(X)[1]):
        A.append(list(np.linspace(col_min[i], col_max[i], num=n)))
    return A


def process_to_pieces(X, y, y_piece, X_piece=5):
    A = calc_attr_belief(X, X_piece)
    D = calc_attr_belief(y.reshape(-1, 1), y_piece)[0]
    return A, D
