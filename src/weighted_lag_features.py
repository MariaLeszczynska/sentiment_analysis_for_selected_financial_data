import numpy as np


def add_weighted_lag_feature(df, col, weights, new_col_name):
    w = np.array(weights, dtype=float)
    w = w / w.sum() # normalise to sum==1
    values = df[col].values
    n = len(values)
    k = len(w)

    result = [None] * n

    for i in range(k-1, n):
        window = values[i-k+1: i+1]
        result[i] = np.dot(window, w)

    
    df[new_col_name] = result
    return df