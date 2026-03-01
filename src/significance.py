# src/significance.py
import numpy as np
from scipy import stats

def diebold_mariano(y_true, y_pred1, y_pred2=None, h=1, alternative="two-sided"):
    """
    Diebold-Mariano test for forecast accuracy comparison.

    Parameters:
        y_true : array-like
            True values
        y_pred1 : array-like
            Forecasts from model 1
        y_pred2 : array-like or None
            Forecasts from model 2 (if None, compare against y_true as baseline)
        h : int
            Forecast horizon (steps ahead)
        alternative : str
            "two-sided", "less", "greater"

    Returns:
        dm_stat : float
            Diebold-Mariano test statistic
        p_value : float
            Two-sided p-value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred1).flatten()

    if y_pred2 is None:
        y_pred2 = np.zeros_like(y_pred1)
    else:
        y_pred2 = np.asarray(y_pred2).flatten()

    # Truncate to the smallest length to avoid shape issues
    min_len = min(len(y_true), len(y_pred1), len(y_pred2))
    y_true = y_true[-min_len:]
    y_pred1 = y_pred1[-min_len:]
    y_pred2 = y_pred2[-min_len:]

    # Compute forecast errors
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Loss differential (squared errors)
    d = e1**2 - e2**2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    # Adjust for autocorrelation for multi-step horizon
    gamma = 0
    for lag in range(1, h):
        gamma += 2 * (1 - lag/h) * np.cov(d[lag:], d[:-lag])[0,1]

    dm_stat = d_mean / np.sqrt((d_var + gamma)/len(d))

    # Two-sided p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return dm_stat, p_value