import math
import numpy as np
from statistics import mean, stdev

# Single exponential smoothing
def _ses_custom(series, alpha, y0=None):
    n = len(series)
    if n == 0:
        return np.array([])
    s = np.zeros(n, dtype=float)
    s[0] = series[0] if y0 is None else float(y0)
    for t in range(1, n):
        s[t] = alpha * series[t] + (1 - alpha) * s[t-1]
    return s

# Second-order exponential smoothing (Brown)
def _brown_double(series, alpha, y0=None):
    n = len(series)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    s1 = _ses_custom(series, alpha, y0)
    s2 = _ses_custom(s1, alpha, s1[0])
    res = 2 * s1 - s2
    return s1, s2, res


# Decompose trajectory into trend and noise
def decompose_trajectory_advanced(trajectory, method='SES', alpha=0.3, beta=0.1):
    if len(trajectory) < 5:
        return {"error": "Insufficient trajectory points"}

    x_vals = np.array([p['x'] for p in trajectory])
    y_vals = np.array([p['y'] for p in trajectory])
    z_vals = np.array([p['z'] for p in trajectory])

    n = len(x_vals)
    method_u = method.upper()
    if method_u == 'SES':
        trend_y = _ses_custom(y_vals, alpha, y0=y_vals[0])
        trend_z = _ses_custom(z_vals, alpha, y0=z_vals[0])
    elif method_u in ('HOLT', 'BROWN', 'DOUBLE'):
        _, _, trend_y = _brown_double(y_vals, alpha, y0=y_vals[0])
        _, _, trend_z = _brown_double(z_vals, alpha, y0=z_vals[0])
    else:
        trend_y = _ses_custom(y_vals, alpha, y0=y_vals[0])
        trend_z = _ses_custom(z_vals, alpha, y0=z_vals[0])

    return {
        "trend_z": trend_z.tolist(),
        "trend_y": trend_y.tolist(),
    }
