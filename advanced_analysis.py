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

    noise_y = y_vals - trend_y
    noise_z = z_vals - trend_z

    noise_std_y = np.std(noise_y)
    noise_std_z = np.std(noise_z)

    signal_range_y = np.ptp(trend_y) 
    signal_range_z = np.ptp(trend_z)

    snr_y = (signal_range_y / (noise_std_y + 1e-6))
    snr_z = (signal_range_z / (noise_std_z + 1e-6))

    dy_dx = np.gradient(trend_y, x_vals)
    dz_dx = np.gradient(trend_z, x_vals)

    curvature_y = np.gradient(dy_dx, x_vals)
    curvature_z = np.gradient(dz_dx, x_vals)

    mean_curvature_y = np.mean(np.abs(curvature_y))
    mean_curvature_z = np.mean(np.abs(curvature_z))

    fft_z = np.fft.fft(noise_z)
    fft_y = np.fft.fft(noise_y)
    power_spectrum_z = np.abs(fft_z) ** 2
    power_spectrum_y = np.abs(fft_y) ** 2

    dominant_freq_z = np.argmax(power_spectrum_z[1:len(power_spectrum_z)//2])
    dominant_freq_y = np.argmax(power_spectrum_y[1:len(power_spectrum_y)//2])

    quality = interpret_tracking_quality(snr_z, snr_y, noise_std_z, noise_std_y)
    remarks = generate_diagnostics(snr_z, snr_y, mean_curvature_z, mean_curvature_y)

    return {
        "trend_z": trend_z.tolist(),
        "trend_y": trend_y.tolist(),
        "noise_z": noise_z.tolist(),
        "noise_y": noise_y.tolist(),
        "noise_std_z": noise_std_z,
        "noise_std_y": noise_std_y,
        "snr_z": snr_z,
        "snr_y": snr_y,
        "curvature_z_mean": mean_curvature_z,
        "curvature_y_mean": mean_curvature_y,
        "dominant_noise_freq_z": int(dominant_freq_z),
        "dominant_noise_freq_y": int(dominant_freq_y),
        "quality_assessment": quality,
        "diagnostic_remarks": remarks,
        "n_points": n
    }


# Interpret signal quality summary
def interpret_tracking_quality(snr_z, snr_y, noise_z, noise_y):
    mean_snr = (snr_z + snr_y) / 2
    mean_noise = (noise_z + noise_y) / 2
    if mean_snr > 25 and mean_noise < 0.01:
        return "🌟 Excellent tracking quality – sensor precision very high"
    elif mean_snr > 10:
        return "✅ Good tracking – minor sensor jitter, clean trajectory"
    elif mean_snr > 5:
        return "⚠️ Moderate tracking – noticeable random fluctuations"
    else:
        return "❌ Poor tracking – noisy signal, consider recalibration"


# Generate brief diagnostics text
def generate_diagnostics(snr_z, snr_y, curv_z, curv_y):
    diagnostics = []
    if snr_z > snr_y:
        diagnostics.append("Vertical motion is more stable than lateral.")
    else:
        diagnostics.append("Lateral motion more consistent — possible swing detected.")
    if curv_z > 0.05:
        diagnostics.append("Height profile shows curvature variation — bounce or dip detected.")
    if curv_y > 0.02:
        diagnostics.append("Lateral curvature noticeable — possible swing or deviation mid-flight.")
    if snr_z > 20 and snr_y > 20:
        diagnostics.append("Trajectory is nearly deterministic with negligible random noise.")
    elif snr_z < 5 or snr_y < 5:
        diagnostics.append("High-frequency jitter dominates — potential tracking sensor error.")
    return " ".join(diagnostics)
