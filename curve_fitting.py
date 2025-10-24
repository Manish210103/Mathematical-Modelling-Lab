from math_utils import sqrt
import math
from math_utils import mean

 # Fit natural cubic spline to trajectory data
def fit_cubic_spline(trajectory):
    n = len(trajectory)
    if n < 4:
        return None
    
    x_points = [p['x'] for p in trajectory]
    y_points = [p['y'] for p in trajectory]
    z_points = [p['z'] for p in trajectory]
    
    spline_z, fpp_z = compute_natural_cubic_spline(x_points, z_points)
    spline_y, fpp_y = compute_natural_cubic_spline(x_points, y_points)
    
    return {
        'x_points': x_points,
        'spline_z': spline_z,
        'spline_y': spline_y,
        'fpp_z': fpp_z,  # Store second derivatives
        'fpp_y': fpp_y,
        'n_segments': n - 1
    }

 # Compute natural cubic spline coefficients
def compute_natural_cubic_spline(x, f):

    n = len(x)
    num_unknowns = n - 2
    
    if num_unknowns <= 0:
        return compute_simple_spline(x, f), [0.0] * n
    
    lower_diag = []  
    main_diag = []   
    upper_diag = []  
    rhs = []         
    
    for i in range(1, n - 1):
        a_i = x[i] - x[i - 1]
        b_i = 2 * (x[i + 1] - x[i - 1])
        c_i = x[i + 1] - x[i]
        
        # Calculate right hand side
        term1 = (6.0 / (x[i + 1] - x[i])) * (f[i + 1] - f[i])
        term2 = (6.0 / (x[i] - x[i - 1])) * (f[i] - f[i - 1])
        rhs_i = term1 - term2
        
        if i > 1:
            lower_diag.append(a_i)
        main_diag.append(b_i)
        if i < n - 2:
            upper_diag.append(c_i)
        rhs.append(rhs_i)
    
    # Solve tridiagonal system
    fpp_inner = solve_tridiagonal_system(lower_diag, main_diag, upper_diag, rhs)
    
    # Construct full f'' array with boundary conditions
    fpp = [0.0] + fpp_inner + [0.0]
    
    # Build spline segments using the formula:
    # S_i(x) = f''_i * (x_{i+1} - x)^3 / (6*h_i) + f''_{i+1} * (x - x_i)^3 / (6*h_i)
    #        + (f_i/h_i - f''_i*h_i/6) * (x_{i+1} - x)
    #        + (f_{i+1}/h_i - f''_{i+1}*h_i/6) * (x - x_i)
    
    spline_coeffs = []
    for i in range(n - 1):
        h_i = x[i + 1] - x[i]
        
        spline_coeffs.append({
            'x_start': x[i],
            'x_end': x[i + 1],
            'h': h_i,
            'f_i': f[i],
            'f_i1': f[i + 1],
            'fpp_i': fpp[i],
            'fpp_i1': fpp[i + 1]
        })
    
    return spline_coeffs, fpp

 # Solve tridiagonal system using Thomas algorithm
def solve_tridiagonal_system(lower, main, upper, rhs):
    n = len(main)
    if n == 0:
        return []
    
    # Forward elimination
    c_prime = [0.0] * n
    d_prime = [0.0] * n
    
    c_prime[0] = upper[0] / main[0] if n > 1 else 0
    d_prime[0] = rhs[0] / main[0]
    
    for i in range(1, n):
        denom = main[i] - (lower[i - 1] if i > 0 else 0) * c_prime[i - 1]
        if abs(denom) < 1e-10:
            denom = 1e-10
        
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - (lower[i - 1] if i > 0 else 0) * d_prime[i - 1]) / denom
    
    # Back substitution
    solution = [0.0] * n
    solution[n - 1] = d_prime[n - 1]
    
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    
    return solution

 # Fallback simple linear spline when not enough points
def compute_simple_spline(x, f):
    spline_coeffs = []
    for i in range(len(x) - 1):
        spline_coeffs.append({
            'x_start': x[i],
            'x_end': x[i + 1],
            'h': x[i + 1] - x[i],
            'f_i': f[i],
            'f_i1': f[i + 1],
            'fpp_i': 0.0,
            'fpp_i1': 0.0
        })
    return spline_coeffs

 # Build human-readable text for spline equations
def get_spline_equations_text(spline_model, coordinate='z'):
    if not spline_model:
        return "No spline model available"
    
    if coordinate == 'z':
        spline_coeffs = spline_model['spline_z']
        fpp = spline_model.get('fpp_z', [])
        coord_name = "Z (Height)"
    else:
        spline_coeffs = spline_model['spline_y']
        fpp = spline_model.get('fpp_y', [])
        coord_name = "Y (Lateral)"
    
    equations_text = f"Cubic Spline Equations for {coord_name}:\n"
    equations_text += "=" * 80 + "\n\n"
    
    # Print second derivatives if available
    if fpp:
        equations_text += "Second Derivatives (f''):\n"
        for i, fpp_val in enumerate(fpp):
            equations_text += f"  f''_{i} = {fpp_val:.6f}\n"
        equations_text += "\n"
    
    # Print each segment equation
    for i, seg in enumerate(spline_coeffs):
        x_i = seg['x_start']
        x_i1 = seg['x_end']
        h = seg['h']
        f_i = seg['f_i']
        f_i1 = seg['f_i1']
        fpp_i = seg['fpp_i']
        fpp_i1 = seg['fpp_i1']
        
        equations_text += f"Segment S_{i}(x) for interval [{x_i:.3f}, {x_i1:.3f}]:\n"
        equations_text += f"  h_{i} = {h:.3f}\n"
        equations_text += f"  f_{i} = {f_i:.6f}, f_{i+1} = {f_i1:.6f}\n"
        equations_text += f"  f''_{i} = {fpp_i:.6f}, f''_{i+1} = {fpp_i1:.6f}\n\n"
        
        # Build the equation symbolically
        equations_text += f"  S_{i}(x) = \n"
        equations_text += f"    {fpp_i:.6f} * (({x_i1:.3f} - x)^3) / (6 * {h:.3f})\n"
        equations_text += f"  + {fpp_i1:.6f} * ((x - {x_i:.3f})^3) / (6 * {h:.3f})\n"
        
        coeff_3 = f_i / h - fpp_i * h / 6
        coeff_4 = f_i1 / h - fpp_i1 * h / 6
        
        equations_text += f"  + {coeff_3:.6f} * ({x_i1:.3f} - x)\n"
        equations_text += f"  + {coeff_4:.6f} * (x - {x_i:.3f})\n"
        equations_text += "\n"
    
    return equations_text

 # Evaluate spline at x
def evaluate_spline(spline_model, x_value, coordinate='z'):
    if coordinate == 'z':
        spline_coeffs = spline_model['spline_z']
    else:
        spline_coeffs = spline_model['spline_y']
    
    # Find the appropriate segment
    for seg in spline_coeffs:
        if seg['x_start'] <= x_value <= seg['x_end']:
            return evaluate_cubic_spline_segment(x_value, seg)
    
    # If beyond last segment, extrapolate using last segment
    if x_value > spline_coeffs[-1]['x_end']:
        return evaluate_cubic_spline_segment(x_value, spline_coeffs[-1])
    
    # If before first segment, use first segment
    if x_value < spline_coeffs[0]['x_start']:
        return evaluate_cubic_spline_segment(x_value, spline_coeffs[0])
    
    return None

 # Evaluate a cubic spline segment at a point
def evaluate_cubic_spline_segment(x_val, seg):
    x_i = seg['x_start']
    x_i1 = seg['x_end']
    h = seg['h']
    f_i = seg['f_i']
    f_i1 = seg['f_i1']
    fpp_i = seg['fpp_i']
    fpp_i1 = seg['fpp_i1']
    
    # Apply the formula
    term1 = fpp_i * ((x_i1 - x_val) ** 3) / (6 * h)
    term2 = fpp_i1 * ((x_val - x_i) ** 3) / (6 * h)
    term3 = (f_i / h - fpp_i * h / 6) * (x_i1 - x_val)
    term4 = (f_i1 / h - fpp_i1 * h / 6) * (x_val - x_i)
    
    return term1 + term2 + term3 + term4

 # Estimate velocity at last tracked point (weighted recent points)
def calculate_velocity_from_trajectory(trajectory):
    n = len(trajectory)
    if n < 3:
        return None
    
    # Use last 5-8 points for better accuracy
    num_points = min(8, n)
    points = trajectory[-num_points:]
    
    # Calculate velocities with weights
    velocities_x = []
    velocities_y = []
    velocities_z = []
    weights = []
    
    for i in range(len(points) - 1):
        dt = points[i+1]['t'] - points[i]['t']
        if dt > 0.0001:
            vx = (points[i+1]['x'] - points[i]['x']) / dt
            vy = (points[i+1]['y'] - points[i]['y']) / dt
            vz = (points[i+1]['z'] - points[i]['z']) / dt
            
            # Weight: more recent points get higher weight
            weight = (i + 1) ** 1.5
            
            velocities_x.append(vx)
            velocities_y.append(vy)
            velocities_z.append(vz)
            weights.append(weight)
    
    if not velocities_x:
        return None
    
    # Weighted average velocity
    total_weight = sum(weights)
    avg_vx = sum(vx * w for vx, w in zip(velocities_x, weights)) / total_weight
    avg_vy = sum(vy * w for vy, w in zip(velocities_y, weights)) / total_weight
    avg_vz = sum(vz * w for vz, w in zip(velocities_z, weights)) / total_weight
    
    return {
        'vx': avg_vx,
        'vy': avg_vy,
        'vz': avg_vz
    }

 # Detect ground hit and apply bounce physics
def detect_bounce_and_apply_physics(z_current, z_new, vz, bounce_state):
    GROUND_THRESHOLD = 0.02  
        
    if z_new <= GROUND_THRESHOLD and z_current > GROUND_THRESHOLD:
        bounce_occurred = True
        
        if bounce_state['bounce_count'] == 0:
            if abs(vz) < 3.0:
                BOUNCE_COEFF = 0.45
            elif abs(vz) < 6.0:
                BOUNCE_COEFF = 0.55
            else:
                BOUNCE_COEFF = 0.65
        else:
            BOUNCE_COEFF = 0.40
        
        vz_final = -vz * BOUNCE_COEFF
        z_final = GROUND_THRESHOLD  # Place ball just on ground
        
        # Update bounce state
        bounce_state['bounce_count'] += 1
        bounce_state['last_bounce_z'] = z_current
        
        return z_final, vz_final, True, bounce_state
    
    elif z_new <= 0:
        # Ball is below ground (shouldn't happen, but safety check)
        z_final = 0.0
        vz_final = 0.0
        return z_final, vz_final, False, bounce_state
    
    else:
        # No bounce, normal trajectory
        return z_new, vz, False, bounce_state

 # Physics-based extrapolation from batsman to stumps with bounce handling
def extrapolate_to_stumps(spline_model, trajectory, stump_x=20.0):
    if not trajectory or len(trajectory) < 3:
        return []
    
    # Get last tracked point (batsman position)
    last_point = trajectory[-1]
    x0 = last_point['x']
    y0 = last_point['y']
    z0 = last_point['z']
    t0 = last_point['t']
    
    # Check if already past stumps
    if x0 >= stump_x - 0.05:
        return []
    
    # Calculate current velocity with improved method
    velocity = calculate_velocity_from_trajectory(trajectory)
    if not velocity:
        return []
    
    vx0 = velocity['vx']
    vy0 = velocity['vy']
    vz0 = velocity['vz']
    
    # Ensure reasonable velocities
    if vx0 <= 0:
        vx0 = 15.0  # Default forward velocity
    
    # Physics constants (tuned for cricket ball behavior)
    GRAVITY = 9.81  # m/s² (standard gravity)
    AIR_DRAG_COEFFICIENT = 0.018  # Reduced drag for smoother curve
    LATERAL_DAMPING = 0.65  # Heavily damped lateral movement
    MAGNUS_EFFECT = 0.008  # Very small spin effect
    
    # Bounce tracking state
    bounce_state = {
        'bounce_count': 0,
        'last_bounce_z': None
    }
    
    # Calculate initial speed for air resistance
    speed_initial = sqrt(vx0**2 + vy0**2 + vz0**2)
    
    extrapolated = []
    
    # Distance to travel
    distance_to_stumps = stump_x - x0
    
    # Time step based on horizontal velocity (0.04m steps for better bounce detection)
    x_step = 0.04
    num_steps = int(distance_to_stumps / x_step) + 15
    
    # Current velocities (will change due to forces)
    vx = vx0
    vy = vy0
    vz = vz0
    
    # Current position
    x_current = x0
    y_current = y0
    z_current = z0
    t_current = t0
    
    for i in range(1, num_steps):
        # Calculate time step
        dt = x_step / vx if vx > 1.0 else 0.01
        
        # Update X position (nearly constant velocity)
        x_new = x_current + vx * dt
        
        if x_new > stump_x + 0.05:
            break
        
        # Calculate current speed for air resistance
        speed = sqrt(vx**2 + vy**2 + vz**2)
        drag_factor = AIR_DRAG_COEFFICIENT * speed
        
        # ==================================================
        # Z-AXIS: DRAMATIC DOWNWARD MOTION (Strong gravity)
        # WITH BOUNCE DETECTION
        # ==================================================
        # Acceleration: gravity + air resistance
        az = -GRAVITY - drag_factor * abs(vz) * (1 if vz > 0 else -1)
        
        # Update vertical velocity
        vz_new = vz + az * dt
        
        # Update vertical position using average velocity
        z_new = z_current + (vz + vz_new) / 2 * dt
        
        # *** BOUNCE DETECTION AND HANDLING ***
        z_final, vz_final, bounce_occurred, bounce_state = detect_bounce_and_apply_physics(
            z_current, z_new, vz_new, bounce_state
        )
        
        # If bounce occurred, add slight lateral variation due to pitch irregularities
        if bounce_occurred:
            # Slight random variation in lateral direction on bounce
            vy = vy * 0.9 + (0.15 if (bounce_state['bounce_count'] % 2) else -0.15)
        
        # ==================================================
        # Y-AXIS: MINIMAL LATERAL DEVIATION (Small change)
        # ==================================================
        # Very small spin effect, heavily damped
        distance_traveled = x_new - x0
        
        # Lateral acceleration (minimal)
        # 1. Damped initial lateral velocity
        ay = -LATERAL_DAMPING * vy
        
        # 2. Very small Magnus effect (spin-induced drift)
        magnus_force = MAGNUS_EFFECT * speed * math.sin(distance_traveled * 0.8)
        ay += magnus_force
        
        # Update lateral velocity (minimal change)
        vy_new = vy + ay * dt
        
        # Update lateral position (small deviation)
        y_new = y_current + (vy + vy_new) / 2 * dt
        
        # ==================================================
        # X-AXIS: Nearly constant (slight air resistance)
        # ==================================================
        ax = -drag_factor * 0.1  # Minimal horizontal deceleration
        vx_new = vx + ax * dt
        
        # Ensure minimum forward velocity
        if vx_new < 5.0:
            vx_new = 5.0
        
        # Update time
        t_new = t_current + dt
        
        # Store extrapolated point
        extrapolated.append({
            'x': round(x_new, 3),
            'y': round(y_new, 4),  # Higher precision for small changes
            'z': round(z_final, 3),
            't': round(t_new, 4),
            'bounced': bounce_occurred  # Track if bounce occurred at this point
        })
        
        # Update current state
        x_current = x_new
        y_current = y_new
        z_current = z_final
        vx = vx_new
        vy = vy_new
        vz = vz_final  # Use the velocity after bounce (if any)
        t_current = t_new
    
    # Ensure final point is exactly at stumps position
    if extrapolated and abs(extrapolated[-1]['x'] - stump_x) > 0.01:
        # Calculate precise position at stumps
        dt_final = (stump_x - x_current) / vx if vx > 0 else 0
        
        # Final Z position (with gravity)
        speed_final = sqrt(vx**2 + vy**2 + vz**2)
        az = -GRAVITY - AIR_DRAG_COEFFICIENT * speed_final * abs(vz)
        vz_final = vz + az * dt_final
        z_final = z_current + (vz + vz_final) / 2 * dt_final
        
        # Check for bounce at final point
        if z_final < 0:
            z_final = 0.0
        
        # Final Y position (minimal change)
        distance_to_stump = stump_x - x0
        magnus_final = MAGNUS_EFFECT * speed_final * math.sin(distance_to_stump * 0.8)
        ay_final = -LATERAL_DAMPING * vy + magnus_final
        vy_final = vy + ay_final * dt_final
        y_final = y_current + (vy + vy_final) / 2 * dt_final
        
        extrapolated.append({
            'x': round(stump_x, 3),
            'y': round(y_final, 4),
            'z': round(max(0.0, z_final), 3),
            't': round(t_current + dt_final, 4),
            'bounced': False
        })
    
    return extrapolated

 # Calculate RMSE of spline fit
def calculate_rmse(trajectory, spline_model):
    errors_z = []
    errors_y = []
    
    for point in trajectory:
        x = point['x']
        z_actual = point['z']
        y_actual = point['y']
        
        z_pred = evaluate_spline(spline_model, x, 'z')
        y_pred = evaluate_spline(spline_model, x, 'y')
        
        if z_pred is not None:
            errors_z.append((z_actual - z_pred) ** 2)
        if y_pred is not None:
            errors_y.append((y_actual - y_pred) ** 2)
    
    rmse_z = sqrt(sum(errors_z) / len(errors_z)) if errors_z else 0
    rmse_y = sqrt(sum(errors_y) / len(errors_y)) if errors_y else 0
    
    return {'rmse_z': rmse_z, 'rmse_y': rmse_y}

# ================================
# Polynomial Interpolation/Regression
# ================================

 # Build Vandermonde matrix
def _design_matrix(x_values, degree):
    return [[(x ** p) for p in range(degree + 1)] for x in x_values]

 # Solve linear system by Gaussian elimination with partial pivoting
def _gaussian_elimination_solve(A, b):
    n = len(A)
    # Build augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    # Forward elimination with partial pivoting
    for k in range(n):
        # Pivot
        pivot_row = max(range(k, n), key=lambda r: abs(M[r][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            # Singular; fall back to zeros
            return [0.0] * n
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        # Eliminate
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

 # Compute normal equations (X^T X, X^T y)
def _normal_equations(X, y):
    m = len(X[0])  # number of features
    # XTX
    XTX = [[0.0 for _ in range(m)] for _ in range(m)]
    XTy = [0.0 for _ in range(m)]
    for i in range(len(X)):
        row = X[i]
        yi = y[i]
        for a in range(m):
            XTy[a] += row[a] * yi
            ra = row[a]
            for b in range(m):
                XTX[a][b] += ra * row[b]
    return XTX, XTy

 # Fit polynomial (least squares) to y(x) and z(x)
def fit_polynomial(trajectory, degree=3):
    if not trajectory or len(trajectory) < degree + 1:
        return None
    x_vals = [p['x'] for p in trajectory]
    y_vals = [p['y'] for p in trajectory]
    z_vals = [p['z'] for p in trajectory]
    X = _design_matrix(x_vals, degree)
    # Solve for z(x)
    XTX_z, XTy_z = _normal_equations(X, z_vals)
    coeffs_z = _gaussian_elimination_solve(XTX_z, XTy_z)
    # Solve for y(x)
    XTX_y, XTy_y = _normal_equations(X, y_vals)
    coeffs_y = _gaussian_elimination_solve(XTX_y, XTy_y)
    return {
        'type': 'polynomial',
        'degree': degree,
        'coeffs_z': coeffs_z,  # c0 + c1 x + ...
        'coeffs_y': coeffs_y,
        'x_min': min(x_vals),
        'x_max': max(x_vals)
    }

 # Evaluate polynomial model at x for z or y
def evaluate_polynomial(poly_model, x_value, coordinate='z'):
    if not poly_model:
        return None
    coeffs = poly_model['coeffs_z'] if coordinate == 'z' else poly_model['coeffs_y']
    # Horner's method
    val = 0.0
    for c in reversed(coeffs):
        val = val * x_value + c
    return val

 # Compute error metrics for z and y
def _calc_errors(trajectory, predictor_fn):
    z_actuals, z_preds = [], []
    y_actuals, y_preds = [], []
    for p in trajectory:
        x = p['x']
        z_a = p['z']
        y_a = p['y']
        z_p, y_p = predictor_fn(x)
        if z_p is not None:
            z_actuals.append(z_a)
            z_preds.append(z_p)
        if y_p is not None:
            y_actuals.append(y_a)
            y_preds.append(y_p)
    def _metrics(actuals, preds):
        if not actuals:
            return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
        n = len(actuals)
        sq = [(a - p) ** 2 for a, p in zip(actuals, preds)]
        ab = [abs(a - p) for a, p in zip(actuals, preds)]
        rmse = sqrt(sum(sq) / n)
        mae = sum(ab) / n
        mu = mean(actuals)
        ss_res = sum(sq)
        ss_tot = sum((a - mu) ** 2 for a in actuals) or 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    return {
        'z': _metrics(z_actuals, z_preds),
        'y': _metrics(y_actuals, y_preds),
        'n': len(z_actuals)
    }

 # Compute RMSE/MAE/R2 for spline fit
def statistics_for_spline(trajectory, spline_model):
    def predictor(x):
        return evaluate_spline(spline_model, x, 'z'), evaluate_spline(spline_model, x, 'y')
    stats = _calc_errors(trajectory, predictor)
    stats['aic'] = None
    stats['bic'] = None
    stats['model_type'] = 'spline'
    return stats

 # Compute RMSE/MAE/R2 and AIC/BIC for polynomial fit
def statistics_for_polynomial(trajectory, poly_model):
    def predictor(x):
        return evaluate_polynomial(poly_model, x, 'z'), evaluate_polynomial(poly_model, x, 'y')
    stats = _calc_errors(trajectory, predictor)
    # Use z-dimension errors for information criteria (primary vertical fit)
    n = stats['n'] if stats['n'] else len(trajectory)
    k = poly_model['degree'] + 1
    # Residual sum of squares (use z)
    rss_z = (stats['z']['rmse'] ** 2) * n
    if rss_z <= 0:
        rss_z = 1e-12
    # AIC/BIC for Gaussian errors: AIC = n*ln(RSS/n) + 2k ; BIC = n*ln(RSS/n) + k*ln(n)
    aic = n * math.log(rss_z / n) + 2 * k
    bic = n * math.log(rss_z / n) + k * math.log(max(n, 1))
    stats['aic'] = aic
    stats['bic'] = bic
    stats['model_type'] = 'polynomial'
    return stats

 # Select best model comparing spline vs polynomial
def select_best_model(spline_stats, poly_stats):
    best = 'spline'
    reason = []
    # Compare RMSE in Z (primary physics dimension)
    if poly_stats['z']['rmse'] + 1e-9 < spline_stats['z']['rmse']:
        best = 'polynomial'
        reason.append('Lower RMSE (Z)')
    elif spline_stats['z']['rmse'] + 1e-9 < poly_stats['z']['rmse']:
        best = 'spline'
        reason.append('Lower RMSE (Z)')
    else:
        # tie-breaker with R2
        if poly_stats['z']['r2'] > spline_stats['z']['r2'] + 1e-9:
            best = 'polynomial'
            reason.append('Higher R² (Z)')
        elif spline_stats['z']['r2'] > poly_stats['z']['r2'] + 1e-9:
            best = 'spline'
            reason.append('Higher R² (Z)')
        else:
            # tie-breaker with MAE
            if poly_stats['z']['mae'] < spline_stats['z']['mae']:
                best = 'polynomial'
                reason.append('Lower MAE (Z)')
            else:
                best = 'spline'
                reason.append('Lower MAE (Z)')
    # If polynomial provides AIC/BIC improvements, mention
    if best == 'polynomial' and poly_stats.get('aic') is not None:
        reason.append('Better information criteria (AIC/BIC)')
    return {'best': best, 'reason': ', '.join(reason)}

 # Build human-readable polynomial equation string
def get_polynomial_equation_text(poly_model, coordinate='z'):
    if not poly_model:
        return 'No polynomial model available'
    deg = poly_model['degree']
    coeffs = poly_model['coeffs_z'] if coordinate == 'z' else poly_model['coeffs_y']
    terms = []
    for p, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        if p == 0:
            terms.append(f"{c:.6f}")
        elif p == 1:
            terms.append(f"{c:.6f}·x")
        else:
            terms.append(f"{c:.6f}·x^{p}")
    eq = " + ".join(terms) if terms else "0"
    coord_name = 'Z (Height)' if coordinate == 'z' else 'Y (Lateral)'
    return f"Polynomial Degree {deg} for {coord_name}:\n  f(x) = {eq}"

def build_lagrange_model(x, f, degree):
    m = len(x)
    if m == 0:
        return None
    deg = max(0, min(degree, m-1))
    pts = deg + 1
    xs = x[-pts:]
    fs = f[-pts:]
    return {'type': 'lagrange', 'x': xs, 'f': fs, 'degree': deg}

def evaluate_lagrange(model, x_val):
    if not model:
        return None
    xs = model['x']
    fs = model['f']
    n = len(xs)
    s = 0.0
    for i in range(n):
        li = 1.0
        xi = xs[i]
        for j in range(n):
            if j == i:
                continue
            xj = xs[j]
            denom = (xi - xj) if (xi - xj) != 0 else 1e-12
            li *= (x_val - xj) / denom
        s += fs[i] * li
    return s

def get_lagrange_polynomial_text(model):
    if not model:
        return 'No Lagrange model available'
    xs = model['x']
    fs = model['f']
    lines = [f"Lagrange polynomial (degree {model['degree']}):"]
    parts = []
    for i in range(len(xs)):
        numer = []
        denom = 1.0
        for j in range(len(xs)):
            if j == i:
                continue
            numer.append(f"(x - {xs[j]:.6f})")
            denom *= (xs[i] - xs[j]) if (xs[i] - xs[j]) != 0 else 1e-12
        parts.append(f"{fs[i]:.6f} · [ {' · '.join(numer)} ] / {denom:.6f}")
    lines.append(" + \n".join(parts))
    return "\n".join(lines)

def _trajectory_xy(trajectory, coord):
    x_vals = [p['x'] for p in trajectory]
    y_vals = [p[coord] for p in trajectory]
    return x_vals, y_vals

def build_lagrange_models_for_trajectory(trajectory, degree):
    x, z = _trajectory_xy(trajectory, 'z')
    _, y = _trajectory_xy(trajectory, 'y')
    z_model = build_lagrange_model(x, z, degree)
    y_model = build_lagrange_model(x, y, degree)
    return {'z': z_model, 'y': y_model}