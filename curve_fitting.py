from math_utils import sqrt
import math
from math_utils import mean

# Fits natural cubic splines to z(x) and y(x) from trajectory points
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
        'fpp_z': fpp_z,
        'fpp_y': fpp_y,
        'n_segments': n - 1
    }

# Computes natural cubic spline second derivatives and segment coefficients
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
        term1 = (6.0 / (x[i + 1] - x[i])) * (f[i + 1] - f[i])
        term2 = (6.0 / (x[i] - x[i - 1])) * (f[i] - f[i - 1])
        rhs_i = term1 - term2
        if i > 1:
            lower_diag.append(a_i)
        main_diag.append(b_i)
        if i < n - 2:
            upper_diag.append(c_i)
        rhs.append(rhs_i)
    fpp_inner = solve_tridiagonal_system(lower_diag, main_diag, upper_diag, rhs)
    fpp = [0.0] + fpp_inner + [0.0]
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

# Solves a tridiagonal linear system via Thomas algorithm
def solve_tridiagonal_system(lower, main, upper, rhs):
    n = len(main)
    if n == 0:
        return []
    
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
    
    solution = [0.0] * n
    solution[n - 1] = d_prime[n - 1]
    
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    
    return solution

# Builds linear spline segments when too few points for a cubic spline
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

# Builds human-readable text for spline segments and second derivatives
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
    
    if fpp:
        equations_text += "Second Derivatives (f''):\n"
        for i, fpp_val in enumerate(fpp):
            equations_text += f"  f''_{i} = {fpp_val:.6f}\n"
        equations_text += "\n"
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
        equations_text += f"  S_{i}(x) = \n"
        equations_text += f"    {fpp_i:.6f} * (({x_i1:.3f} - x)^3) / (6 * {h:.3f})\n"
        equations_text += f"  + {fpp_i1:.6f} * ((x - {x_i:.3f})^3) / (6 * {h:.3f})\n"
        
        coeff_3 = f_i / h - fpp_i * h / 6
        coeff_4 = f_i1 / h - fpp_i1 * h / 6
        
        equations_text += f"  + {coeff_3:.6f} * ({x_i1:.3f} - x)\n"
        equations_text += f"  + {coeff_4:.6f} * (x - {x_i:.3f})\n"
        equations_text += "\n"
    
    return equations_text

# Evaluates the spline at a given x for z or y
def evaluate_spline(spline_model, x_value, coordinate='z'):
    if coordinate == 'z':
        spline_coeffs = spline_model['spline_z']
    else:
        spline_coeffs = spline_model['spline_y']
    
    for seg in spline_coeffs:
        if seg['x_start'] <= x_value <= seg['x_end']:
            return evaluate_cubic_spline_segment(x_value, seg)
    if x_value > spline_coeffs[-1]['x_end']:
        return evaluate_cubic_spline_segment(x_value, spline_coeffs[-1])
    if x_value < spline_coeffs[0]['x_start']:
        return evaluate_cubic_spline_segment(x_value, spline_coeffs[0])
    
    return None

# Evaluates a single cubic spline segment at x
def evaluate_cubic_spline_segment(x_val, seg):
    x_i = seg['x_start']
    x_i1 = seg['x_end']
    h = seg['h']
    f_i = seg['f_i']
    f_i1 = seg['f_i1']
    fpp_i = seg['fpp_i']
    fpp_i1 = seg['fpp_i1']
    
    term1 = fpp_i * ((x_i1 - x_val) ** 3) / (6 * h)
    term2 = fpp_i1 * ((x_val - x_i) ** 3) / (6 * h)
    term3 = (f_i / h - fpp_i * h / 6) * (x_i1 - x_val)
    term4 = (f_i1 / h - fpp_i1 * h / 6) * (x_val - x_i)
    
    return term1 + term2 + term3 + term4

# Estimates velocity components from the last tracked points
def calculate_velocity_from_trajectory(trajectory):
    n = len(trajectory)
    if n < 3:
        return None
    
    num_points = min(8, n)
    points = trajectory[-num_points:]
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
            
            weight = (i + 1) ** 1.5
            
            velocities_x.append(vx)
            velocities_y.append(vy)
            velocities_z.append(vz)
            weights.append(weight)
    
    if not velocities_x:
        return None
    
    total_weight = sum(weights)
    avg_vx = sum(vx * w for vx, w in zip(velocities_x, weights)) / total_weight
    avg_vy = sum(vy * w for vy, w in zip(velocities_y, weights)) / total_weight
    avg_vz = sum(vz * w for vz, w in zip(velocities_z, weights)) / total_weight
    
    return {
        'vx': avg_vx,
        'vy': avg_vy,
        'vz': avg_vz
    }

# Detects ground bounce and applies bounce physics to z and vz
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
        z_final = GROUND_THRESHOLD
        
        bounce_state['bounce_count'] += 1
        bounce_state['last_bounce_z'] = z_current
        
        return z_final, vz_final, True, bounce_state
    
    elif z_new <= 0:
        z_final = 0.0
        vz_final = 0.0
        return z_final, vz_final, False, bounce_state
    
    else:
        return z_new, vz, False, bounce_state

# Extrapolates the ball path to stumps using physics and bounce handling
def extrapolate_to_stumps(spline_model, trajectory, stump_x=20.0):
    if not trajectory or len(trajectory) < 3:
        return []
    last_point = trajectory[-1]
    x0 = last_point['x']
    y0 = last_point['y']
    z0 = last_point['z']
    t0 = last_point['t']
    if x0 >= stump_x - 0.05:
        return []
    velocity = calculate_velocity_from_trajectory(trajectory)
    if not velocity:
        return []
    
    vx0 = velocity['vx']
    vy0 = velocity['vy']
    vz0 = velocity['vz']
    
    if vx0 <= 0:
        vx0 = 15.0  
    GRAVITY = 9.81  
    AIR_DRAG_COEFFICIENT = 0.018
    LATERAL_DAMPING = 0.65 
    MAGNUS_EFFECT = 0.008  
    bounce_state = {
        'bounce_count': 0,
        'last_bounce_z': None
    }
    speed_initial = sqrt(vx0**2 + vy0**2 + vz0**2)
    extrapolated = []
    distance_to_stumps = stump_x - x0
    x_step = 0.04
    num_steps = int(distance_to_stumps / x_step) + 15
    vx = vx0
    vy = vy0
    vz = vz0
    x_current = x0
    y_current = y0
    z_current = z0
    t_current = t0
    for i in range(1, num_steps):
        dt = x_step / vx if vx > 1.0 else 0.01
        x_new = x_current + vx * dt
        if x_new > stump_x + 0.05:
            break
        speed = sqrt(vx**2 + vy**2 + vz**2)
        drag_factor = AIR_DRAG_COEFFICIENT * speed
        az = -GRAVITY - drag_factor * abs(vz) * (1 if vz > 0 else -1)
        vz_new = vz + az * dt
        z_new = z_current + (vz + vz_new) / 2 * dt
        z_final, vz_final, bounce_occurred, bounce_state = detect_bounce_and_apply_physics(
            z_current, z_new, vz_new, bounce_state
        )
        if bounce_occurred:
            vy = vy * 0.9 + (0.15 if (bounce_state['bounce_count'] % 2) else -0.15)
        distance_traveled = x_new - x0
        ay = -LATERAL_DAMPING * vy
        magnus_force = MAGNUS_EFFECT * speed * math.sin(distance_traveled * 0.8)
        ay += magnus_force
        vy_new = vy + ay * dt
        y_new = y_current + (vy + vy_new) / 2 * dt
        ax = -drag_factor * 0.1
        vx_new = vx + ax * dt
        if vx_new < 5.0:
            vx_new = 5.0
        t_new = t_current + dt
        extrapolated.append({
            'x': round(x_new, 3),
            'y': round(y_new, 4),
            'z': round(z_final, 3),
            't': round(t_new, 4),
            'bounced': bounce_occurred
        })
        x_current = x_new
        y_current = y_new
        z_current = z_final
        vx = vx_new
        vy = vy_new
        vz = vz_final
        t_current = t_new
    if extrapolated and abs(extrapolated[-1]['x'] - stump_x) > 0.01:
        dt_final = (stump_x - x_current) / vx if vx > 0 else 0
        speed_final = sqrt(vx**2 + vy**2 + vz**2)
        az = -GRAVITY - AIR_DRAG_COEFFICIENT * speed_final * abs(vz)
        vz_final = vz + az * dt_final
        z_final = z_current + (vz + vz_final) / 2 * dt_final
        if z_final < 0:
            z_final = 0.0
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

# Computes RMSE for spline predictions vs actual trajectory in z and y
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

def _design_matrix(x_values, degree):
    return [[(x ** p) for p in range(degree + 1)] for x in x_values]

# Solves linear system with Gaussian elimination
def _gaussian_elimination_solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(M[r][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            return [0.0] * n
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

# Computes X^T X and X^T y normal equations
def _normal_equations(X, y):
    m = len(X[0])
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

# Fits least-squares polynomials for z(x) and y(x) of a given degree
def fit_polynomial(trajectory, degree=3):
    if not trajectory or len(trajectory) < degree + 1:
        return None
    x_vals = [p['x'] for p in trajectory]
    y_vals = [p['y'] for p in trajectory]
    z_vals = [p['z'] for p in trajectory]
    X = _design_matrix(x_vals, degree)
    XTX_z, XTy_z = _normal_equations(X, z_vals)
    coeffs_z = _gaussian_elimination_solve(XTX_z, XTy_z)
    XTX_y, XTy_y = _normal_equations(X, y_vals)
    coeffs_y = _gaussian_elimination_solve(XTX_y, XTy_y)
    return {
        'type': 'polynomial',
        'degree': degree,
        'coeffs_z': coeffs_z,
        'coeffs_y': coeffs_y,
        'x_min': min(x_vals),
        'x_max': max(x_vals)
    }

# Evaluates a polynomial model at x for z or y using Horner’s method
def evaluate_polynomial(poly_model, x_value, coordinate='z'):
    if not poly_model:
        return None
    coeffs = poly_model['coeffs_z'] if coordinate == 'z' else poly_model['coeffs_y']
    val = 0.0
    for c in reversed(coeffs):
        val = val * x_value + c
    return val

# Computes RMSE, MAE, and R² for z and y predictions
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

# Returns error metrics for spline fit
def statistics_for_spline(trajectory, spline_model):
    def predictor(x):
        return evaluate_spline(spline_model, x, 'z'), evaluate_spline(spline_model, x, 'y')
    stats = _calc_errors(trajectory, predictor)
    stats['model_type'] = 'spline'
    return stats

# Returns error metrics for polynomial fit
def statistics_for_polynomial(trajectory, poly_model):
    def predictor(x):
        return evaluate_polynomial(poly_model, x, 'z'), evaluate_polynomial(poly_model, x, 'y')
    stats = _calc_errors(trajectory, predictor)
    stats['model_type'] = 'polynomial'
    return stats

# Compares spline vs polynomial by Z metrics and picks the best
def select_best_model(spline_stats, poly_stats):
    best = 'spline'
    reason = []
    if poly_stats['z']['rmse'] + 1e-9 < spline_stats['z']['rmse']:
        best = 'polynomial'
        reason.append('Lower RMSE (Z)')
    elif spline_stats['z']['rmse'] + 1e-9 < poly_stats['z']['rmse']:
        best = 'spline'
        reason.append('Lower RMSE (Z)')
    else:
        if poly_stats['z']['r2'] > spline_stats['z']['r2'] + 1e-9:
            best = 'polynomial'
            reason.append('Higher R² (Z)')
        elif spline_stats['z']['r2'] > poly_stats['z']['r2'] + 1e-9:
            best = 'spline'
            reason.append('Higher R² (Z)')
        else:
            if poly_stats['z']['mae'] < spline_stats['z']['mae']:
                best = 'polynomial'
                reason.append('Lower MAE (Z)')
            else:
                best = 'spline'
                reason.append('Lower MAE (Z)')
    return {'best': best, 'reason': ', '.join(reason)}

# Builds a readable polynomial equation string for z or y
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

 