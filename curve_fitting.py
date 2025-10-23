"""
Extrapolation using velocity-based physics with improved cricket ball dynamics
Continues ball path from batsman to stumps using realistic projectile motion
Handles bouncing during extrapolation phase (e.g., yorker deliveries)
Ensures smooth continuation from tracked trajectory with proper cricket physics
NOW WITH CUBIC SPLINE EQUATION PRINTING
"""
from math_utils import sqrt
import math

def fit_cubic_spline(trajectory):
    """Fit natural cubic spline to trajectory data"""
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

def compute_natural_cubic_spline(x, f):
    """
    Compute natural cubic spline coefficients using the mathematical formula:
    
    For i = 1, 2, ..., n-2:
    a_i * f''_{i-1} + b_i * f''_i + c_i * f''_{i+1} = rhs_i
    
    Where:
    - a_i = x_i - x_{i-1}
    - b_i = 2(x_{i+1} - x_{i-1})
    - c_i = x_{i+1} - x_i
    - rhs_i = (6/(x_{i+1} - x_i))(f_{i+1} - f_i) - (6/(x_i - x_{i-1}))(f_i - f_{i-1})
    
    Boundary conditions: f''_0 = 0, f''_{n-1} = 0 (natural spline)
    
    Returns: (spline_coeffs, fpp) where fpp is list of second derivatives
    """
    n = len(x)
    
    # Build tridiagonal system for f'' values
    # For natural spline: f''_0 = 0, f''_{n-1} = 0
    
    # Number of unknowns: f''_1, f''_2, ..., f''_{n-2}
    num_unknowns = n - 2
    
    if num_unknowns <= 0:
        # Not enough points, return simple linear interpolation
        return compute_simple_spline(x, f), [0.0] * n
    
    # Initialize tridiagonal matrix components
    lower_diag = []  # a_i values
    main_diag = []   # b_i values
    upper_diag = []  # c_i values
    rhs = []         # right hand side values
    
    # Build system of equations for i = 1, 2, ..., n-2
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
        
        # Calculate coefficients for cubic polynomial
        # We'll store them in a way that allows easy evaluation
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

def solve_tridiagonal_system(lower, main, upper, rhs):
    """
    Solve tridiagonal system using Thomas algorithm
    lower: lower diagonal (a_i)
    main: main diagonal (b_i)
    upper: upper diagonal (c_i)
    rhs: right hand side
    """
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

def compute_simple_spline(x, f):
    """Fallback for when we don't have enough points"""
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

def get_spline_equations_text(spline_model, coordinate='z'):
    """
    Generate human-readable equations for the cubic spline segments
    
    Returns a formatted string with all segment equations
    """
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

def evaluate_spline(spline_model, x_value, coordinate='z'):
    """
    Evaluate spline at given x value using the cubic spline formula:
    
    S_i(x) = f''_i * (x_{i+1} - x)^3 / (6*h_i) 
           + f''_{i+1} * (x - x_i)^3 / (6*h_i)
           + (f_i/h_i - f''_i*h_i/6) * (x_{i+1} - x)
           + (f_{i+1}/h_i - f''_{i+1}*h_i/6) * (x - x_i)
    """
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

def evaluate_cubic_spline_segment(x_val, seg):
    """
    Evaluate cubic spline at a point using the mathematical formula
    """
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

def calculate_velocity_from_trajectory(trajectory):
    """
    Calculate ball velocity at last tracked point
    Uses weighted average of last 5-8 points for accurate velocity estimation
    Gives more weight to recent points
    """
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

def detect_bounce_and_apply_physics(z_current, z_new, vz, bounce_state):
    """
    Detect if ball hits ground and apply bounce physics
    
    Args:
        z_current: Current Z position
        z_new: New Z position (may be below ground)
        vz: Current vertical velocity
        bounce_state: Dict tracking bounce information
    
    Returns:
        tuple: (z_final, vz_final, bounce_occurred, bounce_state)
    """
    GROUND_THRESHOLD = 0.02  # Consider ground contact if z < 2cm
    
    # Check if ball crosses ground level
    if z_new <= GROUND_THRESHOLD and z_current > GROUND_THRESHOLD:
        # Ball just hit the ground - BOUNCE!
        bounce_occurred = True
        
        # Determine bounce coefficient based on delivery characteristics
        # Yorkers and full tosses have less energy, bounce less
        if bounce_state['bounce_count'] == 0:
            # First bounce in extrapolation
            if abs(vz) < 3.0:
                # Slow/yorker delivery
                BOUNCE_COEFF = 0.45
            elif abs(vz) < 6.0:
                # Normal delivery
                BOUNCE_COEFF = 0.55
            else:
                # High-speed delivery
                BOUNCE_COEFF = 0.65
        else:
            # Subsequent bounces have less energy
            BOUNCE_COEFF = 0.40
        
        # Apply bounce: reverse vertical velocity and reduce magnitude
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

def extrapolate_to_stumps(spline_model, trajectory, stump_x=20.0):
    """
    Extrapolate ball path from batsman to stumps using realistic cricket physics
    NOW WITH BOUNCE DETECTION AND HANDLING!
    
    Key improvements:
    1. Detects ground impact during extrapolation (e.g., yorker deliveries)
    2. Applies bounce physics with realistic coefficient of restitution
    3. Proper velocity calculation using multiple points
    4. Realistic gravity effect (dramatic Z change)
    5. Minimal lateral deviation (small Y change)
    6. Air resistance proportional to velocity squared
    7. Smooth parabolic descent with bounce handling
    
    Physics Model:
    - X-axis: Constant horizontal velocity (minimal air resistance)
    - Z-axis: Strong gravity effect + air resistance + BOUNCE HANDLING
    - Y-axis: Minimal spin/drift effect (SMALL change)
    
    Args:
        spline_model: Fitted spline (unused for extrapolation, kept for interface)
        trajectory: Tracked trajectory data
        stump_x: Position of stumps (20.0m)
    
    Returns:
        List of extrapolated points with realistic cricket ball physics including bounces
    """
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

def calculate_rmse(trajectory, spline_model):
    """Calculate RMSE of spline fit"""
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