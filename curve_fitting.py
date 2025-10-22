"""
Extrapolation using velocity-based physics with improved cricket ball dynamics
Continues ball path from batsman to stumps using realistic projectile motion
Handles bouncing during extrapolation phase (e.g., yorker deliveries)
Ensures smooth continuation from tracked trajectory with proper cricket physics
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
    
    spline_z = compute_natural_cubic_spline(x_points, z_points)
    spline_y = compute_natural_cubic_spline(x_points, y_points)
    
    return {
        'x_points': x_points,
        'spline_z': spline_z,
        'spline_y': spline_y,
        'n_segments': n - 1
    }

def compute_natural_cubic_spline(x, y):
    """Compute natural cubic spline coefficients"""
    n = len(x) - 1
    h = [x[i+1] - x[i] for i in range(n)]
    
    alpha = [0.0] * (n + 1)
    for i in range(1, n):
        alpha[i] = (3.0/h[i])*(y[i+1]-y[i]) - (3.0/h[i-1])*(y[i]-y[i-1])
    
    l = [1.0] * (n + 1)
    mu = [0.0] * (n + 1)
    z = [0.0] * (n + 1)
    
    for i in range(1, n):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        if abs(l[i]) < 1e-10:
            l[i] = 1e-10
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
    
    c = [0.0] * (n + 1)
    b = [0.0] * n
    d = [0.0] * n
    a = y[:]
    
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    
    spline_coeffs = []
    for i in range(n):
        spline_coeffs.append({
            'x_start': x[i],
            'x_end': x[i+1],
            'a': a[i],
            'b': b[i],
            'c': c[i],
            'd': d[i]
        })
    
    return spline_coeffs

def evaluate_spline(spline_model, x_value, coordinate='z'):
    """Evaluate spline at given x value"""
    if coordinate == 'z':
        spline_coeffs = spline_model['spline_z']
    else:
        spline_coeffs = spline_model['spline_y']
    
    for seg in spline_coeffs:
        if seg['x_start'] <= x_value <= seg['x_end']:
            dx = x_value - seg['x_start']
            value = seg['a'] + seg['b']*dx + seg['c']*(dx**2) + seg['d']*(dx**3)
            return value
    
    if x_value > spline_coeffs[-1]['x_end']:
        seg = spline_coeffs[-1]
        dx = x_value - seg['x_start']
        value = seg['a'] + seg['b']*dx + seg['c']*(dx**2) + seg['d']*(dx**3)
        return value
    
    return None

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