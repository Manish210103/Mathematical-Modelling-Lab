"""
Realistic cricket ball trajectory using proper physics
Generates smooth parabolic paths with proper projectile motion
"""
import math

def generate_bouncing_trajectory(scenario_type, tracking_distance):
    """
    Generate realistic cricket ball trajectory using projectile motion physics
    
    Physics formula:
    x(t) = v_x * t
    y(t) = y_0 + v_y * t
    z(t) = z_0 + v_z * t - 0.5 * g * t^2 ( Vertical displacement )
    """
    
    RELEASE_HEIGHT = 2.0
    GRAVITY = 9.81
    
    trajectory = []
    
    # Define initial conditions for each delivery type
    if scenario_type == "good_length":
        # Ball that bounces around 6m
        v_x = 20.0  
        v_z_initial = -2.0  
        v_y_initial = 0.02  
        bounce_coeff = 0.6 
        
    elif scenario_type == "yorker":
        # Bounces very close to batsman
        v_x = 25.0
        v_z_initial = 0.75
        v_y_initial = 0.001
        bounce_coeff = 0.1
        
    elif scenario_type == "short_pitch":
        # Early bounce, rises high
        v_x = 18.0
        v_z_initial = -1.5
        v_y_initial = 0.04
        bounce_coeff = 0.75
        
    elif scenario_type == "half_volley":
        # Medium bounce
        v_x = 19.0
        v_z_initial = -2.5
        v_y_initial = 0
        bounce_coeff = 0.65
    
    t = 0.0
    dt = 0.01
    
    x, y, z = 0.0, 0.0, RELEASE_HEIGHT
    v_z = v_z_initial
    v_y = v_y_initial
    
    has_bounced = False
    
    while x < tracking_distance + 0.1:
        # Calculate position using physics
        x = v_x * t
        
        if x > tracking_distance:
            break
        
        # Vertical motion (with gravity)
        z = RELEASE_HEIGHT + v_z_initial * t - 0.5 * GRAVITY * (t ** 2)
        
        # After bounce, recalculate from bounce point
        if has_bounced:
            time_since_bounce = t - bounce_time
            z = bounce_z + v_z_after_bounce * time_since_bounce - 0.5 * GRAVITY * (time_since_bounce ** 2)
        
        # Lateral motion (with spin effect after bounce)
        if has_bounced:
            time_since_bounce = t - bounce_time
            y = bounce_y + v_y * time_since_bounce + 0.05 * (time_since_bounce ** 1.5)
        else:
            y = v_y_initial * x
        
        # Check for bounce
        if z <= 0 and not has_bounced and x > 1.0:
            has_bounced = True
            bounce_time = t
            bounce_y = y
            bounce_z = 0.0
            
            # Calculate velocity at bounce
            v_z_at_bounce = v_z_initial - GRAVITY * t
            
            # Reverse and reduce vertical velocity (bounce)
            v_z_after_bounce = -v_z_at_bounce * bounce_coeff
            
            z = 0.0
        
        # Ensure ball doesn't go below ground
        if z < 0:
            z = 0.0
        
        # Add point to trajectory
        trajectory.append({
            'x': round(x, 3),
            'y': round(y, 3),
            'z': round(z, 3),
            't': round(t, 4)
        })
        
        t += dt
    
    # Ensure last point is at tracking_distance
    if trajectory and abs(trajectory[-1]['x'] - tracking_distance) > 0.05:
        # Interpolate to exact tracking distance
        last = trajectory[-1]
        second_last = trajectory[-2] if len(trajectory) > 1 else last
        
        ratio = (tracking_distance - second_last['x']) / (last['x'] - second_last['x'])
        
        final_y = second_last['y'] + ratio * (last['y'] - second_last['y'])
        final_z = second_last['z'] + ratio * (last['z'] - second_last['z'])
        
        trajectory.append({
            'x': round(tracking_distance, 3),
            'y': round(final_y, 3),
            'z': round(max(0.0, final_z), 3),
            't': round(last['t'] + 0.01, 4)
        })
    
    # Clean up - remove duplicates and ensure increasing x
    clean_trajectory = []
    prev_x = -1
    for point in trajectory:
        if point['x'] > prev_x:
            clean_trajectory.append(point)
            prev_x = point['x']
    
    # Add frame numbers
    for i, point in enumerate(clean_trajectory):
        point['frame'] = i
    
    return clean_trajectory

def get_available_scenarios():
    """Return list of available trajectory scenarios"""
    return {
        "good_length": "Good Length (bounces ~6m)",
        "yorker": "Yorker (late bounce)",
        "short_pitch": "Short Pitch (early bounce, high)",
        "half_volley": "Half Volley (medium bounce)"
    }