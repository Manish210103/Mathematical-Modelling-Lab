import math
import random

# Generate trajectory with bounce physics
def generate_bouncing_trajectory(scenario_type, tracking_distance):
    RELEASE_HEIGHT = 2.0
    GRAVITY = 9.81
    
    trajectory = []
    
    def sample_narrow(mean, zero_span=0.01, frac=0.10):
        if abs(mean) < 1e-12:
            return random.uniform(-zero_span, zero_span)
        delta = frac * abs(mean)
        return random.uniform(mean - delta, mean + delta)

    def sample_y(mean, zero_span=0.02):
        if abs(mean) < 1e-12:
            return random.uniform(-zero_span, zero_span)
        m = abs(mean)
        return random.uniform(-m, m)

    if scenario_type == "good_length":
        v_x = sample_narrow(20.0)
        v_z_initial = sample_narrow(-2.0)
        v_y_initial = sample_y(0.02)
        bounce_coeff = 0.6 
        
    elif scenario_type == "yorker":
        v_x = sample_narrow(25.0)
        v_z_initial = sample_narrow(0.75)
        v_y_initial = sample_y(0.001)
        bounce_coeff = 0.1
        
    elif scenario_type == "short_pitch":
        v_x = sample_narrow(18.0)
        v_z_initial = sample_narrow(-1.5)
        v_y_initial = sample_y(0.04)
        bounce_coeff = 0.75
        
    elif scenario_type == "half_volley":
        v_x = sample_narrow(19.0)
        v_z_initial = sample_narrow(-2.5)
        v_y_initial = sample_y(0.0, zero_span=0.02)
        bounce_coeff = 0.65
    
    t = 0.0
    dt = 0.01
    
    x, y, z = 0.0, 0.0, RELEASE_HEIGHT
    v_z = v_z_initial
    v_y = v_y_initial
    
    has_bounced = False
    
    while x < tracking_distance + 0.1:
        x = v_x * t
        
        if x > tracking_distance:
            break
        
        z = RELEASE_HEIGHT + v_z_initial * t - 0.5 * GRAVITY * (t ** 2)
        
        if has_bounced:
            time_since_bounce = t - bounce_time
            z = bounce_z + v_z_after_bounce * time_since_bounce - 0.5 * GRAVITY * (time_since_bounce ** 2)
        
        if has_bounced:
            time_since_bounce = t - bounce_time
            y = bounce_y + v_y * time_since_bounce + 0.05 * (time_since_bounce ** 1.5)
        else:
            y = v_y_initial * x
        
        if z <= 0 and not has_bounced and x > 1.0:
            has_bounced = True
            bounce_time = t
            bounce_y = y
            bounce_z = 0.0
            
            v_z_at_bounce = v_z_initial - GRAVITY * t
            
            v_z_after_bounce = -v_z_at_bounce * bounce_coeff
            
            z = 0.0
        
        if z < 0:
            z = 0.0
        
        trajectory.append({
            'x': round(x, 3),
            'y': round(y, 3),
            'z': round(z, 3),
            't': round(t, 4)
        })
        
        t += dt
    
    if trajectory and abs(trajectory[-1]['x'] - tracking_distance) > 0.05:
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
    
    clean_trajectory = []
    prev_x = -1
    for point in trajectory:
        if point['x'] > prev_x:
            clean_trajectory.append(point)
            prev_x = point['x']
    
    for i, point in enumerate(clean_trajectory):
        point['frame'] = i
    
    return clean_trajectory

# Available trajectory scenarios
def get_available_scenarios():
    return {
        "good_length": "Good Length (bounces ~6m)",
        "yorker": "Yorker (late bounce)",
        "short_pitch": "Short Pitch (early bounce, high)",
        "half_volley": "Half Volley (medium bounce)"
    }