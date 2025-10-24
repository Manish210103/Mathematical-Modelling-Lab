def calculate_lbw_decision(tracked_trajectory, extrapolated_trajectory, stump_x=20.0, 
                          stump_height=0.71, stump_width=0.228):
    """
    Determine LBW decision based on real cricket rules
    
    Cricket LBW Rules (DRS):
    1. PITCHING: Ball must pitch between popping crease (bowling) and middle stump
       (or outside off stump is OK, but typically in-line or off-stump)
    2. IMPACT: Ball impact with pad in-line with stumps (lateral position check)
    3. WICKETS: Ball must be hitting stumps (within stump zone)
    4. CONFIDENCE: Decision requires >75% confidence
    """
    
    if not extrapolated_trajectory or len(extrapolated_trajectory) < 2:
        return {
            'is_out': False,
            'decision': 'NOT OUT',
            'reason': 'Insufficient extrapolated data',
            'confidence': 0.0,
            'rule_checks': {
                'pitching_ok': False,
                'impact_ok': False,
                'hitting_stumps': False,
                'confidence_threshold': False
            },
            'details': {}
        }
    
    # Check pitching
    pitching_result = check_pitching(tracked_trajectory)
    pitching_ok = pitching_result['pitching_ok']
    pitching_line = pitching_result['pitch_line']
    
    # Check impact position
    impact_result = check_impact_position(tracked_trajectory, extrapolated_trajectory)
    impact_ok = impact_result['impact_ok']
    impact_x = impact_result['impact_x']
    impact_y = impact_result['impact_y']
    
    # Check if hitting stumps
    stumps_result = check_hitting_stumps(
        extrapolated_trajectory, stump_x, stump_height, stump_width
    )
    hitting_stumps = stumps_result['hitting_stumps']
    x_at_stump = stumps_result['x_at_stump']
    y_at_stump = stumps_result['y_at_stump']
    z_at_stump = stumps_result['z_at_stump']
    
    # Calculate confidence score
    confidence = calculate_confidence(
        y_at_stump, z_at_stump, stump_height, stump_width,
        hitting_stumps, pitching_ok, impact_ok
    )
    
    # Final decision
    all_rules_met = (pitching_ok and impact_ok and hitting_stumps and confidence > 75.0)
    
    if all_rules_met:
        decision = 'OUT'
        is_out = True
        decision_reason = 'Ball pitching OK, impact in-line, hitting stumps with >75% confidence'
    else:
        decision = 'NOT OUT'
        is_out = False
        
        if not pitching_ok:
            decision_reason = f'Pitching check failed: {pitching_line}'
        elif not impact_ok:
            decision_reason = f'Ball impact position not in-line with stumps'
        elif not hitting_stumps:
            decision_reason = f'Ball not hitting stumps (missing by {stumps_result["miss_distance"]:.3f}m)'
        else:
            decision_reason = f'Confidence only {confidence:.1f}% (need >75% for OUT)'
    
    return {
        'is_out': is_out,
        'decision': decision,
        'reason': decision_reason,
        'confidence': confidence,
        'rule_checks': {
            'pitching_ok': pitching_ok,
            'impact_ok': impact_ok,
            'hitting_stumps': hitting_stumps,
            'confidence_threshold': confidence > 75.0
        },
        'details': {
            'pitching': pitching_result,
            'impact': impact_result,
            'stumps': stumps_result,
            'x_at_stump': x_at_stump,
            'y_at_stump': y_at_stump,
            'z_at_stump': z_at_stump
        }
    }


def check_pitching(tracked_trajectory):
    """Check if ball pitched correctly"""
    if not tracked_trajectory or len(tracked_trajectory) < 2:
        return {
            'pitching_ok': False,
            'pitch_line': 'Cannot determine pitching',
            'pitch_point': None
        }
    
    # Find pitch point (first bounce or very close to ground)
    pitch_point = None
    for i, point in enumerate(tracked_trajectory[1:], 1):
        if point['z'] < 0.15:
            pitch_point = point
            break
    
    if not pitch_point:
        pitch_point = tracked_trajectory[int(len(tracked_trajectory) * 0.1)]
    
    pitch_x = pitch_point['x']
    pitch_y = pitch_point['y']
    
    acceptable_lateral = abs(pitch_y) <= 0.15
    
    if acceptable_lateral:
        pitching_ok = True
        pitch_line = 'In-line with stumps'
    elif pitch_y > 0:
        pitching_ok = False
        pitch_line = f'Off-stump side by {abs(pitch_y):.3f}m (acceptable for OFF stump balls)'
        if pitch_y <= 0.25:
            pitching_ok = True
            pitch_line = 'Off-stump line (acceptable)'
    else:
        pitching_ok = False
        pitch_line = f'Leg-stump side (not acceptable for LBW)'
    
    return {
        'pitching_ok': pitching_ok,
        'pitch_line': pitch_line,
        'pitch_point': pitch_point,
        'pitch_y': pitch_y
    }


def check_impact_position(tracked_trajectory, extrapolated_trajectory):
    """Check if ball impact with pad is in-line with stumps"""
    if not tracked_trajectory:
        return {
            'impact_ok': False,
            'impact_x': 0,
            'impact_y': 0
        }
    
    last_tracked = tracked_trajectory[-1]
    impact_x = last_tracked['x']
    impact_y = last_tracked['y']
    
    stump_half_width = 0.228 / 2
    impact_tolerance = stump_half_width * 1.1
    
    impact_ok = abs(impact_y) <= impact_tolerance
    
    return {
        'impact_ok': impact_ok,
        'impact_x': impact_x,
        'impact_y': impact_y,
        'impact_margin': impact_tolerance - abs(impact_y) if impact_ok else 0
    }


def check_hitting_stumps(extrapolated_trajectory, stump_x=20.0, 
                         stump_height=0.71, stump_width=0.228):
    """Check if ball would hit stumps at stump position"""
    if not extrapolated_trajectory:
        return {
            'hitting_stumps': False,
            'x_at_stump': 0,
            'y_at_stump': 0,
            'z_at_stump': 0,
            'miss_distance': 0
        }
    
    # Find point closest to stump position
    impact_point = None
    min_distance = float('inf')
    
    for point in extrapolated_trajectory:
        dist = abs(point['x'] - stump_x)
        if dist < min_distance:
            min_distance = dist
            impact_point = point
    
    if not impact_point:
        return {
            'hitting_stumps': False,
            'x_at_stump': 0,
            'y_at_stump': 0,
            'z_at_stump': 0,
            'miss_distance': float('inf')
        }
    
    y_at_stump = impact_point['y']
    z_at_stump = impact_point['z']
    x_at_stump = impact_point['x']
    
    stump_half_width = stump_width / 2
    
    lateral_ok = abs(y_at_stump) <= stump_half_width
    vertical_ok = 0 <= z_at_stump <= stump_height
    
    hitting_stumps = lateral_ok and vertical_ok
    
    miss_distance = 0
    if not hitting_stumps:
        if not lateral_ok:
            miss_distance = abs(y_at_stump) - stump_half_width
        elif not vertical_ok:
            miss_distance = abs(z_at_stump) if z_at_stump < 0 else z_at_stump - stump_height
    
    return {
        'hitting_stumps': hitting_stumps,
        'x_at_stump': x_at_stump,
        'y_at_stump': y_at_stump,
        'z_at_stump': z_at_stump,
        'lateral_ok': lateral_ok,
        'vertical_ok': vertical_ok,
        'miss_distance': miss_distance,
        'impact_point': impact_point
    }


def calculate_confidence(y_pos, z_pos, stump_height, stump_width,
                        hitting_stumps, pitching_ok, impact_ok):
    """Calculate confidence score for LBW decision"""
    
    if not hitting_stumps:
        return 0.0
    
    if not pitching_ok or not impact_ok:
        return 0.0
    
    stump_half_width = stump_width / 2
    
    # Lateral confidence (0-100)
    if abs(y_pos) <= stump_half_width * 0.3:
        lateral_conf = 100.0
    elif abs(y_pos) <= stump_half_width * 0.6:
        lateral_conf = 95.0
    elif abs(y_pos) <= stump_half_width * 0.9:
        lateral_conf = 85.0
    else:
        lateral_conf = 70.0
    
    # Vertical confidence (0-100)
    middle_start = stump_height * 0.3
    middle_end = stump_height * 0.7
    
    if middle_start <= z_pos <= middle_end:
        vertical_conf = 100.0
    elif z_pos < middle_start:
        if z_pos >= stump_height * 0.1:
            vertical_conf = 75.0
        else:
            vertical_conf = 75.0
    else:
        if z_pos <= stump_height * 0.9:
            vertical_conf = 75.0
        else:
            vertical_conf = 75.0
    
    # Overall confidence (weighted: 50% lateral, 50% vertical)
    overall_confidence = lateral_conf * 0.5 + vertical_conf * 0.5
    
    return overall_confidence


def analyze_ball_path(tracked_trajectory, extrapolated_trajectory):
    """
    Analyze complete ball path from bowler to stumps
    Provides: Bounce detection, trajectory type, path statistics, distance metrics
    """
    
    if not tracked_trajectory:
        return {}
    
    # Detect bounce in tracked trajectory
    bounce_detected = False
    bounce_point = None
    bounce_x = None
    
    for i in range(1, len(tracked_trajectory) - 1):
        prev_z = tracked_trajectory[i-1]['z']
        curr_z = tracked_trajectory[i]['z']
        next_z = tracked_trajectory[i+1]['z']
        
        if curr_z < 0.1 and prev_z > curr_z and next_z >= curr_z:
            bounce_detected = True
            bounce_point = tracked_trajectory[i]
            bounce_x = bounce_point['x']
            break
    
    # Determine trajectory type
    if not bounce_detected:
        trajectory_type = "Full Toss (no bounce - reaches batsman in air)"
    else:
        if bounce_x < 5:
            trajectory_type = "Short Pitch (early bounce ~3-5m, high rise)"
        elif bounce_x < 10:
            trajectory_type = "Good Length (medium bounce ~6-9m)"
        elif bounce_x < 15:
            trajectory_type = "Half Volley (late bounce ~10-15m)"
        else:
            trajectory_type = "Yorker (very late bounce, near batsman)"
    
    # Calculate statistics
    all_points = tracked_trajectory + (extrapolated_trajectory if extrapolated_trajectory else [])
    
    if all_points:
        max_height = max([p['z'] for p in all_points])
        min_height = min([p['z'] for p in all_points])
        
        all_y = [p['y'] for p in all_points]
        max_lateral = max([abs(y) for y in all_y])
        lateral_variation = max(all_y) - min(all_y)
    else:
        max_height = 0
        min_height = 0
        max_lateral = 0
        lateral_variation = 0
    
    # Distance metrics
    total_tracked = tracked_trajectory[-1]['x'] if tracked_trajectory else 0
    
    if extrapolated_trajectory and tracked_trajectory:
        total_extrapolated = extrapolated_trajectory[-1]['x'] - tracked_trajectory[-1]['x']
    else:
        total_extrapolated = 0
    
    total_distance = all_points[-1]['x'] - all_points[0]['x'] if all_points else 0
    
    return {
        'bounce_detected': bounce_detected,
        'bounce_point': bounce_point,
        'bounce_x': bounce_x,
        'trajectory_type': trajectory_type,
        'max_height': max_height,
        'min_height': min_height,
        'max_lateral_deviation': max_lateral,
        'lateral_variation': lateral_variation,
        'total_tracked_distance': total_tracked,
        'total_extrapolated_distance': total_extrapolated,
        'total_distance': total_distance,
        'height_variation': max_height - min_height,
        'num_tracked_points': len(tracked_trajectory),
        'num_extrapolated_points': len(extrapolated_trajectory) if extrapolated_trajectory else 0
    }