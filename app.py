"""
DRS-Lite: Cricket Ball Trajectory Analysis with Cubic Spline & Smooth Curve Extrapolation
Main Streamlit Application - Updated with real cricket LBW rules
"""

import streamlit as st
import pandas as pd

# Import custom modules
from trajectory_data import generate_bouncing_trajectory, get_available_scenarios
from curve_fitting import (fit_cubic_spline, extrapolate_to_stumps, 
                           calculate_rmse, evaluate_spline)
from lbw_analysis import calculate_lbw_decision, analyze_ball_path
from visualization import create_trajectory_plot, create_2d_plots
from math_utils import mean, std_deviation

def main():
    st.set_page_config(page_title="DRS-Lite System", layout="wide", page_icon="🏏")
    
    st.title("🏏 DRS-Lite: Cricket Ball Trajectory Analysis System")
    st.markdown("*Using Cubic Spline Interpolation & Physics-Based Extrapolation with Real Cricket LBW Rules*")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Scenario selection
    scenarios = get_available_scenarios()
    selected_scenario = st.sidebar.selectbox(
        "Select Ball Delivery Type",
        options=list(scenarios.keys()),
        format_func=lambda x: scenarios[x]
    )
    
    # Tracking distance
    tracking_distance = st.sidebar.slider(
        "Ball Tracked Till (Batsman Position)",
        min_value=15.0,
        max_value=18.0,
        value=17.0,
        step=0.5,
        help="Distance till which ball is tracked by cameras (15-18m)"
    )
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Analysis Module",
        ["📊 Trajectory Analysis", 
         "⚖️ LBW Decision", 
         "🎨 3D Visualization"]
    )
    
    st.sidebar.markdown("---")
    
    # Generate trajectory button
    if st.sidebar.button("🎲 Generate Trajectory", type="primary"):
        with st.spinner(f"Generating {scenarios[selected_scenario]}..."):
            trajectory = generate_bouncing_trajectory(selected_scenario, tracking_distance)
            st.session_state.trajectory = trajectory
            st.session_state.tracking_distance = tracking_distance
            st.session_state.scenario_type = selected_scenario
            st.session_state.spline_model = None
            st.session_state.extrapolated = None
            st.session_state.lbw_result = None
            st.success(f"✅ Generated {len(trajectory)} tracked points")
    
    # Initialize session state
    if 'trajectory' not in st.session_state:
        st.session_state.trajectory = generate_bouncing_trajectory("good_length", 17.0)
        st.session_state.tracking_distance = 17.0
        st.session_state.scenario_type = "good_length"
    
    if 'spline_model' not in st.session_state:
        st.session_state.spline_model = None
    if 'extrapolated' not in st.session_state:
        st.session_state.extrapolated = None
    if 'lbw_result' not in st.session_state:
        st.session_state.lbw_result = None
    
    # Render selected page
    if page == "📊 Trajectory Analysis":
        render_trajectory_analysis_page()
    elif page == "⚖️ LBW Decision":
        render_lbw_decision_page()
    elif page == "🎨 3D Visualization":
        render_3d_visualization_page()
    
def render_trajectory_analysis_page():
    """Render trajectory analysis page"""
    st.header("📊 Ball Trajectory Analysis")
    
    trajectory = st.session_state.trajectory
    scenario_info = get_available_scenarios()
    
    # Display scenario info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Delivery Type:**\n{scenario_info[st.session_state.scenario_type]}")
    with col2:
        st.info(f"**Tracked Distance:**\n{st.session_state.tracking_distance}m")
    with col3:
        st.info(f"**Tracked Points:**\n{len(trajectory)} points")
    
    st.markdown("---")
    
    # Trajectory statistics
    st.subheader("📈 Tracked Trajectory Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_height = max([p['z'] for p in trajectory])
        st.metric("Max Height", f"{max_height:.2f} m")
    
    with col2:
        max_lateral = max([abs(p['y']) for p in trajectory])
        st.metric("Max Lateral Dev.", f"{max_lateral:.3f} m")
    
    with col3:
        duration = trajectory[-1]['t'] if trajectory else 0
        st.metric("Duration", f"{duration:.3f} s")
    
    with col4:
        distance = trajectory[-1]['x'] if trajectory else 0
        st.metric("Distance Covered", f"{distance:.2f} m")
    
    # Fit spline and extrapolate
    st.markdown("---")
    st.subheader("📬 Cubic Spline Fitting & Extrapolation to Stumps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Cubic Spline Interpolation:**
        - Fits smooth curve through tracked points
        - Natural cubic spline (zero 2nd derivative at endpoints)
        - Each segment: S(x) = a + b(x-xᵢ) + c(x-xᵢ)² + d(x-xᵢ)³
        """)
    
    with col2:
        st.markdown("""
        **Physics-Based Extrapolation:**
        - Continues from last tracked point to stumps
        - Projectile motion with gravity
        - Air resistance and spin effects
        - Maintains velocity continuity
        """)
    
    if st.button("📬 Fit Spline & Extrapolate to Stumps", type="primary"):
        with st.spinner("Fitting cubic spline and extrapolating with physics..."):
            spline_model = fit_cubic_spline(trajectory)
            st.session_state.spline_model = spline_model
            
            extrapolated = extrapolate_to_stumps(spline_model, trajectory, stump_x=20.0)
            st.session_state.extrapolated = extrapolated
            
            rmse_values = calculate_rmse(trajectory, spline_model)
            
            st.success("✅ Spline fitted successfully!")
            st.info(f"📊 RMSE - Z: {rmse_values['rmse_z']:.5f}m, Y: {rmse_values['rmse_y']:.5f}m")
            st.success(f"✅ Extrapolated {len(extrapolated)} points to stumps")
    
    # Display fitted model info
    if st.session_state.spline_model:
        st.markdown("---")
        st.subheader("📈 Spline Model Details")
        
        spline = st.session_state.spline_model
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Segments", spline['n_segments'])
            st.caption("Cubic polynomial segments")
        
        with col2:
            rmse_values = calculate_rmse(trajectory, spline)
            st.metric("Fit Quality (RMSE)", f"{rmse_values['rmse_z']:.5f} m")
            st.caption("Root Mean Square Error in Z-direction")
    
    # Display extrapolated points
    if st.session_state.extrapolated:
        st.markdown("---")
        st.subheader("🎯 Extrapolated Points (Physics-Based)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            df_extrap = pd.DataFrame(st.session_state.extrapolated[:])
            st.dataframe(df_extrap, use_container_width=True)
            st.caption("Extrapolated points - smooth parabolic curve from batsman to stumps")
        
        with col2:
            extrap_distance = st.session_state.extrapolated[-1]['x'] - trajectory[-1]['x']
            st.metric("Extrapolated Distance", f"{extrap_distance:.2f} m")
            st.metric("Total Points", len(st.session_state.extrapolated))
            st.metric("Start X", f"{trajectory[-1]['x']:.2f} m")
            st.metric("End X", f"{st.session_state.extrapolated[-1]['x']:.2f} m")
    
    # 2D Plots
    if st.session_state.extrapolated:
        st.markdown("---")
        st.subheader("📉 2D Trajectory Views")
        
        fig_2d = create_2d_plots(trajectory, st.session_state.extrapolated)
        st.plotly_chart(fig_2d, use_container_width=True)

def render_lbw_decision_page():
    """Render LBW decision page with real cricket rules"""
    st.header("⚖️ LBW (Leg Before Wicket) Decision System")
    
    if not st.session_state.extrapolated:
        st.warning("⚠️ Please fit spline and extrapolate trajectory first")
        return
    
    trajectory = st.session_state.trajectory
    extrapolated = st.session_state.extrapolated
    
    st.markdown("### Real Cricket LBW Rules (DRS Implementation)")
    st.markdown("""
    For a batsman to be given **OUT LBW**, ALL of the following must be true:
    
    1. **PITCHING**: Ball must pitch in-line with or outside off stump (acceptable line)
    2. **IMPACT**: Ball impact with pad must be in-line with stumps (lateral position)
    3. **WICKETS**: Ball would go on to hit the stumps (height and position)
    4. **CONFIDENCE**: Decision requires >90% confidence (tracking accuracy)
    
    **Analysis Method:**
    - Grey trajectory: Ball tracked from bowler (0m) to batsman
    - Red trajectory: Extrapolated path from batsman to stumps (20m)
    - All checks must pass AND confidence > 90% for OUT decision
    """)
    
    st.markdown("---")
    
    # Stump specifications
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stump Position", "20.0 m")
    with col2:
        st.metric("Stump Height", "0.71 m")
    with col3:
        st.metric("Stump Width", "0.228 m")
    
    st.markdown("---")
    
    # Calculate LBW
    if st.button("⚖️ Calculate LBW Decision (All Rules)", type="primary"):
        with st.spinner("Analyzing ball trajectory with real cricket rules..."):
            lbw_result = calculate_lbw_decision(trajectory, extrapolated)
            ball_path_analysis = analyze_ball_path(trajectory, extrapolated)
            
            st.session_state.lbw_result = lbw_result
            st.session_state.ball_path = ball_path_analysis
            
            st.success("✅ LBW analysis complete")
    
    # Display LBW result
    if st.session_state.lbw_result:
        result = st.session_state.lbw_result
        
        st.markdown("---")
        st.subheader("⚖️ FINAL DECISION")
        
        # Big decision display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if result['is_out']:
                st.error("# 🚫 OUT LBW")
                st.markdown(f"### Confidence: {result['confidence']:.1f}%")
            else:
                st.success("# ✅ NOT OUT")
                st.markdown(f"### Confidence: {result['confidence']:.1f}%")
            
            st.info(f"**Decision Reason:** {result['reason']}")
        
        st.markdown("---")
        
        # Rule checks
        st.subheader("🔍 Rule Verification")
        
        checks = result['rule_checks']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ PASS" if checks['pitching_ok'] else "❌ FAIL"
            st.metric("Pitching Check", status)
            st.caption("Ball in acceptable line")
        
        with col2:
            status = "✅ PASS" if checks['impact_ok'] else "❌ FAIL"
            st.metric("Impact Position", status)
            st.caption("In-line with stumps")
        
        with col3:
            status = "✅ PASS" if checks['hitting_stumps'] else "❌ FAIL"
            st.metric("Hitting Stumps", status)
            st.caption("Within stump zone")
        
        with col4:
            status = "✅ PASS" if checks['confidence_threshold'] else "❌ FAIL"
            st.metric("Confidence >90%", status)
            st.caption("Required for OUT")
        
        st.markdown("---")
        
        # Detailed position analysis
        st.subheader("📍 Impact Point Analysis at Stumps (20m)")
        
        details = result['details']
        stumps = details['stumps']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("X Position", f"{stumps['x_at_stump']:.3f} m")
            st.caption("At stump line")
        
        with col2:
            y_pos = stumps['y_at_stump']
            st.metric("Y Position (Lateral)", f"{y_pos:.3f} m")
            if stumps['lateral_ok']:
                st.success("✅ Within stumps")
            else:
                st.error(f"❌ Outside by {stumps['miss_distance']:.3f}m")
        
        with col3:
            z_pos = stumps['z_at_stump']
            st.metric("Z Position (Height)", f"{z_pos:.3f} m")
            if stumps['vertical_ok']:
                st.success("✅ Within stump height")
            else:
                st.error(f"❌ Miss by {stumps['miss_distance']:.3f}m")
        
        # Ball path characteristics
        if 'ball_path' in st.session_state:
            st.markdown("---")
            st.subheader("🎯 Ball Path Characteristics")
            
            path = st.session_state.ball_path
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bounce_status = "Yes ✅" if path['bounce_detected'] else "No ❌"
                st.metric("Bounce Detected", bounce_status)
            
            with col2:
                if path['bounce_detected'] and path['bounce_x']:
                    st.metric("Bounce Position", f"{path['bounce_x']:.2f} m")
                else:
                    st.metric("Bounce Position", "Full Toss")
            
            with col3:
                st.metric("Max Height", f"{path['max_height']:.2f} m")
            
            with col4:
                st.metric("Lateral Variation", f"{path['lateral_variation']:.3f} m")
            
            st.info(f"**Trajectory Type:** {path['trajectory_type']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tracked Distance", f"{path['total_tracked_distance']:.2f} m")
            with col2:
                st.metric("Extrapolated Distance", f"{path['total_extrapolated_distance']:.2f} m")

def render_3d_visualization_page():
    """Render 3D visualization page"""
    st.header("🎨 Interactive 3D Ball Trajectory Visualization")
    
    trajectory = st.session_state.trajectory
    
    st.markdown("""
    **Visualization Legend:**
    - 🔴 **Red Solid Line**: Tracked trajectory (camera data from bowler to batsman)
    - 🔵 **Blue Dashed Line**: Extrapolated smooth curve path (batsman to stumps)
    - 🟡 **Yellow Lines**: Three stumps at 20m
    - ⚪ **White Marker**: Batsman position
    - 🔵 **Cyan Marker**: Bowler release point
    - 🔴 **Red Diamond**: Impact point at stumps
    """)
    
    st.markdown("---")
    
    # Create 3D plot
    fig = create_trajectory_plot(
        trajectory,
        st.session_state.extrapolated,
        st.session_state.spline_model
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    st.markdown("---")
    st.subheader("📊 Trajectory Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tracked Points", len(trajectory))
    
    with col2:
        if st.session_state.extrapolated:
            st.metric("Extrapolated Points", len(st.session_state.extrapolated))
        else:
            st.metric("Extrapolated Points", "0")
    
    with col3:
        max_height = max([p['z'] for p in trajectory])
        st.metric("Peak Height", f"{max_height:.2f} m")
    
    with col4:
        distance = trajectory[-1]['x']
        st.metric("Tracking Distance", f"{distance:.2f} m")

if __name__ == "__main__":
    main()