import streamlit as st
import math
import pandas as pd
import plotly.graph_objects as go

try:
    from scipy.stats import f as scipy_f_dist
except Exception:
    scipy_f_dist = None

from trajectory_data import generate_bouncing_trajectory, get_available_scenarios
from curve_fitting import (fit_cubic_spline, extrapolate_to_stumps, 
                           calculate_rmse, evaluate_spline,
                           fit_polynomial, statistics_for_spline,
                           statistics_for_polynomial, select_best_model,
                           get_polynomial_equation_text,
                           build_lagrange_models_for_trajectory,
                           get_lagrange_polynomial_text,
                           evaluate_lagrange)
from lbw_analysis import calculate_lbw_decision, analyze_ball_path
from visualization import create_trajectory_plot, create_2d_plots
from math_utils import mean, std_deviation
from advanced_analysis import decompose_trajectory_advanced

 # App entrypoint
def main():
    st.set_page_config(page_title="DRS System", layout="wide", page_icon="🏏")
    
    st.title("🏏 DRS: Cricket Ball Trajectory Analysis System")
    
    st.sidebar.title("Configuration")
    
    scenarios = get_available_scenarios()
    selected_scenario = st.sidebar.selectbox(
        "Select Ball Delivery Type",
        options=list(scenarios.keys()),
        format_func=lambda x: scenarios[x]
    )
    
    tracking_distance = st.sidebar.slider(
        "Ball Tracked Till (Batsman Position)",
        min_value=15.0,
        max_value=18.0,
        value=17.0,
        step=0.5,
        help="Distance till which ball is tracked by cameras (15-18m)"
    )
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Analysis Module",
        ["Trajectory Analysis",
         "LBW Decision",
         "3D Visualization",
         "Advanced Analysis"]
    )
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Generate Trajectory", type="primary"):
        with st.spinner(f"Generating {scenarios[selected_scenario]}..."):
            trajectory = generate_bouncing_trajectory(selected_scenario, tracking_distance)
            st.session_state.trajectory = trajectory
            st.session_state.tracking_distance = tracking_distance
            st.session_state.scenario_type = selected_scenario
            st.session_state.spline_model = None
            st.session_state.extrapolated = None
            st.session_state.lbw_result = None
            st.success(f"✅ Generated {len(trajectory)} tracked points")
    
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
    if 'deliveries_history' not in st.session_state:
        st.session_state.deliveries_history = []
    if 'poly_model' not in st.session_state:
        st.session_state.poly_model = None
    if 'model_stats' not in st.session_state:
        st.session_state.model_stats = None
    
    if page == "Trajectory Analysis":
        render_trajectory_analysis_page()
    elif page == "LBW Decision":
        render_lbw_decision_page()
    elif page == "3D Visualization":
        render_3d_visualization_page()
    elif page == "Advanced Analysis":
        render_advanced_analysis_page()

def render_trajectory_analysis_page():
    st.header("Ball Trajectory Analysis")

    trajectory = st.session_state.trajectory
    scenario_info = get_available_scenarios()
    # Initialize placeholders; will refresh after fitting block
    poly = st.session_state.get('poly_model')
    stats = st.session_state.get('model_stats')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Delivery Type:**\n{scenario_info[st.session_state.scenario_type]}")
    with col2:
        st.info(f"**Tracked Distance:**\n{st.session_state.tracking_distance}m")
    with col3:
        st.info(f"**Tracked Points:**\n{len(trajectory)} points")

    st.markdown("---")

    st.subheader("Tracked Trajectory Statistics")

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

    degree = st.slider("Polynomial Degree", min_value=1, max_value=6, value=3, step=1,
                        help="Degree for polynomial least-squares fit of y(x) and z(x)")
    if st.button("Fit Spline + Polynomial & Extrapolate", type="primary"):
        with st.spinner("Fitting cubic spline and extrapolating with physics..."):
            spline_model = fit_cubic_spline(trajectory)
            st.session_state.spline_model = spline_model
            
            extrapolated = extrapolate_to_stumps(spline_model, trajectory, stump_x=20.0)
            st.session_state.extrapolated = extrapolated
            
            poly_model = fit_polynomial(trajectory, degree=degree)
            st.session_state.poly_model = poly_model

            spline_stats = statistics_for_spline(trajectory, spline_model) if spline_model else None
            poly_stats = statistics_for_polynomial(trajectory, poly_model) if poly_model else None
            comparison = select_best_model(spline_stats, poly_stats) if (spline_stats and poly_stats) else None
            st.session_state.model_stats = {
                'spline': spline_stats,
                'polynomial': poly_stats,
                'comparison': comparison
            }

            st.success("✅ Models fitted successfully!")
            st.success(f"✅ Extrapolated {len(extrapolated)} points to stumps")

    if st.session_state.spline_model:
        st.markdown("---")
        st.subheader("Model Details and Comparison")
        
        # Refresh latest models to reflect the just-fitted state in this same run
        spline = st.session_state.spline_model
        poly = st.session_state.get('poly_model')
        stats = st.session_state.get('model_stats')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Segments", spline['n_segments'])
            st.caption("Cubic polynomial segments")
        
        with col2:
            if stats and stats['spline']:
                st.metric("Spline RMSE (Z)", f"{stats['spline']['z']['rmse']:.5f} m")
                st.caption("Root Mean Square Error (Z)")
        with col3:
            if stats and stats['polynomial']:
                st.metric("Poly RMSE (Z)", f"{stats['polynomial']['z']['rmse']:.5f} m")
                st.caption("Root Mean Square Error (Z)")

        st.markdown("---")

        with st.expander("📐 View Model Equations (Click to Expand)", expanded=False):
            from curve_fitting import get_spline_equations_text
            
            tabs = st.tabs(["Spline Z", "Spline Y", "Poly Z", "Poly Y"])
            with tabs[0]:
                equations_z = get_spline_equations_text(spline, 'z')
                st.code(equations_z, language='text')
            with tabs[1]:
                equations_y = get_spline_equations_text(spline, 'y')
                st.code(equations_y, language='text')
            with tabs[2]:
                if poly:
                    st.code(get_polynomial_equation_text(poly, 'z'), language='text')
                else:
                    st.info("Polynomial not fitted yet.")
            with tabs[3]:
                if poly:
                    st.code(get_polynomial_equation_text(poly, 'y'), language='text')
                else:
                    st.info("Polynomial not fitted yet.")

        with st.expander("Interpolation (Lagrange)", expanded=False):
            max_deg = max(1, min(6, len(trajectory) - 1))
            fixed_deg = 2 if max_deg >= 2 else 1
            models_lagr = build_lagrange_models_for_trajectory(trajectory, degree=fixed_deg)
            tabs_interp = st.tabs(["Lagrange Z Poly", "Lagrange Y Poly"])
            with tabs_interp[0]:
                st.code(get_lagrange_polynomial_text(models_lagr['z']), language='text')
            with tabs_interp[1]:
                st.code(get_lagrange_polynomial_text(models_lagr['y']), language='text')

        if stats and stats['comparison']:
            st.markdown("---")
            st.subheader("Best Fit Selection")
            best = stats['comparison']
            st.success(f"Best Model: {best['best'].capitalize()} — {best['reason']}")
            colA, colB, colC, colD, colE = st.columns(5)
            if stats['spline']:
                with colA:
                    st.metric("Spline R² (Z)", f"{stats['spline']['z']['r2']:.4f}")
                with colB:
                    st.metric("Spline MAE (Z)", f"{stats['spline']['z']['mae']:.5f} m")
            if stats['polynomial']:
                with colC:
                    st.metric("Poly R² (Z)", f"{stats['polynomial']['z']['r2']:.4f}")
                with colD:
                    st.metric("Poly MAE (Z)", f"{stats['polynomial']['z']['mae']:.5f} m")

        if poly:
            st.markdown("---")
            st.subheader("Hypothesis Testing (Model Significance)")
            # Significance level selector
            alpha_sig = st.selectbox("Significance level (α)", options=[0.01, 0.05, 0.1], index=1, key="alpha_sig_ht")
            stats_poly = stats['polynomial'] if stats else None
            if not stats_poly:
                st.info("Polynomial statistics not available yet. Click 'Fit Spline + Polynomial & Extrapolate'.")
            elif 'z' not in stats_poly or 'r2' not in stats_poly['z']:
                st.info("R² for Z not computed yet. Try refitting the polynomial.")
            elif stats_poly and stats_poly['n'] <= (poly['degree'] + 1 + 1):
                st.info(f"Insufficient samples for F-test. Need n > k + 1 where k = degree+1. Current n={stats_poly['n']}, k={poly['degree'] + 1}.")
            else:
                n = stats_poly['n']
                k = poly['degree'] + 1
                r2 = stats_poly['z']['r2']
                # Overall F-test for model significance
                if isinstance(r2, (int, float)) and 0 < r2 < 1:
                    F = (r2 / k) / ((1 - r2) / (n - k - 1))
                    st.code(f"Overall F-stat (Z): F = {F:.3f} with df=({k}, {n-k-1})", language='text')
                    st.text("H0: All slope coefficients are 0 (no relationship).\nH1: At least one slope coefficient ≠ 0.")
                    # Optional p-value if SciPy is available
                    p_val = None
                    try:
                        if scipy_f_dist is not None and math.isfinite(F):
                            p_val = float(scipy_f_dist.sf(F, k, n - k - 1))
                    except Exception:
                        p_val = None
                    if p_val is not None:
                        st.write(f"p-value: {p_val:.4g}")
                        decision = "Reject H0" if p_val < alpha_sig else "Fail to Reject H0"
                        st.write(f"Decision (@ α = {alpha_sig}): {decision}")
                        if p_val < alpha_sig:
                            st.success("Model is statistically significant")
                        else:
                            st.info("Model is NOT statistically significant")
                    else:
                        # Heuristic interpretation without p-value
                        if F > 4.0:
                            st.success("Model likely significant (heuristic; SciPy not available). Decision: Reject H0")
                        elif F > 1.0:
                            st.info("Model shows some explanatory power; borderline without exact p-value. Decision: Inconclusive")
                        else:
                            st.warning("Model likely not significant (heuristic). Decision: Fail to Reject H0")
                else:
                    st.info(f"Insufficient or invalid variance for F-stat computation. r2={r2}.")
                # Nested F-test for degree k vs k-1
                try:
                    from curve_fitting import _design_matrix, _normal_equations, _gaussian_elimination_solve
                    x_vals = [p['x'] for p in trajectory]
                    z_vals = [p['z'] for p in trajectory]
                    def fit_and_rss(deg):
                        X = _design_matrix(x_vals, deg)
                        XTX, XTy = _normal_equations(X, z_vals)
                        coeffs = _gaussian_elimination_solve(XTX, XTy)
                        rss = 0.0
                        for xi, zi in zip(x_vals, z_vals):
                            yhat = 0.0
                            for c in reversed(coeffs):
                                yhat = yhat * xi + c
                            rss += (zi - yhat) ** 2
                        return rss
                    if poly['degree'] >= 1:
                        rss_small = fit_and_rss(poly['degree'] - 1)
                        rss_large = fit_and_rss(poly['degree'])
                        if rss_large > 0 and rss_small > 0:
                            LR = len(x_vals) * math.log(rss_small / rss_large)
                            st.code(f"Nested LR-stat (Z): {LR:.3f} (df = 1). Larger is more evidence for higher degree.", language='text')
                            # Classical nested F-test between degree k and k-1
                            df1 = 1
                            df2 = len(x_vals) - ((poly['degree'] + 1)) - 1
                            if rss_small > rss_large and df2 > 0:
                                F_nested = ((rss_small - rss_large) / df1) / (rss_large / df2)
                                st.code(f"Nested F-test: F = {F_nested:.3f} with df=({df1}, {df2})", language='text')
                                st.text("H0: Added degree term does not improve fit (coefficient = 0).\nH1: Added degree term improves fit (coefficient ≠ 0).")
                                p_nested = None
                                try:
                                    if scipy_f_dist is not None and math.isfinite(F_nested):
                                        p_nested = float(scipy_f_dist.sf(F_nested, df1, df2))
                                except Exception:
                                    p_nested = None
                                if p_nested is not None:
                                    st.write(f"p-value (k vs k-1): {p_nested:.4g}")
                                    decision_nested = "Reject H0 (keep higher degree)" if p_nested < alpha_sig else "Fail to Reject H0 (prefer lower degree)"
                                    st.write(f"Decision (@ α = {alpha_sig}): {decision_nested}")
                                    if p_nested < alpha_sig:
                                        st.success("Higher degree is justified")
                                    else:
                                        st.info("No significant improvement; consider lower degree")
                                else:
                                    if F_nested > 4.0:
                                        st.success("Higher degree likely justified (heuristic). Decision: Reject H0")
                                    else:
                                        st.info("Improvement appears modest; lower degree may suffice (heuristic). Decision: Fail to Reject H0")
                            else:
                                st.info(f"Cannot run nested F-test. Conditions: rss_small({rss_small}) > rss_large({rss_large}) and df2({df2}) > 0 must hold.")
                except Exception:
                    pass
        else:
            st.info("Polynomial model not fitted yet. Use the button above to fit the model.")

    if st.session_state.extrapolated:
        st.markdown("---")
        st.subheader("Extrapolated Points (Physics-Based)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            df_extrap = pd.DataFrame(st.session_state.extrapolated[:])
            st.dataframe(df_extrap, use_container_width=True)        
        with col2:
            extrap_distance = st.session_state.extrapolated[-1]['x'] - trajectory[-1]['x']
            st.metric("Extrapolated Distance", f"{extrap_distance:.2f} m")
            st.metric("Total Points", len(st.session_state.extrapolated))
            st.metric("Start X", f"{trajectory[-1]['x']:.2f} m")
            st.metric("End X", f"{st.session_state.extrapolated[-1]['x']:.2f} m")
    
    if st.session_state.extrapolated:
        st.markdown("---")
        st.subheader("2D Trajectory Views")
        
        fig_2d = create_2d_plots(trajectory, st.session_state.extrapolated)
        st.plotly_chart(fig_2d, use_container_width=True)

def render_lbw_decision_page():
    st.header("LBW (Leg Before Wicket) Decision System")
    
    if not st.session_state.extrapolated:
        st.warning("Please fit spline and extrapolate trajectory first")
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
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stump Position", "20.0 m")
    with col2:
        st.metric("Stump Height", "0.71 m")
    with col3:
        st.metric("Stump Width", "0.228 m")
    
    st.markdown("---")
    
    if st.button("Calculate LBW Decision (All Rules)", type="primary"):
        with st.spinner("Analyzing ball trajectory with real cricket rules..."):
            lbw_result = calculate_lbw_decision(trajectory, extrapolated)
            ball_path_analysis = analyze_ball_path(trajectory, extrapolated)
            
            st.session_state.lbw_result = lbw_result
            st.session_state.ball_path = ball_path_analysis
            
            st.session_state.deliveries_history.append({
                'trajectory': trajectory,
                'extrapolated': extrapolated,
                'scenario_type': st.session_state.scenario_type,
                'lbw_result': lbw_result
            })
            
            st.success("✅ LBW analysis complete")
    
    if st.session_state.lbw_result:
        result = st.session_state.lbw_result
        
        st.markdown("---")
        st.subheader("FINAL DECISION")
        
        # Big decision display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if result['is_out']:
                st.error("# 🚫 OUT LBW")
            else:
                st.success("# ✅ NOT OUT")
            
            st.info(f"**Decision Reason:** {result['reason']}")
        
        st.markdown("---")
        
        st.subheader("✓ Rule Verification")
        
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
            st.caption(f"Confidence: {result['confidence']:.1f}%")
        
        st.markdown("---")
        
        st.subheader("Impact Point Analysis at Stumps (20m)")
        
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
        
        if 'ball_path' in st.session_state:
            st.markdown("---")
            st.subheader("Ball Path Characteristics")
            
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
            
def render_3d_visualization_page():
    st.header("Interactive 3D Ball Trajectory Visualization")
    
    trajectory = st.session_state.trajectory
    
    st.markdown("---")
    
    fig = create_trajectory_plot(
        trajectory,
        st.session_state.extrapolated,
        st.session_state.spline_model,
        st.session_state.poly_model
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Trajectory Summary")
    
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
def render_advanced_analysis_page():

    if not st.session_state.get("trajectory"):
        st.warning("Please load trajectory data first (go to Trajectory Analysis tab).")
        return

    trajectory = st.session_state.trajectory

    st.subheader("Exponential Smoothing First and Second Order")
    method = st.selectbox("Smoothing Method", ["Exponential (First Order)", "Exponential (Second Order)"])
    alpha = st.slider("Alpha (smoothing)", 0.05, 0.95, 0.30, 0.05)

    if st.button("Analyze Decomposition", key="decomp_btn"):
        with st.spinner("Decomposing trajectory..."):
            use_method = 'SES' if method.startswith('Exponential (First') else 'BROWN'
            decomp = decompose_trajectory_advanced(trajectory, method=use_method, alpha=alpha)

            if "error" in decomp:
                st.error(decomp["error"])
                return

            st.subheader("Original vs Smoothed")

            # Original vs Smoothed plots
            x_vals = [p['x'] for p in trajectory]
            z_actual = [p['z'] for p in trajectory]
            y_actual = [p['y'] for p in trajectory]

            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=x_vals, y=z_actual, mode='lines+markers', name='Z Actual'))
            fig_z.add_trace(go.Scatter(x=x_vals, y=decomp['trend_z'], mode='lines', name='Z Smoothed'))
            fig_z.update_layout(title='Original vs Smoothed (Z)', xaxis_title='x (m)', yaxis_title='z (m)', legend_title='Series')
            st.plotly_chart(fig_z, use_container_width=True)

            fig_y = go.Figure()
            fig_y.add_trace(go.Scatter(x=x_vals, y=y_actual, mode='lines+markers', name='Y Actual'))
            fig_y.add_trace(go.Scatter(x=x_vals, y=decomp['trend_y'], mode='lines', name='Y Smoothed'))
            fig_y.update_layout(title='Original vs Smoothed (Y)', xaxis_title='x (m)', yaxis_title='y (m)', legend_title='Series')
            st.plotly_chart(fig_y, use_container_width=True)

            # Smoothing values table
            st.subheader("Smoothing Table (Exponential)")
            df_smooth = pd.DataFrame({
                'x': x_vals,
                'z_actual': z_actual,
                'z_smoothed': decomp['trend_z'],
                'y_actual': y_actual,
                'y_smoothed': decomp['trend_y'],
            })
            st.dataframe(df_smooth, use_container_width=True)

            csv = df_smooth.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Smoothing Table (CSV)",
                data=csv,
                file_name="exponential_smoothing_values.csv",
                mime="text/csv",
                key="download_smoothing_table"
            )

            
            
               
if __name__ == "__main__":
    main()