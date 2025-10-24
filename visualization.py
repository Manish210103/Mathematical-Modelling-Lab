import plotly.graph_objects as go

# Create interactive 3D trajectory plot
def create_trajectory_plot(tracked_trajectory, extrapolated_trajectory=None, spline_model=None, poly_model=None):
    
    fig = go.Figure()
    
    if tracked_trajectory:
        x_tracked = [p['x'] for p in tracked_trajectory]
        y_tracked = [p['y'] for p in tracked_trajectory]
        z_tracked = [p['z'] for p in tracked_trajectory]
        
        fig.add_trace(go.Scatter3d(
            x=x_tracked,
            y=y_tracked,
            z=z_tracked,
            mode='lines+markers',
            name='Tracked Trajectory',
            line=dict(color='red', width=8),
            marker=dict(size=4, color='red'),
            hovertemplate='<b>Tracked</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>'
        ))
    
    if extrapolated_trajectory:
        x_extrap = [p['x'] for p in extrapolated_trajectory]
        y_extrap = [p['y'] for p in extrapolated_trajectory]
        z_extrap = [p['z'] for p in extrapolated_trajectory]
        
        fig.add_trace(go.Scatter3d(
            x=x_extrap,
            y=y_extrap,
            z=z_extrap,
            mode='lines+markers',
            name='Extrapolated (Physics)',
            line=dict(color='blue', width=8, dash='dash'),
            marker=dict(size=3, color='blue', opacity=0.6),
            hovertemplate='<b>Extrapolated</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>'
        ))
    
    if spline_model and tracked_trajectory:
        from curve_fitting import evaluate_spline
        x_spline = []
        y_spline = []
        z_spline = []
        
        x_min = tracked_trajectory[0]['x']
        x_max = tracked_trajectory[-1]['x']
        x_range = x_max - x_min
        
        for i in range(100):
            x_val = x_min + (x_range * i / 99)
            z_val = evaluate_spline(spline_model, x_val, 'z')
            y_val = evaluate_spline(spline_model, x_val, 'y')
            
            if z_val is not None and y_val is not None:
                x_spline.append(x_val)
                y_spline.append(y_val)
                z_spline.append(z_val)
        
        fig.add_trace(go.Scatter3d(
            x=x_spline,
            y=y_spline,
            z=z_spline,
            mode='lines',
            name='Cubic Spline Fit',
            line=dict(color='green', width=4, dash='dot'),
            opacity=0.7,
            hovertemplate='<b>Spline Fit</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>'
        ))

    if poly_model and tracked_trajectory:
        from curve_fitting import evaluate_polynomial
        x_poly, y_poly, z_poly = [], [], []
        x_min = tracked_trajectory[0]['x']
        x_max = tracked_trajectory[-1]['x']
        x_range = x_max - x_min
        for i in range(100):
            x_val = x_min + (x_range * i / 99)
            z_val = evaluate_polynomial(poly_model, x_val, 'z')
            y_val = evaluate_polynomial(poly_model, x_val, 'y')
            if z_val is not None and y_val is not None:
                x_poly.append(x_val)
                y_poly.append(y_val)
                z_poly.append(z_val)
        fig.add_trace(go.Scatter3d(
            x=x_poly,
            y=y_poly,
            z=z_poly,
            mode='lines',
            name='Polynomial Fit',
            line=dict(color='orange', width=4),
            opacity=0.7,
            hovertemplate='<b>Polynomial Fit</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>'
        ))
    
    fig.add_trace(go.Mesh3d(
        x=[0, 20, 20, 0],
        y=[-1.5, -1.5, 1.5, 1.5],
        z=[0, 0, 0, 0],
        color='lightgreen',
        opacity=0.3,
        name='Pitch',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    STUMP_X = 20.0
    STUMP_HEIGHT = 0.71
    stump_positions = [-0.114, 0, 0.114]  # Left, middle, right stumps
    
    for i, y_pos in enumerate(stump_positions):
        fig.add_trace(go.Scatter3d(
            x=[STUMP_X, STUMP_X],
            y=[y_pos, y_pos],
            z=[0, STUMP_HEIGHT],
            mode='lines',
            name='Stumps' if i == 0 else None,
            showlegend=(i == 0),
            line=dict(color='gold', width=12),
            hovertemplate=f'<b>Stump</b><br>X: {STUMP_X}m<br>Height: {STUMP_HEIGHT}m<extra></extra>'
        ))
    
    stump_box_x = [STUMP_X]*8
    stump_box_y = [-0.15, 0.15, 0.15, -0.15, -0.15, 0.15, 0.15, -0.15]
    stump_box_z = [0, 0, 0, 0, STUMP_HEIGHT, STUMP_HEIGHT, STUMP_HEIGHT, STUMP_HEIGHT]
    
    fig.add_trace(go.Mesh3d(
        x=stump_box_x,
        y=stump_box_y,
        z=stump_box_z,
        color='yellow',
        opacity=0.15,
        name='Stump Zone',
        showlegend=True,
        hoverinfo='skip',
        i=[0, 0, 0, 0, 4, 4, 4, 4, 0, 1, 2, 3],
        j=[1, 2, 3, 4, 5, 6, 7, 0, 4, 5, 6, 7],
        k=[2, 3, 0, 5, 6, 7, 4, 1, 5, 6, 7, 4]
    ))
    
    if tracked_trajectory:
        batsman_x = tracked_trajectory[-1]['x']
        fig.add_trace(go.Scatter3d(
            x=[batsman_x, batsman_x],
            y=[0, 0],
            z=[0, 1.8],
            mode='lines+markers',
            name='Batsman Position',
            line=dict(color='white', width=14),
            marker=dict(size=8, color='white', symbol='diamond'),
            hovertemplate=f'<b>Batsman</b><br>X: {batsman_x:.1f}m<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 2.0],
        mode='lines+markers',
        name='Bowler',
        line=dict(color='cyan', width=14),
        marker=dict(size=8, color='cyan', symbol='diamond'),
        hovertemplate='<b>Bowler Release</b><br>X: 0m<extra></extra>'
    ))
    
    if extrapolated_trajectory:
        # Find impact point closest to stumps
        impact_point = None
        for p in extrapolated_trajectory:
            if p['x'] >= STUMP_X - 0.1:
                impact_point = p
                break
        
        if impact_point:
            fig.add_trace(go.Scatter3d(
                x=[impact_point['x']],
                y=[impact_point['y']],
                z=[impact_point['z']],
                mode='markers',
                name='Impact Point',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(color='white', width=3)
                ),
                hovertemplate=f'<b>Impact Point</b><br>X: {impact_point["x"]:.3f}m<br>Y: {impact_point["y"]:.3f}m<br>Z: {impact_point["z"]:.3f}m<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': 'Cricket Ball Trajectory Analysis - DRS System',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white', 'family': 'Arial Black'}
        },
        scene=dict(
            xaxis=dict(
                title='Distance (m) → Stumps',
                backgroundcolor='rgb(20, 50, 20)',
                gridcolor='white',
                showbackground=True,
                range=[0, 21]
            ),
            yaxis=dict(
                title='Lateral Deviation (m)',
                backgroundcolor='rgb(20, 50, 20)',
                gridcolor='white',
                showbackground=True,
                range=[-1, 1]
            ),
            zaxis=dict(
                title='Height (m)',
                backgroundcolor='rgb(100, 150, 200)',
                gridcolor='white',
                showbackground=True,
                range=[0, 2.5]
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=2.5, y=0.6, z=0.5)
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=2,
            font=dict(color='white', size=11)
        ),
        paper_bgcolor='rgb(10, 40, 10)',
        plot_bgcolor='rgb(10, 40, 10)',
        height=750,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig

# Create 2D side view plots
def create_2d_plots(tracked_trajectory, extrapolated_trajectory=None):
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Side View (X-Z)'),
        specs=[[{'type': 'scatter'}]]
    )
    
    if tracked_trajectory:
        x_tracked = [p['x'] for p in tracked_trajectory]
        z_tracked = [p['z'] for p in tracked_trajectory]
        
        fig.add_trace(
            go.Scatter(x=x_tracked, y=z_tracked, mode='lines+markers',
                      name='Tracked', line=dict(color='red', width=3),
                      marker=dict(size=6)),
            row=1, col=1
        )
    
    if extrapolated_trajectory:
        x_extrap = [p['x'] for p in extrapolated_trajectory]
        z_extrap = [p['z'] for p in extrapolated_trajectory]
        
        fig.add_trace(
            go.Scatter(x=x_extrap, y=z_extrap, mode='lines+markers',
                      name='Extrapolated', line=dict(color='blue', width=3, dash='dash'),
                      marker=dict(size=5)),
            row=1, col=1
        )
    
    fig.add_hline(y=0.71, line_dash="dash", line_color="gold", 
                  annotation_text="Stump Height", row=1, col=1)
    fig.add_vline(x=20.0, line_dash="dash", line_color="gold",
                  annotation_text="Stumps", row=1, col=1)
    
    fig.update_xaxes(title_text="Distance (m)", row=1, col=1)
    fig.update_yaxes(title_text="Height (m)", row=1, col=1)
    
    fig.update_layout(height=400, showlegend=True)
    
    return fig