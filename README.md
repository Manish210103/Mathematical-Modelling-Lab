# DRS-Lite System - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [User Guide](#user-guide)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 2GB minimum
- **OS**: Windows, macOS, or Linux

### Check Python Version
```bash
python --version
# or
python3 --version
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/)

---

## Installation

### Step 1: Create Project Directory
```bash
mkdir drs-lite-system
cd drs-lite-system
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Create Project Files

Create the following files in your project directory:

#### 1. requirements.txt
```
streamlit==1.28.0
pandas==2.0.3
```

#### 2. app.py
Copy the complete `app.py` code from the main artifact.

#### 3. utils.py (Optional)
Copy the `utils.py` code if you want additional functionality.

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

Verify installation:
```bash
pip list
```

You should see `streamlit` and `pandas` in the list.

---

## Running the Application

### Quick Start
```bash
streamlit run app.py
```

The application will:
1. Start a local web server
2. Automatically open in your default browser at `http://localhost:8501`
3. Display the DRS-Lite interface

### Alternative: Specify Port
```bash
streamlit run app.py --server.port 8080
```

### Run in Background
```bash
# Linux/macOS
nohup streamlit run app.py &

# Windows (PowerShell)
Start-Process streamlit -ArgumentList "run app.py" -WindowStyle Hidden
```

---

## User Guide

### Module 1: Data Generation 📊

**Purpose**: Generate synthetic 3D ball trajectories using physics

**Steps**:
1. Navigate to "📊 Data Generation" tab
2. Adjust parameters:
   - **Release Speed** (20-45 m/s): Higher = faster ball
   - **Elevation Angle** (-5° to 15°): Positive = upward trajectory
   - **Azimuth Angle** (-10° to 10°): Lateral direction
   - **Noise Level** (0-0.1): Measurement error simulation
   - **Spin Rate** (0-50 rpm): Ball rotation
3. Click "🎲 Generate New Trajectory"
4. View:
   - Generated data points (first 20 rows)
   - Side view (X-Z plane)
   - Top view (X-Y plane)
   - Summary statistics

**Tips**:
- Start with default values (Speed: 35, Elevation: 3°)
- Low noise (0.02) = cleaner data
- High noise (0.08) = more realistic measurement errors

---

### Module 2: Trajectory Fitting 📈

**Purpose**: Fit mathematical models to trajectory data

**Steps**:
1. Generate trajectory first (Module 1)
2. Navigate to "📈 Trajectory Fitting"
3. Click "🔬 Fit All Models"
4. Compare results:
   - **Physics Model**: Based on projectile motion equations
   - **Polynomial Model**: Cubic polynomial fit
5. Review:
   - Model parameters
   - RMSE (lower = better fit)
   - Residual plots

**Understanding Results**:
- **RMSE < 0.05**: Excellent fit
- **RMSE 0.05-0.10**: Good fit
- **RMSE > 0.10**: Poor fit (increase data quality)

**Model Selection**:
- Physics model: Better for extrapolation
- Polynomial model: Better for interpolation

---

### Module 3: LBW Analysis 🎯

**Purpose**: Calculate probability of LBW (ball hitting stumps)

**Steps**:
1. Fit models first (Module 2)
2. Navigate to "🎯 LBW Analysis"
3. Review stump configuration
4. Adjust "Prediction Uncertainty" (0.01-0.1)
5. Click "🎯 Calculate LBW Probability"
6. View results:
   - Decision: OUT or NOT OUT
   - LBW Probability (%)
   - Ball position at stump plane
   - Lateral and vertical checks

**Interpreting Results**:
- **Probability > 50%**: Likely OUT
- **Probability 40-60%**: Marginal call
- **Probability < 40%**: Likely NOT OUT

**Decision Criteria**:
- ✓ Lateral check: Ball within stump width (±0.114m)
- ✓ Vertical check: Ball between 0-0.71m height
- Both checks must pass for OUT decision

---

### Module 4: Win Probability 🏆

**Purpose**: Calculate batting team's win probability

**Steps**:
1. Navigate to "🏆 Win Probability"
2. Enter current match state:
   - Current Score
   - Wickets Lost
   - Overs Remaining
   - Current Run Rate
3. Enter target information:
   - Target Score
4. Click "📊 Calculate Win Probability"
5. Review:
   - Win probability percentage
   - Required run rate
   - Resource availability
   - What-if scenarios

**Understanding Probability**:
- **>70%**: Strong winning position
- **50-70%**: Slight advantage
- **30-50%**: Close contest
- **<30%**: Difficult situation

**Key Metrics**:
- **Required RR**: Must maintain this run rate to win
- **Resource Available**: Remaining wickets and overs
- **Runs Needed**: Target minus current score

---

## Troubleshooting

### Issue 1: Module Not Found Error
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**:
```bash
pip install --upgrade streamlit pandas
```

### Issue 2: Port Already in Use
```
Port 8501 is already in use
```

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

### Issue 3: Browser Doesn't Open
**Solution**:
Manually open: `http://localhost:8501`

### Issue 4: Trajectory Generation Fails
**Possible Causes**:
- Invalid parameter ranges
- Insufficient memory

**Solution**:
- Reset to default values
- Reduce noise level
- Restart application

### Issue 5: Division by Zero Error
**Solution**:
- Ensure trajectory has multiple points
- Check that time values are increasing
- Regenerate trajectory with different parameters

---

## Advanced Usage

### Custom Data Generation

Modify parameters in `app.py`:

```python
# Change constants
GRAVITY = 9.81  # Adjust for different planets!
STUMP_X = 20.12  # Change pitch length
TIME_STEP = 0.012  # Adjust temporal resolution
```

### Export Data

Add to your code:
```python
import csv

# Export trajectory
with open('trajectory.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['t', 'x', 'y', 'z'])
    writer.writeheader()
    writer.writerows(st.session_state.trajectory)
```

### Batch Processing

Generate multiple trajectories:
```python
for i in range(10):
    trajectory = generate_trajectory(
        release_speed=30 + i,
        elevation_angle=2,
        azimuth_angle=0,
        noise_level=0.02
    )
    # Save or process each trajectory
```

### Custom Visualizations

Using Plotly (add to requirements.txt):
```python
import plotly.graph_objects as go

# 3D trajectory plot
fig = go.Figure(data=[go.Scatter3d(
    x=[p['x'] for p in trajectory],
    y=[p['y'] for p in trajectory],
    z=[p['z'] for p in trajectory],
    mode='lines+markers'
)])
st.plotly_chart(fig)
```

---

## Performance Optimization

### For Large Datasets
```python
# Reduce Monte Carlo samples
MC_SAMPLES = 500  # Instead of 1000

# Reduce time step
TIME_STEP = 0.024  # Instead of 0.012
```

### Memory Management
```python
# Clear session state
if st.button("Clear All Data"):
    st.session_state.clear()
    st.rerun()
```

---

## API Reference

### Key Functions

#### generate_trajectory()
```python
trajectory = generate_trajectory(
    release_speed=35.0,      # m/s
    elevation_angle=5.0,     # degrees
    azimuth_angle=0.0,       # degrees
    noise_level=0.02,        # meters
    spin_rate=20             # rpm (optional)
)
```

Returns: List of dicts with keys: `t`, `x`, `y`, `z`, `frame`

#### fit_physics_model()
```python
model = fit_physics_model(trajectory)
```

Returns: Dict with keys: `x0`, `vx`, `z0`, `vz`, `a`, `rmse_x`, `rmse_z`

#### calculate_lbw_probability()
```python
result = calculate_lbw_probability(
    trajectory,
    physics_model,
    poly_model,
    noise_level=0.02
)
```

Returns: Dict with LBW decision and metrics

#### calculate_win_probability()
```python
win_result = calculate_win_probability(
    current_score=120,
    wickets=3,
    overs_remaining=8.0,
    target_score=165,
    current_run_rate=7.5
)
```

Returns: Dict with win probability and analysis

---

## Mathematical Background

### Projectile Motion
```
x(t) = x₀ + vₓt
z(t) = z₀ + vᵤt - ½gt²
```

### Ordinary Least Squares
```
β = (XᵀX)⁻¹Xᵀy
```

### Monte Carlo Integration
```
P(event) = (1/N) Σ I(event occurs in sample i)
```

### Win Probability Model
```
P_win = f(wickets, overs, runs_needed, run_rate)
```

---

## Testing

### Run Unit Tests
```python
# In Python console
from utils import test_mathematical_functions
test_mathematical_functions()
```

### Validate Results

Check trajectory:
```python
# All z values should be positive until bounce
assert all(p['z'] >= 0 for p in trajectory)

# Time should be monotonic
times = [p['t'] for p in trajectory]
assert times == sorted(times)
```

---

## Deployment

### Deploy to Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

---

## Support & Resources

### Documentation
- Streamlit: [docs.streamlit.io](https://docs.streamlit.io)
- Python: [python.org/doc](https://www.python.org/doc/)

### Common Commands
```bash
# Update Streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear

# Check version
streamlit --version
```

---

## License & Credits

**License**: MIT  
**Author**: DRS-Lite Development Team  
**Version**: 1.0.0  
**Last Updated**: 2025

Built with ❤️ and pure mathematics (no ML/AI)

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| Start App | `streamlit run app.py` |
| Stop App | `Ctrl + C` |
| Clear Cache | `streamlit cache clear` |
| Change Port | `--server.port 8080` |
| Help | `streamlit --help` |

**Default Parameters**:
- Speed: 35 m/s
- Elevation: 3°
- Noise: 0.02 m
- Stump Distance: 20.12 m
- Stump Height: 0.71 m