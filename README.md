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

Built with ❤️ and pure mathematics

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
