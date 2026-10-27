# Root app.py entry point for Streamlit Cloud
import os
import sys

# Add project root and src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# Execute the main streamlit app
app_path = os.path.join(BASE_DIR, "streamlit_app", "app.py")
if not os.path.exists(app_path):
    app_path = os.path.join(BASE_DIR, "human+activity+recognition+using+smartphones", "HAR_HMM_Project", "streamlit_app", "app.py")

with open(app_path) as f:
    code = compile(f.read(), app_path, "exec")
    exec(code, globals())
