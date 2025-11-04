from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import requests # Used to call our orchestrator API
import time
from datetime import datetime

# --- Configuration ---
app = Flask(__name__)
# This is a secret key for Flask's session/flash messages
app.config['SECRET_KEY'] = 'a-very-secret-key-for-the-management-panel' 

# This is the *internal* URL for your API.
# Since this app and the API run on the same server,
# we can call it directly via localhost.
API_URL = "http://localhost:10041" 

# --- Utility Functions ---

@app.template_filter('timestamp_to_pretty')
def timestamp_to_pretty(ts):
    """Converts a UNIX timestamp to a pretty 'time ago' string."""
    if not isinstance(ts, (int, float)):
        return "never"
    now = int(time.time())
    diff = now - int(ts)
    
    if diff < 10:
        return "just now"
    if diff < 60:
        return f"{diff} seconds ago"
    if diff < 3600:
        return f"{diff // 60} minutes ago"
    if diff < 86400:
        return f"{diff // 3600} hours ago"
    else:
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')

# --- Main Routes ---

@app.route("/", methods=['GET'])
def dashboard():
    """Main dashboard page. Fetches all data from the API."""
    
    # Data to pass to the template
    pending_workers = []
    approved_workers = []
    training_state = {}
    
    try:
        # Call our API's /status endpoint
        response = requests.get(f"{API_URL}/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            pending_workers = data.get("pending", [])
            approved_workers = data.get("approved", [])
            training_state = data.get("training_state", {})
        else:
            flash(f"Error fetching status from API: {response.status_code}", "error")
            
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API at {API_URL}. Is it running?", "error")
    except Exception as e:
        flash(f"An unknown error occurred: {e}", "error")

    return render_template("index.html", 
                           pending_workers=pending_workers, 
                           approved_workers=approved_workers,
                           training_state=training_state)

@app.route("/approve", methods=['POST'])
def approve_worker():
    """Handles the 'Approve Worker' form submission."""
    token = request.form.get("token")
    if not token:
        flash("You must provide a token.", "error")
        return redirect(url_for("dashboard"))
        
    try:
        # Call the API's /approve-worker endpoint
        payload = {"token": token.replace("-", "")} # Remove dashes if user typed them
        response = requests.post(f"{API_URL}/approve-worker", json=payload, timeout=5)
        
        if response.status_code == 200:
            flash(f"Successfully approved worker!", "success")
        else:
            error_msg = response.json().get("error", "Unknown error")
            flash(f"Failed to approve token: {error_msg}", "error")
            
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
    
    return redirect(url_for("dashboard"))

@app.route("/start-training", methods=['POST'])
def start_training():
    """Handles the 'Start Training' form submission."""
    try:
        # Collect form data
        payload = {
            "model_name": request.form.get("model_name", "ABIDE_CNN_v1"),
            "dataset_name": request.form.get("dataset_name", "ABIDE_FCM_CPAC"),
            "learning_rate": float(request.form.get("learning_rate", 0.001)),
            "total_epochs": int(request.form.get("total_epochs", 10))
        }
        
        # Call the API's /start-training endpoint
        response = requests.post(f"{API_URL}/start-training", json=payload, timeout=5)
        
        if response.status_code == 200:
            flash("Training run started successfully!", "success")
        else:
            error_msg = response.json().get("error", "Unknown error")
            flash(f"Failed to start training: {error_msg}", "error")
            
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
    except Exception as e:
        flash(f"An error occurred: {e}", "error")
        
    return redirect(url_for("dashboard"))

@app.route("/stop-training", methods=['POST'])
def stop_training():
    """Handles the 'Stop Training' button press."""
    try:
        response = requests.post(f"{API_URL}/stop-training", timeout=5)
        
        if response.status_code == 200:
            flash("Training run stopped successfully!", "success")
        else:
            error_msg = response.json().get("error", "Unknown error")
            flash(f"Failed to stop training: {error_msg}", "error")

    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))

# --- NEW: Chart Data Endpoint ---
# This endpoint is for our dashboard's live charts
@app.route("/chart-data", methods=['GET'])
def chart_data():
    """Provides just the chart data (loss/accuracy) to the dashboard."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=2)
        if response.status_code == 200:
            state = response.json().get("training_state", {})
            data = {
                "loss": state.get("loss_history", []),
                "accuracy": state.get("accuracy_history", [])
            }
            return jsonify(data)
    except:
        # Fail silently
        pass
    return jsonify({"loss": [], "accuracy": []})


if __name__ == '__main__':
    # Run this on port 10040
    app.run(debug=True, port=10040, host='0.0.0.0')


