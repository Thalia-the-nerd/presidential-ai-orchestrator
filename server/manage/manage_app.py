from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session, send_file
import requests # Used to call our orchestrator API
import time
from datetime import datetime
import json
import io

# --- Configuration ---
app = Flask(__name__)
# This is a secret key for Flask's session/flash messages
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this' 

# This is the *internal* URL for your API.
# Since this app and the API run on the same server,
# we can call it directly via localhost.
API_URL = "http://localhost:10041" 

# --- NEW: Simple Password Protection ---
# In a real app, use Flask-Login, hashing, and a database.
# For this project, we'll use a simple password stored in the session.
ADMIN_PASSWORD = "Mcds2024" # <-- CHANGE THIS!

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

@app.route("/login", methods=['GET', 'POST'])
def login():
    """Shows the login page."""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['logged_in'] = True
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Incorrect password.", "error")
    # This will render the login.html file from your Canvas
    return render_template("login.html") 

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route("/", methods=['GET'])
def dashboard():
    """Main dashboard page. Fetches all data from the API."""
    
    # Check if logged in
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
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
        flash(f"Error: Cannot connect to the Orchestrator API at {API_URL}. Is it running?", "error")
    except Exception as e:
        flash(f"An unknown error occurred: {e}", "error")

    # Sort workers by nickname or ID for a stable list
    approved_workers.sort(key=lambda w: w.get('nickname') or w.get('worker_id'))

    # This will render the main index.html (which we'll create next)
    return render_template("index.html", 
                           pending_workers=pending_workers, 
                           approved_workers=approved_workers,
                           training_state=training_state)

@app.route("/approve", methods=['POST'])
def approve_worker():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    token = request.form.get("token")
    if not token:
        flash("You must provide a token.", "error")
        return redirect(url_for("dashboard"))
        
    try:
        payload = {"token": token.replace("-", "")} # Remove dashes
        response = requests.post(f"{API_URL}/approve-worker", json=payload, timeout=5)
        
        if response.status_code == 200:
            flash(f"Successfully approved worker!", "success")
        else:
            flash(f"Failed to approve token: {response.json().get('error', 'Unknown')}", "error")
            
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
    
    return redirect(url_for("dashboard"))

@app.route("/start-training", methods=['POST'])
def start_training():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        # Parse hyperparameters from text area
        hyperparams_str = request.form.get("hyperparameters", "{}")
        try:
            hyperparams_json = json.loads(hyperparams_str)
        except json.JSONDecodeError:
            flash("Hyperparameters are not valid JSON.", "error")
            return redirect(url_for("dashboard"))
            
        payload = {
            "model_name": request.form.get("model_name"),
            "dataset_name": request.form.get("dataset_name"),
            "learning_rate": float(request.form.get("learning_rate")),
            "total_epochs": int(request.form.get("total_epochs")),
            "hyperparameters": hyperparams_json # Send the parsed JSON
        }
        
        response = requests.post(f"{API_URL}/start-training", json=payload, timeout=5)
        
        if response.status_code == 200:
            flash("Training run started successfully!", "success")
        else:
            flash(f"Failed to start training: {response.json().get('error', 'Unknown')}", "error")
            
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
    except Exception as e:
        flash(f"An error occurred: {e}", "error")
        
    return redirect(url_for("dashboard"))

@app.route("/stop-training", methods=['POST'])
def stop_training():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        response = requests.post(f"{API_URL}/stop-training", timeout=5)
        if response.status_code == 200:
            flash("Training run stopped successfully!", "success")
        else:
            flash(f"Failed to stop training: {response.json().get('error', 'Unknown')}", "error")
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))

@app.route("/set-global-state", methods=['POST'])
def set_global_state():
    """Handles global Pause/Resume buttons."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        payload = {"state": request.form.get("state")}
        response = requests.post(f"{API_URL}/set-global-state", json=payload, timeout=5)
        if response.status_code == 200:
            flash(f"Global state set to {payload['state']}", "success")
        else:
            flash(f"Failed: {response.json().get('error', 'Unknown')}", "error")
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))
    
@app.route("/set-worker-state", methods=['POST'])
def set_worker_state():
    """Handles per-worker Pause/Resume buttons."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        payload = {
            "worker_id": request.form.get("worker_id"),
            "state": request.form.get("state")
        }
        response = requests.post(f"{API_URL}/set-worker-state", json=payload, timeout=5)
        if response.status_code != 200:
            flash(f"Failed: {response.json().get('error', 'Unknown')}", "error")
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))

@app.route("/kick-worker", methods=['POST'])
def kick_worker():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        payload = {"worker_id": request.form.get("worker_id")}
        response = requests.post(f"{API_URL}/kick-worker", json=payload, timeout=5)
        if response.status_code == 200:
            flash("Worker kicked successfully.", "success")
        else:
            flash(f"Failed: {response.json().get('error', 'Unknown')}", "error")
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))

@app.route("/set-worker-nickname", methods=['POST'])
def set_worker_nickname():
    """Handles the AJAX request from the nickname modal."""
    if not session.get('logged_in'):
        return jsonify({"error": "Not authenticated"}), 401
        
    try:
        payload = request.json
        response = requests.post(f"{API_URL}/set-worker-nickname", json=payload, timeout=5)
        if response.status_code == 200:
            return jsonify({"success": True})
        else:
            return jsonify({"error": response.json().get('error', 'Unknown')}), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to API"}), 500

@app.route("/save-checkpoint", methods=['POST'])
def save_checkpoint():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        response = requests.post(f"{API_URL}/save-checkpoint", timeout=5)
        if response.status_code == 200:
            flash(f"Checkpoint saved: {response.json().get('checkpoint_name')}", "success")
        else:
            flash(f"Failed: {response.json().get('error', 'Unknown')}", "error")
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))

@app.route("/rollback-checkpoint", methods=['POST'])
def rollback_checkpoint():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        payload = {"checkpoint_name": request.form.get("checkpoint_name")}
        response = requests.post(f"{API_URL}/rollback-checkpoint", json=payload, timeout=5)
        if response.status_code == 200:
            flash(f"Rollback to {payload['checkpoint_name']} initiated.", "success")
        else:
            flash(f"Failed: {response.json().get('error', 'Unknown')}", "error")
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))

@app.route("/download-checkpoint/<path:name>", methods=['GET'])
def download_checkpoint(name):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    try:
        # This streams the download from the API
        response = requests.get(f"{API_URL}/download-checkpoint/{name}", stream=True, timeout=10)
        if response.status_code == 200:
            # Pass the raw response back to the user's browser
            return (response.content, response.status_code, response.headers.items())
        else:
            flash(f"Failed to download: {response.json().get('error', 'Unknown')}", "error")
    except requests.exceptions.ConnectionError:
        flash("Error: Cannot connect to the Orchestrator API.", "error")
        
    return redirect(url_for("dashboard"))

# --- Chart Data Endpoint ---
@app.route("/chart-data", methods=['GET'])
def chart_data():
    """Provides just the chart data (loss/accuracy) to the dashboard."""
    if not session.get('logged_in'):
        return jsonify({"error": "Not authenticated"}), 401
        
    try:
        response = requests.get(f"{API_URL}/status", timeout=2)
        if response.status_code == 200:
            state = response.json().get("training_state", {})
            data = {
                "loss": state.get("loss_history", []),
                "accuracy": state.get("accuracy_history", []),
                "confusion_matrix": state.get("confusion_matrix", [[0,0],[0,0]])
            }
            return jsonify(data)
    except:
        pass # Fail silently on chart updates
    return jsonify({"loss": [], "accuracy": [], "confusion_matrix": [[0,0],[0,0]]})


if __name__ == '__main__':
    # Run this on port 10040
    app.run(debug=True, port=10040, host='0.0.0.0')


