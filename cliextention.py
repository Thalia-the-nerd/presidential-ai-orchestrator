# web_dashboard.py

import redis
import json
import time
import os
from datetime import datetime
from flask import Flask, jsonify, render_template_string

# pip install Flask

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WORKER_TIMEOUT_SECONDS = 30

# --- Flask App Setup ---
app = Flask(__name__)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# --- HTML, CSS & JavaScript Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroLearn Dashboard</title>
    <!-- 1. Include Chart.js library -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: auto; background-color: #1e1e1e; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.5); }
        h1, h2 { text-align: center; color: #4a90e2; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #333; }
        th { color: #bb86fc; }
        td span { font-weight: bold; color: #f2f2f2; }
        .status-running { color: #00ff7f; } .status-paused { color: #ffd700; } .status-offline { color: #ff4500; }
        .controls { text-align: center; margin-top: 30px; display: flex; justify-content: center; gap: 15px; }
        button { background-color: #4a90e2; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }
        button:hover { background-color: #357abd; } button.reset { background-color: #c0392b; } button.reset:hover { background-color: #a93226; }
        #workers-list { list-style: none; padding: 0; } #workers-list li { background-color: #282828; padding: 8px; margin-top: 5px; border-radius: 4px; font-family: monospace; }
        /* 2. Style the graph container */
        #chart-container { width: 100%; margin-top: 30px; background-color: #282828; padding: 15px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Project NeuroLearn - Live Dashboard</h1>
        <table>
            <!-- Table content is the same -->
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Timestamp</td><td><span id="timestamp">--</span></td></tr>
            <tr><td>Orchestrator Status</td><td><span id="orchestrator_status">--</span></td></tr>
            <tr><td>Training Control</td><td><span id="training_status">--</span></td></tr>
            <tr><td>Current Epoch</td><td><span id="epoch_counter">--</span></td></tr>
            <tr><td>Model Updates (Total)</td><td><span id="model_updates">--</span></td></tr>
            <tr><td>Tasks in Queue (To Do)</td><td><span id="task_queue">--</span></td></tr>
            <tr><td>Gradients in Queue (To Process)</td><td><span id="results_queue">--</span></td></tr>
            <tr><td>Last Validation Accuracy</td><td><span id="val_accuracy">--</span></td></tr>
        </table>
        
        <!-- 3. Add the canvas for the chart -->
        <div id="chart-container">
            <canvas id="accuracyChart"></canvas>
        </div>

        <h2>Active Workers (<span id="worker_count">0</span>)</h2>
        <ul id="workers-list"></ul>
        <div class="controls">
            <button onclick="sendCommand('resume')">Resume</button>
            <button onclick="sendCommand('pause')">Pause</button>
            <button class="reset" onclick="sendCommand('reset')">Reset System</button>
        </div>
    </div>

    <script>
        let accuracyChart; // 4. Global variable to hold the chart instance

        function updateDashboard() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('timestamp').textContent = data.timestamp;
                    updateStatus('orchestrator_status', data.orchestrator_status);
                    updateStatus('training_status', data.training_status);
                    document.getElementById('epoch_counter').textContent = data.epoch_counter;
                    document.getElementById('model_updates').textContent = data.model_updates;
                    document.getElementById('task_queue').textContent = data.task_queue;
                    document.getElementById('results_queue').textContent = data.results_queue;
                    document.getElementById('val_accuracy').textContent = data.val_accuracy;

                    document.getElementById('worker_count').textContent = data.active_workers.length;
                    const workersList = document.getElementById('workers-list');
                    workersList.innerHTML = '';
                    data.active_workers.forEach(worker => {
                        const li = document.createElement('li'); li.textContent = worker; workersList.appendChild(li);
                    });
                    
                    // 5. Update the chart with historical data
                    updateChart(data.validation_history);
                });
        }
        function updateStatus(elementId, statusText) {
            const el = document.getElementById(elementId); el.textContent = statusText; el.className = '';
            if (statusText.toLowerCase().includes('running')) { el.classList.add('status-running'); }
            else if (statusText.toLowerCase().includes('paused')) { el.classList.add('status-paused'); }
            else { el.classList.add('status-offline'); }
        }
        function sendCommand(command) {
            if (command === 'reset' && !confirm('Are you sure you want to reset the system?')) { return; }
            fetch('/' + command, { method: 'POST' }).then(response => response.json()).then(data => {
                console.log(data.message); updateDashboard();
            });
        }

        // 6. Function to initialize or update the chart
        function updateChart(history) {
            const ctx = document.getElementById('accuracyChart').getContext('2d');
            
            // Prepare data for the chart
            const labels = history.map(item => 'Epoch ' + item.epoch);
            const accuracyData = history.map(item => item.accuracy);

            if (!accuracyChart) { // If chart doesn't exist, create it
                accuracyChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Validation Accuracy (%)',
                            data: accuracyData,
                            borderColor: '#00ff7f',
                            backgroundColor: 'rgba(0, 255, 127, 0.2)',
                            fill: true,
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { min: 0, max: 100, ticks: { color: '#e0e0e0' } },
                            x: { ticks: { color: '#e0e0e0' } }
                        },
                        plugins: { legend: { labels: { color: '#e0e0e0' } } }
                    }
                });
            } else { // If chart exists, just update its data
                accuracyChart.data.labels = labels;
                accuracyChart.data.datasets[0].data = accuracyData;
                accuracyChart.update();
            }
        }

        setInterval(updateDashboard, 2000);
        document.addEventListener('DOMContentLoaded', updateDashboard);
    </script>
</body>
</html>
"""

# --- Backend API Routes (with one addition) ---

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/data')
def get_data():
    # Fetch all the same data as before...
    cutoff_time = time.time() - WORKER_TIMEOUT_SECONDS
    active_workers = redis_client.zrangebyscore("worker_heartbeats", min=cutoff_time, max="+inf")
    val_results_str = redis_client.get("last_validation_results")
    val_accuracy = "--"
    if val_results_str:
        val_results = json.loads(val_results_str)
        val_accuracy = f"{val_results.get('accuracy', 0):.2f}%"

    # ... and also fetch the validation history
    history_raw = redis_client.lrange("validation_history", 0, -1)
    validation_history = [json.loads(item) for item in history_raw]

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "orchestrator_status": redis_client.get("orchestrator_status") or "OFFLINE",
        "training_status": redis_client.get("training_status") or "NOT STARTED",
        "epoch_counter": redis_client.get("epoch_counter") or "0",
        "model_updates": redis_client.get("model_update_counter") or "0",
        "task_queue": redis_client.llen("task_queue"),
        "results_queue": redis_client.llen("results_queue"),
        "active_workers": active_workers,
        "val_accuracy": val_accuracy,
        "validation_history": validation_history, # Add history to the API response
    }
    return jsonify(data)

@app.route('/pause', methods=['POST'])
def pause_training(): redis_client.set("training_status", "PAUSED"); return jsonify({"status": "ok", "message": "PAUSE command sent."})
@app.route('/resume', methods=['POST'])
def resume_training(): redis_client.set("training_status", "RUNNING"); return jsonify({"status": "ok", "message": "RESUME command sent."})
@app.route('/reset', methods=['POST'])
def reset_system():
    keys_to_delete = ["task_queue", "results_queue", "model_update_counter", "worker_heartbeats", "training_status", "epoch_counter", "last_validation_results", "validation_history"]
    redis_client.delete(*keys_to_delete)
    return jsonify({"status": "ok", "message": "RESET command sent. Please restart orchestrator."})

# --- Run the App ---
if __name__ == '__main__':
    print("--- Starting Flask Web Dashboard (V2 with Graph) ---")
    print("Open your web browser and go to http://<your_server_ip>:5000")
    app.run(host='0.0.0.0', port=5000)
