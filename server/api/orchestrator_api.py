from flask import Flask, request, jsonify
import random
import string
import time
import threading

# --- Configuration ---
app = Flask(__name__)

# --- In-Memory Database & State ---
# This will store our worker and token info.
db = {
    "workers": {
        # "worker-uuid-1234": {
        #     "status": "pending", # "pending", "approved"
        #     "current_token": "12345678",
        #     "api_key": "long-secret-key-abc",
        #     "last_seen": 1678886400,
        #     "worker_id": "worker-uuid-1234"
        # }
    },
    "token_map": {
        # "12345678": "worker-uuid-1234"
    }
}

# --- NEW: Global Training State ---
# This dictionary will manage the entire training run
global_state = {
    "status": "IDLE", # IDLE, TRAINING, PAUSED
    "current_model": "None",
    "current_dataset": "None",
    "learning_rate": 0.001,
    "current_epoch": 0,
    "total_epochs": 0,
    "task_queue": [], # This will be a list of batch_ids
    "completed_tasks": [],
    "global_model_version": 1,
    "global_model_parameters": None, # In a real app, this would be the model's state_dict
    "start_time": None,
    "loss_history": [],
    "accuracy_history": []
}

# --- NEW: Thread Lock ---
# To prevent race conditions when multiple workers access the state
state_lock = threading.Lock()

# --- Utility Functions ---
def generate_token():
    """Generates a simple 8-digit token."""
    return str(random.randint(10000000, 99999999))

def generate_api_key():
    """Generates a secure API key for an approved worker."""
    return 'key-' + ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def is_authorized(request):
    """Checks if a worker is approved and has a valid API key."""
    key = request.headers.get('X-API-Key')
    if not key:
        return None
    
    # Find the worker with this key
    for worker_id, info in db["workers"].items():
        if info.get("api_key") == key and info.get("status") == "approved":
            # Update last_seen timestamp
            with state_lock:
                db["workers"][worker_id]["last_seen"] = int(time.time())
            return worker_id # Return worker_id on success
    return None

def load_dataset(dataset_name):
    """
    --- STUB FUNCTION ---
    This is where you'd load your ABIDE dataset.
    For now, it just creates a dummy task queue.
    """
    print(f"Loading dataset: {dataset_name}...")
    # TODO: Replace with real data logic
    # We'll create 100 dummy tasks for this epoch
    return [f"batch_{i+1:04d}" for i in range(100)]


# --- Management Panel Endpoints (Called by manage_app.py) ---

@app.route("/approve-worker", methods=['POST'])
def approve_worker():
    # (Security: In a real app, this endpoint itself should be secured)
    token = request.json.get("token")
    if not token or token not in db["token_map"]:
        return jsonify({"error": "Invalid or expired token"}), 404

    with state_lock:
        worker_id = db["token_map"].get(token)
        if not worker_id or worker_id not in db["workers"]:
            return jsonify({"error": "No worker found for this token"}), 404

        worker = db["workers"][worker_id]
        worker["status"] = "approved"
        worker["api_key"] = generate_api_key()
        worker["current_token"] = None 
        
        del db["token_map"][token] 
    
    print(f"✅ Worker {worker_id} approved!")
    return jsonify({"success": True, "worker_id": worker_id, "api_key": worker["api_key"]})

@app.route("/status", methods=['GET'])
def get_status():
    """Provides full system status to the management panel."""
    # (Security: In a real app, this endpoint should be secured)
    pending_workers = []
    approved_workers = []
    
    with state_lock:
        now = int(time.time())
        for worker_id, info in db["workers"].items():
            # Check if worker is "Online" (seen in the last 30 seconds)
            is_online = (now - info.get("last_seen", 0)) < 30
            
            if info["status"] == "pending":
                pending_workers.append({
                    "worker_id": worker_id,
                    "token": info["current_token"],
                    "last_seen": info["last_seen"]
                })
            elif info["status"] == "approved":
                approved_workers.append({
                    "worker_id": worker_id,
                    "last_seen": info["last_seen"],
                    "status": "Online" if is_online else "Offline"
                })
        
        # Also send the global training state
        response_data = {
            "pending": pending_workers,
            "approved": approved_workers,
            "training_state": global_state.copy() # Send a copy
        }
    
    return jsonify(response_data)

@app.route("/start-training", methods=['POST'])
def start_training():
    """Called by the management panel to start the training run."""
    with state_lock:
        if global_state["status"] == "TRAINING":
            return jsonify({"error": "Training is already in progress"}), 400
        
        data = request.json
        global_state["status"] = "TRAINING"
        global_state["current_model"] = data.get("model_name", "Unknown")
        global_state["current_dataset"] = data.get("dataset_name", "Unknown")
        global_state["learning_rate"] = float(data.get("learning_rate", 0.001))
        global_state["total_epochs"] = int(data.get("total_epochs", 10))
        global_state["current_epoch"] = 1
        global_state["start_time"] = int(time.time())
        global_state["loss_history"] = []
        global_state["accuracy_history"] = []
        
        # --- This is the key part ---
        # Load the dataset and populate the task queue
        global_state["task_queue"] = load_dataset(global_state["current_dataset"])
        global_state["completed_tasks"] = []
        
        print(f"🚀 STARTING TRAINING: Model={global_state['current_model']}, Dataset={global_state['current_dataset']}")
        
    return jsonify({"success": True, "state": global_state})

@app.route("/stop-training", methods=['POST'])
def stop_training():
    """Called by the management panel to stop the training run."""
    with state_lock:
        if global_state["status"] == "IDLE":
            return jsonify({"error": "Training is not running"}), 400
            
        print("🛑 STOPPING TRAINING... resetting state.")
        global_state["status"] = "IDLE"
        global_state["task_queue"] = []
        global_state["completed_tasks"] = []
        global_state["start_time"] = None
        
    return jsonify({"success": True, "state": global_state})


# --- Client Worker Endpoints (Called by client.py) ---

@app.route("/heartbeat", methods=['POST'])
def heartbeat():
    data = request.json
    worker_id = data.get("worker_id")
    token = data.get("token")

    if not worker_id:
        return jsonify({"error": "worker_id is required"}), 400

    with state_lock:
        # Case 1: Brand new worker
        if worker_id not in db["workers"]:
            if not token:
                 return jsonify({"error": "token is required for new worker"}), 400
            if token in db["token_map"]:
                return jsonify({"error": "Token collision, please restart worker"}), 409
            
            print(f"👋 New worker registered: {worker_id} with token {token}")
            db["workers"][worker_id] = {
                "worker_id": worker_id,
                "status": "pending",
                "current_token": token,
                "api_key": None,
                "last_seen": int(time.time())
            }
            db["token_map"][token] = worker_id
            return jsonify({"status": "pending_approval"})

        # Case 2: Existing worker checking status
        worker = db["workers"][worker_id]
        worker["last_seen"] = int(time.time())

        if worker["status"] == "approved":
            return jsonify({"status": "approved", "api_key": worker["api_key"]})
        
        return jsonify({"status": "pending_approval"})
        

@app.route("/get-task", methods=['GET'])
def get_task():
    worker_id = is_authorized(request)
    if not worker_id:
        return jsonify({"error": "Unauthorized"}), 401

    with state_lock:
        if global_state["status"] != "TRAINING":
            return jsonify({"status": "no_tasks", "message": "Training is not active."})
        
        if not global_state["task_queue"]:
            # --- Epoch Finished! ---
            # Check if all epochs are done
            if global_state["current_epoch"] >= global_state["total_epochs"]:
                print("🎉 Training Complete!")
                global_state["status"] = "IDLE"
                global_state["start_time"] = None
                return jsonify({"status": "no_tasks", "message": "Training run complete."})
            else:
                # --- Start Next Epoch ---
                print(f"Epoch {global_state['current_epoch']} complete. Starting epoch {global_state['current_epoch'] + 1}...")
                global_state["current_epoch"] += 1
                global_state["task_queue"] = load_dataset(global_state["current_dataset"])
                global_state["completed_tasks"] = []
                # (TODO: Add logic for validation pass here)

        # Pop a task from the queue
        if global_state["task_queue"]:
            task_id = global_state["task_queue"].pop(0)
            
            task = {
                "batch_id": task_id,
                "model_version": global_state["global_model_version"],
                "model_params": global_state["global_model_parameters"], # Send current model to worker
                "learning_rate": global_state["learning_rate"]
                # TODO: Add data/labels for this batch_id
            }
            print(f"Assigning task {task_id} to worker {worker_id[:8]}...")
            return jsonify(task)
        else:
            return jsonify({"status": "no_tasks", "message": "Epoch complete, server is resetting."})

@app.route("/submit-work", methods=['POST'])
def submit_work():
    worker_id = is_authorized(request)
    if not worker_id:
        return jsonify({"error": "Unauthorized"}), 401

    results = request.json
    batch_id = results.get("batch_id")
    
    with state_lock:
        if global_state["status"] != "TRAINING":
            return jsonify({"error": "Training is not active, rejecting work"}), 400
            
        print(f"Client {worker_id[:8]} submitted work for {batch_id}")
        
        # --- TODO: Gradient Aggregation Logic ---
        # 1. Deserialize results["gradients"]
        # 2. Average them into global_state["global_model_parameters"]
        # 3. If (N) gradients received, update model and increment global_model_version
        # ---
        
        # Record loss for plotting
        global_state["loss_history"].append(results.get("loss"))
        global(f"Got loss: {results.get('loss')}")
        
        global_state["completed_tasks"].append(batch_id)
    
    return jsonify({"status": "received", "batch_id": batch_id})


if __name__ == '__main__':
    # Run this on port 10041, and Nginx will handle the rest.
    app.run(debug=True, port=10041, host='0.0.0.0')


