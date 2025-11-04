from flask import Flask, request, jsonify, send_file
import random
import string
import time
import threading
from datetime import datetime
import io
import os
import numpy as np # NEW: For loading real data
import json # NEW: For serializing data

# --- Configuration ---
app = Flask(__name__)

# --- NEW: Configuration ---
MAX_HISTORY_LENGTH = 500 # Store the last 500 data points for charts
MAX_LOG_LENGTH = 100 # Store the last 100 log messages
BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
# NEW: Directory to store your .npy dataset files
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
    print(f"INFO: Created dataset directory at {DATASET_DIR}")
    print(f"INFO: Please put your 'X_train.npy' and 'y_train.npy' files there.")


# --- In-Memory Database & State ---
db = {
    "workers": {
        # ... worker data ...
    },
    "token_map": {
        # ... token data ...
    }
}

# --- Global Training State ---
global_state = {
    "status": "IDLE", # IDLE, TRAINING, PAUSED
    "current_model": "None",
    "current_dataset": "None",
    "learning_rate": 0.001,
    "current_epoch": 0,
    "total_epochs": 0,
    "task_queue": [], # This will now be a queue of BATCH INDICES, e.g. [[0,1,2], [3,4,5]]
    "completed_tasks": [],
    "global_model_version": 1,
    "global_model_parameters": None, # This should be the model's state_dict
    "start_time": None,
    "loss_history": [],
    "accuracy_history": [],
    "activity_log": [], # For live log feed
    "hyperparameters": {}, # For storing training config
    "checkpoints": [], # List of {"name": "...", "accuracy": 0.85, "epoch": 1}
    "confusion_matrix": [[0, 0], [0, 0]],
    
    # --- NEW: In-Memory Dataset ---
    "X_train": None, # Will hold all training matrices (e.g., (800, 200, 200))
    "y_train": None, # Will hold all training labels (e.g., (800,))
    "X_val": None,
    "y_val": None
}

# --- Thread Lock ---
state_lock = threading.Lock()

# --- Utility Functions ---

def log_activity(level, message):
    """Adds a message to the global activity log and prints it."""
    # ... (same as before) ...
    print(f"[{level}] {message}")
    with state_lock:
        log_entry = {
            "time": int(time.time()),
            "level": level,
            "message": message
        }
        global_state["activity_log"].append(log_entry)
        # Trim log if it's too long
        if len(global_state["activity_log"]) > MAX_LOG_LENGTH:
            global_state["activity_log"] = global_state["activity_log"][-MAX_LOG_LENGTH:]

def format_time_delta(seconds):
    """Converts seconds into a HH:MM:SS string."""
    # ... (same as before) ...
    if seconds is None:
        return "00:00:00"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def generate_token():
    # ... (same as before) ...
    return str(random.randint(10000000, 99999999))

def generate_api_key():
    # ... (same as before) ...
    return 'key-' + ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def is_authorized(request):
    """Checks if a worker is approved and has a valid API key."""
    # ... (same as before) ...
    key = request.headers.get('X-API-Key')
    if not key:
        return None
    
    for worker_id, info in db["workers"].items():
        if info.get("api_key") == key and info.get("status") == "approved":
            with state_lock:
                # Update last_seen timestamp
                db["workers"][worker_id]["last_seen"] = int(time.time())
            return worker_id # Return worker_id on success
    return None

# --- MODIFIED: load_dataset ---
def load_dataset(dataset_name):
    """
    Loads the dataset from the /dataset directory into memory.
    Returns a list of batch indices for the task queue.
    """
    log_activity("INFO", f"Loading dataset: {dataset_name}...")
    
    # --- TODO: Place your real dataset files here ---
    x_train_file = os.path.join(DATASET_DIR, "X_train.npy")
    y_train_file = os.path.join(DATASET_DIR, "y_train.npy")
    # x_val_file = os.path.join(DATASET_DIR, "X_val.npy")
    # y_val_file = os.path.join(DATASET_DIR, "y_val.npy")
    
    try:
        # Load data into global state
        global_state["X_train"] = np.load(x_train_file)
        global_state["y_train"] = np.load(y_train_file)
        # global_state["X_val"] = np.load(x_val_file)
        # global_state["y_val"] = np.load(y_val_file)
        
        num_samples = len(global_state["y_train"])
        log_activity("INFO", f"Successfully loaded {num_samples} training samples.")
        
    except FileNotFoundError:
        log_activity("ERROR", f"Dataset files not found in {DATASET_DIR}. Please add X_train.npy and y_train.npy.")
        log_activity("ERROR", "Using dummy data instead.")
        # Create dummy data if files not found
        global_state["X_train"] = np.random.rand(100, 50, 50) # 100 samples, 50x50 matrices
        global_state["y_train"] = np.random.randint(0, 2, 100) # 100 labels (0 or 1)
        num_samples = 100
        
    except Exception as e:
        log_activity("ERROR", f"Failed to load dataset: {e}")
        return None

    # --- Create the Task Queue ---
    # Get batch size from hyperparameters, default to 32
    batch_size = global_state["hyperparameters"].get("batch_size", 32)
    
    # Create a list of indices (0, 1, 2, ... num_samples-1)
    indices = np.arange(num_samples)
    
    # --- IMPORTANT: Shuffle the dataset at the start of each epoch ---
    np.random.shuffle(indices)
    
    # Split the shuffled indices into batches
    task_queue = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        task_queue.append(batch_indices.tolist()) # Add batch (as a list) to the queue
        
    log_activity("INFO", f"Created {len(task_queue)} tasks with batch size {batch_size}.")
    
    return task_queue


def get_current_checkpoints():
    """Scans the checkpoint directory and returns a list of files."""
    # ... (same as before) ...
    # This is a stub. In a real app, you'd scan the CHECKPOINT_DIR
    # and load metadata. For now, we just return the in-memory list.
    return global_state["checkpoints"]


# --- Management Panel Endpoints (Called by manage_app.py) ---
# ... (All endpoints from /approve-worker to /rollback-checkpoint are identical) ...

@app.route("/approve-worker", methods=['POST'])
def approve_worker():
    # ... (same as before) ...
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
    
    log_activity("EVENT", f"Worker {worker_id[:8]}... approved.")
    return jsonify({"success": True, "worker_id": worker_id, "api_key": worker["api_key"]})

@app.route("/status", methods=['GET'])
def get_status():
    """Provides full system status to the management panel."""
    # ... (same as before) ...
    pending_workers = []
    approved_workers = []
    
    with state_lock:
        now = int(time.time())
        for worker_id, info in db["workers"].items():
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
                    "status": "Online" if is_online else "Offline",
                    "desired_state": info.get("desired_state", "TRAIN"),
                    "nickname": info.get("nickname"),
                    "device": info.get("device")
                })
        
        # Calculate Time/ETA
        state_copy = global_state.copy()
        
        # --- IMPORTANT: Don't send the whole dataset in the status! ---
        # These are huge and will crash the JSON response.
        state_copy.pop("X_train", None)
        state_copy.pop("y_train", None)
        state_copy.pop("X_val", None)
        state_copy.pop("y_val", None)
        state_copy.pop("global_model_parameters", None) # Also too big
        
        if state_copy["status"] != "IDLE" and state_copy["start_time"]:
            elapsed = time.time() - state_copy["start_time"]
            state_copy["time_elapsed"] = format_time_delta(elapsed)
            
            total_tasks = len(state_copy["task_queue"]) + len(state_copy["completed_tasks"])
            completed_tasks = len(state_copy["completed_tasks"])
            if completed_tasks > 0 and total_tasks > 0:
                time_per_task = elapsed / completed_tasks
                remaining_tasks = total_tasks - completed_tasks
                eta_seconds = time_per_task * remaining_tasks
                state_copy["eta"] = format_time_delta(eta_seconds)
            else:
                state_copy["eta"] = "Calculating..."
        else:
            state_copy["time_elapsed"] = "00:00:00"
            state_copy["eta"] = "00:00:00"
            
        # Reverse log for display (newest first)
        state_copy["activity_log"] = list(reversed(state_copy["activity_log"]))
        # Get checkpoints
        state_copy["checkpoints"] = get_current_checkpoints()

        response_data = {
            "pending": pending_workers,
            "approved": approved_workers,
            "training_state": state_copy
        }
    
    return jsonify(response_data)

@app.route("/start-training", methods=['POST'])
def start_training():
    with state_lock:
        if global_state["status"] != "IDLE":
            return jsonify({"error": "Can only start training from IDLE state"}), 400
        
        data = request.json
        global_state["status"] = "TRAINING"
        global_state["current_model"] = data.get("model_name", "Unknown")
        global_state["current_dataset"] = data.get("dataset_name", "Unknown")
        global_state["learning_rate"] = float(data.get("learning_rate", 0.001))
        global_state["total_epochs"] = int(data.get("total_epochs", 10))
        global_state["hyperparameters"] = data.get("hyperparameters", {})
        global_state["current_epoch"] = 1
        global_state["start_time"] = int(time.time())
        # Reset all stats
        global_state["loss_history"] = []
        global_state["accuracy_history"] = []
        global_state["checkpoints"] = []
        global_state["confusion_matrix"] = [[0, 0], [0, 0]]
        global_state["activity_log"] = []
        
        # --- MODIFIED: Load dataset and create task queue ---
        task_queue = load_dataset(global_state["current_dataset"])
        if task_queue is None:
            # Loading failed, reset to IDLE
            global_state["status"] = "IDLE"
            return jsonify({"error": "Failed to load dataset. Check logs."}), 500
            
        global_state["task_queue"] = task_queue
        global_state["completed_tasks"] = []
        
        # Set all approved workers to "TRAIN"
        for worker_id in db["workers"]:
            if db["workers"][worker_id]["status"] == "approved":
                db["workers"][worker_id]["desired_state"] = "TRAIN"
        
    log_activity("EVENT", f"🚀 Training Started: Model={global_state['current_model']}, Epochs={global_state['total_epochs']}")
    # --- MODIFIED: Don't send the giant state back ---
    return jsonify({"success": True, "message": "Training started."})

@app.route("/stop-training", methods=['POST'])
def stop_training():
    # ... (same as before) ...
    with state_lock:
        if global_state["status"] == "IDLE":
            return jsonify({"error": "Training is not running"}), 400
            
        global_state["status"] = "IDLE"
        global_state["task_queue"] = []
        global_state["completed_tasks"] = []
        global_state["start_time"] = None
        
        # --- NEW: Clear dataset from memory ---
        global_state["X_train"] = None
        global_state["y_train"] = None
        global_state["X_val"] = None
        global_state["y_val"] = None
        
    log_activity("EVENT", "🛑 Training Stopped by user. Cleared data from memory.")
    return jsonify({"success": True, "state": global_state})

@app.route("/set-global-state", methods=['POST'])
def set_global_state():
    """Handles global Pause/Resume commands."""
    # ... (same as before) ...
    data = request.json
    new_state = data.get("state") # "TRAINING" (Resume) or "PAUSED"
    
    if new_state not in ["TRAINING", "PAUSED"]:
        return jsonify({"error": "Invalid state"}), 400
        
    with state_lock:
        if global_state["status"] == "IDLE":
            return jsonify({"error": "Training is not running"}), 400
        
        if global_state["status"] == new_state:
            return jsonify({"success": True, "state": global_state})
            
        global_state["status"] = new_state
        log_activity("EVENT", f"Global state set to {new_state}")
        
    return jsonify({"success": True, "state": global_state})

@app.route("/set-worker-state", methods=['POST'])
def set_worker_state():
    # ... (same as before) ...
    data = request.json
    worker_id = data.get("worker_id")
    new_state = data.get("state") # "TRAIN" or "PAUSE"
    
    if not worker_id or not new_state:
        return jsonify({"error": "worker_id and state are required"}), 400
    if new_state not in ["TRAIN", "PAUSE"]:
        return jsonify({"error": "Invalid state"}), 400
        
    with state_lock:
        if worker_id not in db["workers"]:
            return jsonify({"error": "Worker not found"}), 404
        if db["workers"][worker_id]["status"] != "approved":
            return jsonify({"error": "Worker is not approved"}), 400
            
        db["workers"][worker_id]["desired_state"] = new_state
        
    log_activity("INFO", f"Set worker {db['workers'][worker_id].get('nickname', worker_id[:8])} state to {new_state}")
    return jsonify({"success": True, "worker_id": worker_id, "new_state": new_state})

@app.route("/kick-worker", methods=['POST'])
def kick_worker():
    """Removes a worker from the db."""
    # ... (same as before) ...
    data = request.json
    worker_id = data.get("worker_id")
    
    with state_lock:
        if worker_id not in db["workers"]:
            return jsonify({"error": "Worker not found"}), 404
            
        nickname = db['workers'][worker_id].get('nickname', worker_id[:8])
        # Remove from db and token_map if it exists
        db["workers"].pop(worker_id)
        for token, w_id in list(db["token_map"].items()):
            if w_id == worker_id:
                del db["token_map"][token]
                break
                
    log_activity("EVENT", f"Worker {nickname} was kicked by admin.")
    return jsonify({"success": True, "worker_id": worker_id})

@app.route("/set-worker-nickname", methods=['POST'])
def set_worker_nickname():
    # ... (same as before) ...
    data = request.json
    worker_id = data.get("worker_id")
    nickname = data.get("nickname")
    
    with state_lock:
        if worker_id not in db["workers"]:
            return jsonify({"error": "Worker not found"}), 404
        
        old_nickname = db["workers"][worker_id].get("nickname", worker_id[:8])
        db["workers"][worker_id]["nickname"] = nickname
        
    log_activity("INFO", f"Worker {old_nickname} nickname set to '{nickname}'")
    return jsonify({"success": True, "old_nickname": old_nickname})

@app.route("/save-checkpoint", methods=['POST'])
def save_checkpoint():
    """Saves the current model state."""
    # ... (same as before, but with real save logic commented out) ...
    with state_lock:
        if global_state["status"] == "IDLE":
            return jsonify({"error": "No model is training"}), 400
            
        current_acc = 0.0
        if global_state["accuracy_history"]:
            current_acc = global_state["accuracy_history"][-1]
            
        name = f"ckpt_epoch_{global_state['current_epoch']}_acc_{current_acc:.2f}.pth"
        filepath = os.path.join(CHECKPOINT_DIR, name)
        
        # --- TODO: Actual model saving logic ---
        # model_params = global_state["global_model_parameters"]
        # if model_params:
        #     # This assumes model_params is a PyTorch state_dict
        #     # You'll need to import torch for this
        #     # torch.save(model_params, filepath)
        #     pass # Placeholder
        # else:
        #     log_activity("ERROR", "Model has no parameters to save")
        #     return jsonify({"error": "Model has no parameters to save"}), 400
        # ---
        
        # --- STUB: Create a dummy file ---
        try:
            with open(filepath, 'w') as f:
                f.write(json.dumps({"name": name, "accuracy": current_acc}))
        except Exception as e:
            log_activity("ERROR", f"Failed to save checkpoint file: {e}")
            return jsonify({"error": f"Failed to write file: {e}"}), 500
        # --- End Stub ---
        
        global_state["checkpoints"].append({
            "name": name,
            "accuracy": current_acc,
            "epoch": global_state["current_epoch"]
        })
        
    log_activity("EVENT", f"Checkpoint saved: {name}")
    return jsonify({"success": True, "checkpoint_name": name})

@app.route("/rollback-checkpoint", methods=['POST'])
def rollback_checkpoint():
    """Loads a checkpoint and prepares for training."""
    # ... (same as before, but with real load logic commented out) ...
    with state_lock:
        if global_state["status"] != "IDLE":
            return jsonify({"error": "Can only load a checkpoint when IDLE"}), 400
            
        name = request.json.get("checkpoint_name")
        filepath = os.path.join(CHECKPOINT_DIR, name)
        
        if not os.path.exists(filepath):
            log_activity("ERROR", f"Checkpoint file not found: {name}")
            return jsonify({"error": "Checkpoint file not found"}), 404
        
        # --- TODO: Actual model loading logic ---
        # try:
        #     # You'll need to import torch for this
        #     # model_params = torch.load(filepath) 
        #     # global_state["global_model_parameters"] = model_params
        #     # global_state["global_model_version"] += 1 
        #     pass # Placeholder
        # except Exception as e:
        #     log_activity("ERROR", f"Failed to load checkpoint: {e}")
        #     return jsonify({"error": f"Failed to load checkpoint: {e}"}), 500
        # ---
        
        # Find checkpoint info
        ckpt_info = next((c for c in global_state["checkpoints"] if c["name"] == name), None)
        
        # Set state
        global_state["current_model"] = f"Loaded_{name}"
        global_state["current_dataset"] = "Loaded_Data"
        global_state["current_epoch"] = ckpt_info.get("epoch", 0) if ckpt_info else 0
        global_state["accuracy_history"] = [ckpt_info.get("accuracy", 0)] if ckpt_info else []
        
    log_activity("EVENT", f"Rollback to {name} complete. Model is loaded and ready.")
    return jsonify({"success": True, "message": f"Loaded {name}"})
    
@app.route("/download-checkpoint/<path:name>", methods=['GET'])
def download_checkpoint(name):
    """Securely sends a checkpoint file from the checkpoint directory."""
    # ... (same as before) ...
    log_activity("INFO", f"Checkpoint download requested: {name}")
    
    filepath = os.path.join(CHECKPOINT_DIR, name)
    
    if not os.path.abspath(filepath).startswith(os.path.abspath(CHECKPOINT_DIR)):
        return jsonify({"error": "Forbidden"}), 403
        
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    return send_file(
        filepath,
        as_attachment=True,
        download_name=name,
        mimetype='application/octet-stream'
    )


# --- Client Worker Endpoints (Called by client.py) ---
# ... (heartbeat is identical) ...

@app.route("/heartbeat", methods=['POST'])
def heartbeat():
    # ... (same as before) ...
    data = request.json
    worker_id = data.get("worker_id")
    token = data.get("token")
    device_info = data.get("device_info") # Get device info

    if not worker_id:
        return jsonify({"error": "worker_id is required"}), 400

    with state_lock:
        # Case 1: Brand new worker
        if worker_id not in db["workers"]:
            if not token:
                 return jsonify({"error": "token is required for new worker"}), 400
            if token in db["token_map"]:
                return jsonify({"error": "Token collision, please restart worker"}), 409
            
            db["workers"][worker_id] = {
                "worker_id": worker_id,
                "status": "pending",
                "current_token": token,
                "api_key": None,
                "last_seen": int(time.time()),
                "desired_state": "TRAIN", # Default to TRAIN
                "nickname": None, 
                "device": device_info 
            }
            db["token_map"][token] = worker_id
            log_activity("INFO", f"New worker registering: {worker_id[:8]}... with token {token}")
            return jsonify({"status": "pending_approval"})

        # Case 2: Existing worker checking status
        worker = db["workers"][worker_id]
        worker["last_seen"] = int(time.time())
        if device_info and not worker.get("device"):
            worker["device"] = device_info

        if worker["status"] == "approved":
            return jsonify({
                "status": "approved", 
                "api_key": worker["api_key"]
            })
        
        return jsonify({"status": "pending_approval"})
        

# --- MODIFIED: /get-task ---
@app.route("/get-task", methods=['GET'])
def get_task():
    worker_id = is_authorized(request)
    if not worker_id:
        return jsonify({"error": "Unauthorized"}), 401

    with state_lock:
        # Check global state
        if global_state["status"] != "TRAINING":
            return jsonify({"status": "no_tasks", "message": f"Training status is {global_state['status']}."})
        
        # Check per-worker state
        worker_state = db["workers"][worker_id].get("desired_state", "TRAIN")
        if worker_state == "PAUSE":
            return jsonify({"status": "no_tasks", "message": "Worker is paused by admin."})
        
        if not global_state["task_queue"]:
            # --- Epoch Finished! ---
            if global_state["current_epoch"] >= global_state["total_epochs"]:
                global_state["status"] = "IDLE"
                global_state["start_time"] = None
                global_state["X_train"] = None # Clear memory
                global_state["y_train"] = None
                log_activity("EVENT", "🎉 Training Complete! All epochs finished.")
                return jsonify({"status": "no_tasks", "message": "Training run complete."})
            else:
                # --- Start Next Epoch ---
                log_activity("EVENT", f"Epoch {global_state['current_epoch']} complete. Starting epoch {global_state['current_epoch'] + 1}...")
                global_state["current_epoch"] += 1
                
                # --- MODIFIED: Reload dataset and create new task queue (shuffled) ---
                task_queue = load_dataset(global_state["current_dataset"])
                if task_queue is None:
                    global_state["status"] = "IDLE"
                    log_activity("ERROR", "Failed to reload dataset for next epoch.")
                    return jsonify({"status": "no_tasks", "message": "Error loading dataset for next epoch."})
                    
                global_state["task_queue"] = task_queue
                global_state["completed_tasks"] = []
                
                # --- STUB: Run Validation ---
                # TODO: This should be triggered, not faked.
                dummy_acc = (global_state["current_epoch"] / global_state["total_epochs"]) * 0.8 + random.uniform(-0.05, 0.05)
                global_state["accuracy_history"].append(round(dummy_acc, 4))
                if len(global_state["accuracy_history"]) > MAX_HISTORY_LENGTH:
                    global_state["accuracy_history"] = global_state["accuracy_history"][-MAX_HISTORY_LENGTH:]
                
                global_state["confusion_matrix"] = [
                    [random.randint(80, 100), random.randint(5, 20)],
                    [random.randint(5, 20), random.randint(80, 100)]
                ]

        # Pop a task from the queue
        if global_state["task_queue"]:
            # --- MODIFIED: Task is now a list of indices ---
            batch_indices = global_state["task_queue"].pop(0)
            
            # --- NEW: Get the actual data from memory ---
            try:
                # Select data from the in-memory numpy arrays
                batch_X_data = global_state["X_train"][batch_indices]
                batch_y_data = global_state["y_train"][batch_indices]
                
                # Serialize the data to be sent over JSON
                # .tolist() is the simplest way
                task = {
                    "batch_id": f"epoch_{global_state['current_epoch']}_b_{len(global_state['completed_tasks'])}",
                    "model_version": global_state["global_model_version"],
                    "model_params_base64": global_state["global_model_parameters"], # Sending params (TODO: serialize)
                    "learning_rate": global_state["learning_rate"],
                    "hyperparameters": global_state["hyperparameters"],
                    
                    # --- NEW: Send the actual data ---
                    "data_X": batch_X_data.tolist(),
                    "data_y": batch_y_data.tolist()
                }
                
                # log_activity("INFO", f"Assigning task {task['batch_id']} to worker {worker_id[:8]}...")
                return jsonify(task)
                
            except Exception as e:
                log_activity("ERROR", f"Failed to serialize batch! {e}")
                # Put the task back in the queue if serialization failed
                global_state["task_queue"].insert(0, batch_indices)
                return jsonify({"status": "no_tasks", "message": "Server error serializing task."})
        else:
            return jsonify({"status": "no_tasks", "message": "Epoch complete, server is resetting."})

@app.route("/submit-work", methods=['POST'])
def submit_work():
    # ... (same as before) ...
    worker_id = is_authorized(request)
    if not worker_id:
        return jsonify({"error": "Unauthorized"}), 401

    results = request.json
    batch_id = results.get("batch_id")
    
    with state_lock:
        if global_state["status"] != "TRAINING":
            return jsonify({"error": "Training is not active, rejecting work"}), 400
            
        # log_activity("INFO", f"Client {worker_id[:8]} submitted work for {batch_id}")
        
        # --- TODO: Gradient Aggregation Logic ---
        # 1. Deserialize results["gradients_base64"]
        # 2. Average them into a temporary buffer
        # 3. If (N) gradients received, update global_model_parameters and increment global_model_version
        # 4. For now, we'll just store the first gradients we get as the new model
        
        # --- STUB: Model Aggregation ---
        # This is a very basic "replace" strategy, not "averaging"
        # In a real system, you'd average gradients from ALL workers in a batch
        new_params = results.get("model_params_base64")
        if new_params:
            global_state["global_model_parameters"] = new_params
            global_state["global_model_version"] += 1
        # --- End Stub ---
        
        loss_value = results.get("loss")
        if loss_value is not None:
            global_state["loss_history"].append(loss_value)
            if len(global_state["loss_history"]) > MAX_HISTORY_LENGTH:
                global_state["loss_history"] = global_state["loss_history"][-MAX_HISTORY_LENGTH:]
        
        if batch_id not in global_state["completed_tasks"]:
             global_state["completed_tasks"].append(batch_id)
    
    return jsonify({"status": "received", "batch_id": batch_id})


if __name__ == '__main__':
    # Add numpy to your requirements.txt!
    # pip install flask numpy
    
    # Run this on port 10041, and Nginx will handle the rest.
    app.run(debug=True, port=10041, host='0.0.0.0')


