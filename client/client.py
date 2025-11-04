import requests
import time
import os
import json
import uuid
import torch # We import torch to get the GPU/CPU detection

# --- Configuration ---
# This is the *public* URL for your orchestrator API
# Make sure this domain points to your home server's Nginx proxy
ORCHESTRATOR_URL = "https://api.presidentialAIchallenge.thaliathenerd.dev" 

# File to store this worker's unique ID and API key
CONFIG_FILE = "worker.json" 

# --- Worker State ---
worker_config = {
    "worker_id": None,
    "api_key": None
}

# --- Utility Functions ---

def load_or_create_config():
    """Loads worker config or creates a new one if not found."""
    global worker_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                worker_config = json.load(f)
                if "worker_id" in worker_config and "api_key" in worker_config:
                    print(f"✅ Config loaded. Worker ID: {worker_config['worker_id'][:8]}...")
                    return True
        except Exception as e:
            print(f"Error loading config, creating new one. Error: {e}")
            os.remove(CONFIG_FILE) # Corrupted file
    
    # Create new config
    worker_config["worker_id"] = str(uuid.uuid4())
    worker_config["api_key"] = None # Will get this from server
    save_config()
    print(f"👋 New worker config created. Worker ID: {worker_config['worker_id']}")
    return False

def save_config():
    """Saves the current config to disk."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(worker_config, f, indent=4)
    except Exception as e:
        print(f"CRITICAL: Could not save config file! Error: {e}")

def get_device():
    """Detects and returns the best available PyTorch device."""
    if torch.cuda.is_available():
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ No GPU found, will use CPU. Training will be slow.")
        return torch.device("cpu")

def register_and_get_api_key():
    """
    Called on startup if we don't have an API key.
    Generates a token and pings /heartbeat until approved.
    """
    global worker_config
    token = str(random.randint(10000000, 99999999))
    print(f"\n--- NEW WORKER REGISTRATION ---")
    print(f"Please approve this PC in your management panel using the token:")
    print(f"TOKEN: {token[:4]}-{token[4:]}")
    print("---------------------------------")
    print("Waiting for approval from server...")

    while True:
        try:
            payload = {
                "worker_id": worker_config["worker_id"],
                "token": token
            }
            response = requests.post(f"{ORCHESTRATOR_URL}/heartbeat", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "approved" and data.get("api_key"):
                    worker_config["api_key"] = data["api_key"]
                    save_config()
                    print("\n✅ This worker has been APPROVED!")
                    print("Received API key. Starting main loop...")
                    return True
                elif data.get("status") == "pending_approval":
                    # This is the expected state, just wait
                    pass
            else:
                print(f"Error from server: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            print("❌ Connection failed. Retrying in 10s... (Is the server running?)")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        time.sleep(10) # Wait 10s before pinging again

# --- Main Training Loop ---

def get_task():
    """Asks the server for a new task."""
    headers = {"X-API-Key": worker_config["api_key"]}
    try:
        response = requests.get(f"{ORCHESTRATOR_URL}/get-task", headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 401:
            print("❌ Unauthorized (401). API key may be invalid. Stopping.")
            return {"status": "error", "message": "unauthorized"}
    except Exception as e:
        print(f"Error getting task: {e}")
    return None

def process_task(task, device):
    """
    This is the core training function.
    It processes the data on the CPU or GPU.
    """
    batch_id = task.get("batch_id")
    print(f"Processing task {batch_id} on {device.type}...")
    
    # --- TODO: Replace this with real model logic ---
    # 1. Download data from task["data_url"]
    # 2. Load data and labels into PyTorch tensors
    # 3. Move tensors to `device`
    # 4. model.zero_grad()
    # 5. outputs = model(data)
    # 6. loss = criterion(outputs, labels)
    # 7. loss.backward()
    # 8. Get gradients: gradients = [param.grad.clone() for param in model.parameters()]
    # 9. Serialize gradients (e.g., convert to lists or save to .npy)
    # --- End of TODO ---

    # Dummy processing:
    time.sleep(random.randint(2, 5)) # Simulate work
    
    # Dummy results:
    results = {
        "batch_id": batch_id,
        "loss": round(random.random(), 4),
        "gradients": "DUMMY_GRADIENTS_DATA" # TODO: Replace with real gradients
    }
    print(f"✅ Finished task {batch_id}. Loss: {results['loss']}")
    return results

def submit_work(results):
    """Sends the completed work (gradients) back to the server."""
    headers = {"X-API-Key": worker_config["api_key"]}
    try:
        response = requests.post(f"{ORCHESTRATOR_URL}/submit-work", headers=headers, json=results, timeout=10)
        if response.status_code == 200:
            return True
    except Exception as e:
        print(f"Error submitting work: {e}")
    return False

# --- Main Execution ---

def main():
    print("--- Starting Presidential AI Worker ---")
    
    # 1. Load config
    load_or_create_config()

    # 2. Get API key if we don't have one
    if not worker_config.get("api_key"):
        register_and_get_api_key()

    # 3. Detect hardware
    device = get_device()
    
    # 4. Start the main work loop
    print("\n--- Worker is online and ready for tasks ---")
    while True:
        task = get_task()
        
        if task and task.get("status") == "no_tasks":
            print("No tasks available. Waiting 30s...")
            time.sleep(30)
            continue
        
        if task and task.get("batch_id"):
            # We have a task!
            results = process_task(task, device)
            submit_work(results)
            # Short pause to prevent spamming the server
            time.sleep(1) 
            continue
        
        if task and task.get("status") == "error":
            # Got a critical error, stop the client
            break
            
        # If task is None or malformed
        print("Could not get task from server. Retrying in 15s...")
        time.sleep(15)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting.")
        os._exit(0)


