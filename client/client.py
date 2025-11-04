import requests
import time
import os
import json
import uuid
import torch # We import torch to handle device detection

# --- Configuration ---
# This URL should point to your home server's Nginx proxy
API_URL = "https://api.presidentialAIchallenge.thaliathenerd.dev"
# API_URL = "http://192.168.1.100:10041" # <-- Use this for local testing if needed

WORKER_CONFIG_FILE = "worker.json"
HEARTBEAT_INTERVAL = 10 # Seconds

# --- Worker Identity & State ---
worker_id = None
api_key = None
device = None # This will be 'cuda' or 'cpu'

def load_worker_identity():
    """Loads worker ID and API key from the config file."""
    global worker_id, api_key
    if os.path.exists(WORKER_CONFIG_FILE):
        try:
            with open(WORKER_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                worker_id = config.get("worker_id")
                api_key = config.get("api_key")
                if worker_id and api_key:
                    print(f"✅ Worker identity loaded. ID: {worker_id}")
                    return True
        except Exception as e:
            print(f"Error loading {WORKER_CONFIG_FILE}: {e}. Creating new identity.")
            
    # If file doesn't exist or is corrupt, create a new identity
    worker_id = str(uuid.uuid4())
    api_key = None
    save_worker_identity(None) # Save the new ID
    print(f"👋 New worker identity created. ID: {worker_id}")
    return False

def save_worker_identity(new_api_key):
    """Saves worker ID and new API key to the config file."""
    global api_key
    api_key = new_api_key
    try:
        with open(WORKER_CONFIG_FILE, 'w') as f:
            json.dump({"worker_id": worker_id, "api_key": api_key}, f, indent=4)
    except Exception as e:
        print(f"CRITICAL: Could not save worker config to {WORKER_CONFIG_FILE}: {e}")

def detect_device():
    """Detects if a GPU is available and sets the global device."""
    global device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ GPU (CUDA) is available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("⚠️ GPU not found. Using CPU.")
    except Exception as e:
        print(f"Error during PyTorch device detection: {e}")
        print("Defaulting to CPU.")
        device = torch.device("cpu")

def register_and_approve():
    """Handles the initial registration and approval loop."""
    global api_key
    print("--- NEW WORKER REGISTRATION ---")
    # Generate a temporary token for this registration attempt
    # The server doesn't store this; it's just for the user to see
    temp_token = str(random.randint(10000000, 99999999))
    print("Please approve this PC in your management panel using the token:")
    print(f"TOKEN: {temp_token[:4]}-{temp_token[4:]}")
    print("---------------------------------")
    
    while True:
        try:
            payload = {
                "worker_id": worker_id,
                "token": temp_token
            }
            response = requests.post(f"{API_URL}/heartbeat", json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "approved":
                    new_api_key = data.get("api_key")
                    if new_api_key:
                        print("✅ This worker has been APPROVED!")
                        save_worker_identity(new_api_key)
                        return True
                elif data.get("status") == "pending_approval":
                    print("Waiting for approval from server...")
                else:
                    print(f"Unknown server response: {data}")

            elif response.status_code == 409:
                print("Token collision. Generating new token...")
                temp_token = str(random.randint(10000000, 99999999))
                print(f"NEW TOKEN: {temp_token[:4]}-{temp_token[4:]}")
            else:
                print(f"Error connecting to server ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            print("Server offline. Retrying...")
        except Exception as e:
            print(f"An error occurred: {e}")
            
        time.sleep(HEARTBEAT_INTERVAL)

def send_heartbeat():
    """Sends a simple heartbeat to check in with the server."""
    try:
        headers = {"X-API-Key": api_key}
        payload = {"worker_id": worker_id}
        response = requests.post(f"{API_URL}/heartbeat", json=payload, headers=headers, timeout=5)
        
        if response.status_code == 401:
             print("API key is invalid or was revoked by server. Re-registering...")
             save_worker_identity(None) # Clear the bad API key
             register_and_approve()
        
    except Exception as e:
        print(f"Heartbeat failed: {e}")

def process_task(task):
    """
    --- STUB FUNCTION ---
    This is where the real AI work will happen.
    """
    batch_id = task.get("batch_id")
    print(f"⚙️ Processing task: {batch_id} (Version {task.get('model_version')})...")
    
    # --- TODO: REAL AI LOGIC ---
    # 1. Deserialize `task.get("model_params")`
    # 2. Create your 2D-CNN model and load the params
    # 3. Move model and data (from task) to `device` (GPU or CPU)
    # 4. Run the training step (forward pass, loss calculation, backward pass)
    # 5. Extract the gradients and the loss value
    # ---
    
    # For now, we'll just simulate the work
    time.sleep(random.uniform(2.0, 5.0))
    
    # Dummy results
    dummy_loss = round(random.uniform(0.1, 1.5), 4)
    dummy_gradients = "---DUMMY_GRADIENTS---" # This would be a serialized tensor
    
    print(f"✅ Task {batch_id} complete. Loss: {dummy_loss}")
    
    return {
        "batch_id": batch_id,
        "loss": dummy_loss,
        "gradients": dummy_gradients
    }

def main_loop():
    """The main work loop for the client."""
    print("--- Worker is online and ready for tasks ---")
    while True:
        try:
            headers = {"X-API-Key": api_key}
            response = requests.get(f"{API_URL}/get-task", headers=headers, timeout=10)
            
            if response.status_code == 401:
                print("API key is invalid. Re-registering...")
                save_worker_identity(None)
                register_and_approve()
                continue # Restart the loop
                
            if response.status_code == 200:
                task = response.json()
                
                if task.get("status") == "no_tasks":
                    # Server is idle or training is complete
                    print(f"{task.get('message', 'No tasks available.')} Waiting 30s...")
                    time.sleep(30)
                    continue
                
                # --- We have a task! ---
                results = process_task(task)
                
                # Send the results back
                submit_response = requests.post(f"{API_URL}/submit-work", json=results, headers=headers, timeout=10)
                
                if not submit_response.status_code == 200:
                    print(f"Error submitting work: {submit_response.text}")
                    
            else:
                print(f"Error getting task ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            print("Server connection lost. Retrying...")
            time.sleep(HEARTBEAT_INTERVAL)
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            time.sleep(HEARTBEAT_INTERVAL)

if __name__ == "__main__":
    if not load_worker_identity():
        # This is a new worker, needs approval
        register_and_approve()
    
    if not api_key:
        # This should not happen, but as a fallback
        print("Registration failed. Exiting.")
        exit()

    # Detect hardware
    detect_device()
    
    # Start the main loop
    main_loop()


