import requests
import time
import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import io
import base64
import random

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
device_info = "CPU" # NEW: More descriptive device info

# --- NEW: PyTorch Model Definition ---
# This is a simple 2D-CNN designed for 2D matrices (like your FCMs)
# It assumes your input matrices are 1 channel (like a grayscale image)
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Formula: (Width - Kernel + 2*Padding) / Stride + 1
        # Assumes 50x50 input, but will adapt
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # (50x50) -> (25x25)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (25x25) -> (12x12)

        # We need to calculate the flattened size
        # We'll do a dummy forward pass to get it
        self._to_linear = None
        self._get_conv_output((input_channels, 50, 50)) # Assume 50x50, will re-check
        
        self.fc1 = nn.Linear(self._to_linear, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(input))))))
            self._to_linear = int(np.prod(output.shape[1:])) # Get flat size

    def forward(self, x):
        # Re-check input size if needed
        if self._to_linear is None:
            # Get shape (C, H, W) from (N, C, H, W)
            self._get_conv_output(x.shape[1:])
            self.fc1 = nn.Linear(self._to_linear, 64).to(x.device)
            self.fc2 = nn.Linear(64, 2).to(x.device)

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1) 
        
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) # No softmax here, CrossEntropyLoss will do it
        return x

# --- NEW: Model Serialization ---

def serialize_model(model):
    """Saves model state_dict to a base64-encoded string."""
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode('utf-8')
        return b64_data
    except Exception as e:
        print(f"Error serializing model: {e}")
        return None

def deserialize_model(model, b64_data):
    """Loads model state_dict from a base64-encoded string."""
    if b64_data is None:
        print("No model parameters received, using new model.")
        return model # Return the initialized model
    try:
        buffer = io.BytesIO(base64.b64decode(b64_data))
        model.load_state_dict(torch.load(buffer))
        print("Successfully loaded model parameters from server.")
        return model
    except Exception as e:
        print(f"Error deserializing model: {e}. Using new model.")
        return model # Return the initialized model on error


# --- Worker Identity ---
def load_worker_identity():
    """Loads worker ID and API key from the config file."""
    # ... (same as before) ...
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
    # ... (same as before) ...
    global api_key
    api_key = new_api_key
    try:
        with open(WORKER_CONFIG_FILE, 'w') as f:
            json.dump({"worker_id": worker_id, "api_key": api_key}, f, indent=4)
    except Exception as e:
        print(f"CRITICAL: Could not save worker config to {WORKER_CONFIG_FILE}: {e}")

# --- MODIFIED: detect_device ---
def detect_device():
    """Detects if a GPU is available and sets the global device."""
    global device, device_info
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_info = f"GPU: {torch.cuda.get_device_name(0)}"
            print(f"✅ {device_info}")
        else:
            device = torch.device("cpu")
            device_info = "CPU"
            print("⚠️ GPU not found. Using CPU.")
    except Exception as e:
        print(f"Error during PyTorch device detection: {e}")
        print("Defaulting to CPU.")
        device = torch.device("cpu")
        device_info = "CPU (PyTorch Error)"

# --- MODIFIED: register_and_approve ---
def register_and_approve():
    """Handles the initial registration and approval loop."""
    global api_key
    print("--- NEW WORKER REGISTRATION ---")
    temp_token = str(random.randint(10000000, 99999999))
    print("Please approve this PC in your management panel using the token:")
    print(f"TOKEN: {temp_token[:4]}-{temp_token[4:]}")
    print("---------------------------------")
    
    while True:
        try:
            payload = {
                "worker_id": worker_id,
                "token": temp_token,
                "device_info": device_info # NEW: Send device info on first register
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
    # ... (same as before) ...
    try:
        headers = {"X-API-Key": api_key}
        payload = {"worker_id": worker_id, "device_info": device_info}
        response = requests.post(f"{API_URL}/heartbeat", json=payload, headers=headers, timeout=5)
        
        if response.status_code == 401:
             print("API key is invalid or was revoked by server. Re-registering...")
             save_worker_identity(None) # Clear the bad API key
             register_and_approve()
        
    except Exception as e:
        print(f"Heartbeat failed: {e}")

# --- MODIFIED: process_task ---
def process_task(task):
    """
    This is the REAL AI work.
    """
    batch_id = task.get("batch_id")
    print(f"⚙️ Processing task: {batch_id} (Version {task.get('model_version')})...")
    
    try:
        # --- 1. Get Data ---
        # Server sends data as lists, convert back to numpy arrays
        X_list = task.get("data_X")
        y_list = task.get("data_y")
        
        X_np = np.array(X_list, dtype=np.float32)
        y_np = np.array(y_list, dtype=np.int64) # CrossEntropyLoss expects int64
        
        # Add a "channel" dimension for the 2D-CNN (N, H, W) -> (N, C, H, W)
        # Our CNN expects (N, 1, H, W)
        X_np = np.expand_dims(X_np, axis=1) 
        
        # Convert to PyTorch Tensors
        X_tensor = torch.from_numpy(X_np).to(device)
        y_tensor = torch.from_numpy(y_np).to(device)

        # --- 2. Get Model ---
        model = SimpleCNN().to(device)
        b64_params = task.get("model_params_base64")
        model = deserialize_model(model, b64_params)
        
        # --- 3. Get Training Hyperparameters ---
        lr = task.get("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # --- 4. Run Training Step ---
        model.train() # Set model to training mode
        
        optimizer.zero_grad()  # Clear old gradients
        
        outputs = model(X_tensor) # Forward pass
        
        loss = criterion(outputs, y_tensor) # Calculate loss
        
        loss.backward() # Backward pass (calculate gradients)
        
        optimizer.step() # Update model weights
        
        # --- 5. Serialize New Model ---
        # In this strategy, we send the *full new model* back, not just gradients.
        # This is simpler to implement than federated averaging.
        new_b64_params = serialize_model(model)
        
        loss_value = loss.item()
        print(f"✅ Task {batch_id} complete. Loss: {loss_value:.4f}")
        
        return {
            "batch_id": batch_id,
            "loss": loss_value,
            "model_params_base64": new_b64_params, # Send the new model back
            # "gradients_base64": ... (Alternative: send gradients)
        }

    except Exception as e:
        print(f"Error processing task {batch_id}: {e}")
        # Send back a failure
        return {
            "batch_id": batch_id,
            "loss": None,
            "error": str(e)
        }

def main_loop():
    """The main work loop for the client."""
    # ... (same as before, but with logic to skip failed tasks) ...
    print("--- Worker is online and ready for tasks ---")
    while True:
        try:
            headers = {"X-API-Key": api_key}
            response = requests.get(f"{API_URL}/get-task", headers=headers, timeout=10)
            
            if response.status_code == 401:
                print("API key is invalid. Re-registering...")
                save_worker_identity(None)
                register_and_approve()
                continue
                
            if response.status_code == 200:
                task = response.json()
                
                if task.get("status") == "no_tasks":
                    print(f"{task.get('message', 'No tasks available.')} Waiting 30s...")
                    time.sleep(30)
                    continue
                
                # --- We have a task! ---
                results = process_task(task)
                
                # If processing failed, results will have an "error" key
                if results.get("error"):
                    print(f"Failed task {results.get('batch_id')}, reporting error to server.")
                    # We still submit the error so the server can log it
                
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
    # Detect hardware *first*
    detect_device()

    if not load_worker_identity():
        # This is a new worker, needs approval
        register_and_approve()
    
    if not api_key:
        print("Registration failed. Exiting.")
        exit()
    
    # Start the main loop
    main_loop()


