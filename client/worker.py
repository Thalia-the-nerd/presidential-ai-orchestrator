# client/worker.py

import redis
import time
import os  # <-- This import was missing from the previous main() scope
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import io
import msgpack
import msgpack_numpy as m
import psutil

# --- Configuration (No changes) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CPU_THRESHOLD = 70.0

m.patch()

# --- PyTorch Model Definition (No changes) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(1); x = self.pool(self.relu(self.conv1(x))); x = self.pool(self.relu(self.conv2(x))); x = x.view(-1, 8 * 50 * 50); x = self.relu(self.fc1(x)); return self.sigmoid(self.fc2(x))

# --- Helper Functions (No changes) ---
def connect_to_redis(host, port):
    while True:
        try:
            r = redis.Redis(host=host, port=port, db=0)
            r.ping()
            return r
        except redis.exceptions.ConnectionError:
            print("Worker connection failed. Retrying..."); time.sleep(5)

# --- Main Worker Logic ---
def main():
    # === FIX #1: Corrected print statement ===
    print("--- Starting Client Worker (V2.3 - Corrected) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === FIX #2: Ensured 'os' module is available for this line ===
    worker_id = f"worker_{os.getpid()}@{os.uname().nodename}"
    print(f"Starting worker '{worker_id}' on device: {device.upper()}")

    redis_client = connect_to_redis(REDIS_HOST, REDIS_PORT)
    
    # === FIX #3: Simplified data loading, workers only need the training set ===
    try:
        print("Downloading training dataset...")
        serialized_X = redis_client.get("X_train")
        serialized_y = redis_client.get("y_train")
        if not serialized_X or not serialized_y:
            raise RuntimeError("Could not fetch training dataset from Redis. Is the orchestrator running?")
        X_train = msgpack.unpackb(serialized_X, object_hook=m.decode)
        y_train = msgpack.unpackb(serialized_y, object_hook=m.decode)
        print(f"Training set received. Shape: {X_train.shape}")
    except RuntimeError as e:
        print(f"FATAL: {e}"); return

    local_model = SimpleCNN().to(device)
    optimizer = optim.Adam(local_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print("\nWorker is ready. Entering training loop.")
    try:
        while True:
            status = (redis_client.get("training_status") or b'RUNNING').decode('utf-8')
            if status == "STOP":
                print("STOP command received from server. Exiting."); break
            if status == "PAUSED":
                print("Training is PAUSED by administrator. Standing by..."); time.sleep(5); continue

            current_time = time.time()
            redis_client.zadd("worker_heartbeats", {worker_id: current_time})

            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > CPU_THRESHOLD:
                print(f"CPU usage high ({cpu_usage}%). Standing by..."); time.sleep(10); continue

            task = redis_client.brpop("task_queue", timeout=5)
            if not task:
                continue
            
            _, task_data = task
            
            serialized_weights = redis_client.get("global_model_weights")
            if serialized_weights:
                local_model.load_state_dict(torch.load(io.BytesIO(serialized_weights)))

            batch_indices = [int(i) for i in task_data.decode('utf-8').split(',')]
            
            batch_X = torch.tensor(X_train[batch_indices], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_train[batch_indices], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = local_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            gradients = [p.grad.cpu().numpy() for p in local_model.parameters() if p.grad is not None]
            
            redis_client.lpush("results_queue", msgpack.packb(gradients, default=m.encode))
            
    except KeyboardInterrupt:
        print("\nWorker shutting down.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        redis_client.zrem("worker_heartbeats", worker_id)
        print("--- Worker Stopped ---")

if __name__ == "__main__":
    main()
