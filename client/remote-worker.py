# client/worker_remote.py

import redis
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import io
import msgpack
import msgpack_numpy as m
import psutil
import platform # To get hostname

# --- Configuration ---
# This worker is configured using Environment Variables for portability.
# Before running, you MUST set the REDIS_HOST variable.
#
# On Linux/macOS:
#   export REDIS_HOST=146.168.99.93
#   python3 worker_remote.py
#
# On Windows (Command Prompt):
#   set REDIS_HOST=146.168.99.93
#   python worker_remote.py
#
# On Windows (PowerShell):
#   $env:REDIS_HOST="146.168.99.93"
#   python worker_remote.py

# Fetch the server IP from an environment variable. Exit if it's not set.
SERVER_IP = os.getenv("REDIS_HOST")
if not SERVER_IP:
    print("FATAL ERROR: The REDIS_HOST environment variable is not set.")
    print("Please set it to your server's IP address before running.")
    exit(1)

REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CPU_THRESHOLD = 70.0

m.patch()

# --- PyTorch Model & Helpers (Identical to previous worker) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__();self.conv1=nn.Conv2d(1,4,3,1,1);self.pool=nn.MaxPool2d(2,2,0);self.conv2=nn.Conv2d(4,8,3,1,1);self.fc1=nn.Linear(8*50*50,128);self.fc2=nn.Linear(128,1);self.relu=nn.ReLU();self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=x.unsqueeze(1);x=self.pool(self.relu(self.conv1(x)));x=self.pool(self.relu(self.conv2(x)));x=x.view(-1,8*50*50);x=self.relu(self.fc1(x));return self.sigmoid(self.fc2(x))

def connect_to_redis(host, port):
    while True:
        try:
            print(f"Worker attempting to connect to remote server at {host}:{port}...")
            r = redis.Redis(host=host, port=port, db=0); r.ping()
            print(f"SUCCESS! Connected to remote Redis server.")
            return r
        except redis.exceptions.ConnectionError as e:
            print(f"Worker connection failed: {e}. Retrying in 10 seconds...")
            time.sleep(10)

# --- Main Worker Logic ---
def main():
    print("--- Starting Remote Client Worker ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    worker_id = f"worker_{os.getpid()}@{platform.node()}"
    print(f"Starting worker '{worker_id}' on device: {device.upper()}")

    redis_client = connect_to_redis(SERVER_IP, REDIS_PORT)
    
    try:
        print("Downloading training dataset from remote server...")
        serialized_X = redis_client.get("X_train"); serialized_y = redis_client.get("y_train")
        if not serialized_X or not serialized_y: raise RuntimeError("Could not fetch dataset from Redis.")
        X_train = msgpack.unpackb(serialized_X, object_hook=m.decode); y_train = msgpack.unpackb(serialized_y, object_hook=m.decode)
        print(f"Training set received. Shape: {X_train.shape}")
    except RuntimeError as e:
        print(f"FATAL: {e}"); return

    local_model = SimpleCNN().to(device); optimizer = optim.Adam(local_model.parameters(), lr=0.001); criterion = nn.BCELoss()
    
    print("\nWorker is ready. Entering training loop.")
    try:
        while True:
            status = (redis_client.get("training_status") or b'RUNNING').decode('utf-8')
            if status == "STOP": print("STOP command received. Exiting."); break
            if status == "PAUSED": print("Training is PAUSED. Standing by..."); redis_client.zadd("worker_heartbeats", {worker_id: time.time()}); time.sleep(5); continue

            redis_client.zadd("worker_heartbeats", {worker_id: time.time()})

            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > CPU_THRESHOLD: print(f"CPU usage high ({cpu_usage}%). Standing by..."); time.sleep(10); continue

            task = redis_client.brpop("task_queue", timeout=5)
            if not task: continue
            
            _, task_data = task
            
            serialized_weights = redis_client.get("global_model_weights")
            if serialized_weights: local_model.load_state_dict(torch.load(io.BytesIO(serialized_weights)))

            batch_indices = [int(i) for i in task_data.decode('utf-8').split(',')]
            batch_X = torch.tensor(X_train[batch_indices], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_train[batch_indices], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad(); outputs = local_model(batch_X); loss = criterion(outputs, batch_y); loss.backward()
            gradients = [p.grad.cpu().numpy() for p in local_model.parameters() if p.grad is not None]
            redis_client.lpush("results_queue", msgpack.packb(gradients, default=m.encode))
            
    except KeyboardInterrupt: print("\nWorker shutting down.")
    except Exception as e: print(f"\nAn error occurred: {e}")
    finally: redis_client.zrem("worker_heartbeats", worker_id); print("--- Worker Stopped ---")

if __name__ == "__main__":
    main()
