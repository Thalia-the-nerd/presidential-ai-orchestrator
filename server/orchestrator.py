# server/orchestrator.py

import redis
import time
import os
import numpy as np
import torch
import torch.nn as nn
import io
import msgpack
import msgpack_numpy as m

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset')
BATCH_SIZE = 32

# Set the numpy resolver for msgpack
m.patch()

# --- PyTorch Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 8 * 50 * 50)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# --- Helper Functions ---
def connect_to_redis(host, port):
    """Attempts to connect to Redis."""
    # Using decode_responses=False for binary data like models and arrays
    while True:
        try:
            print(f"Attempting to connect to Redis at {host}:{port}...")
            r = redis.Redis(host=host, port=port, db=0)
            r.ping()
            print("Successfully connected to Redis.")
            return r
        except redis.exceptions.ConnectionError as e:
            print(f"Connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def load_dataset(path):
    """Loads the dataset from the specified path."""
    print(f"Loading dataset from {path}...")
    try:
        X_train = np.load(os.path.join(path, 'X_train.npy'))
        y_train = np.load(os.path.join(path, 'y_train.npy'))
        print(f"Dataset loaded successfully. Shape of X_train: {X_train.shape}")
        return X_train, y_train
    except FileNotFoundError:
        print(f"ERROR: Dataset files not found in {path}. Please ensure 'X_train.npy' and 'y_train.npy' are present.")
        exit(1)

# --- Main Orchestrator Logic ---
def main():
    """The main function for the orchestrator."""
    print("--- Starting Orchestrator (V3) ---")
    redis_client = connect_to_redis(REDIS_HOST, REDIS_PORT)

    # 1. Clear old data for a clean start
    print("Clearing old queues and data from Redis...")
    redis_client.delete("task_queue", "results_queue", "global_model_weights", "X_train", "y_train")

    # 2. Load the dataset from disk
    X_train, y_train = load_dataset(DATASET_PATH)

    # 3. Serialize and push the entire dataset to Redis for fast access by clients
    print("Serializing and pushing dataset to Redis...")
    # Using MsgPack for efficient NumPy array serialization
    serialized_X = msgpack.packb(X_train, default=m.encode)
    serialized_y = msgpack.packb(y_train, default=m.encode)
    redis_client.set("X_train", serialized_X)
    redis_client.set("y_train", serialized_y)
    print("Dataset successfully stored in Redis.")
    del X_train, y_train # Free up memory

    # 4. Initialize the Global Model and push it to Redis
    global_model = SimpleCNN()
    print("PyTorch 2D-CNN model initialized.")
    
    # Serialize the model's state_dict to a byte buffer
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    buffer.seek(0)
    
    # Store the serialized model in Redis
    redis_client.set("global_model_weights", buffer.getvalue())
    print("Initial model weights have been pushed to Redis.")

    # 5. Create and distribute batch tasks
    num_samples = len(serialized_y) # Get length from the serialized object
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    print(f"Creating tasks for {num_samples} samples with batch size {BATCH_SIZE}...")
    pipe = redis_client.pipeline()
    task_count = 0
    for i in range(0, num_samples, BATCH_SIZE):
        batch_indices = indices[i:i + BATCH_SIZE]
        task = ",".join(map(str, batch_indices))
        pipe.lpush("task_queue", task)
        task_count += 1
    
    pipe.execute()
    print(f"Pushed {task_count} tasks to 'task_queue'.")

    # 6. Main loop for monitoring
    try:
        while True:
            redis_client.set("orchestrator_status", "running", ex=10)
            
            task_queue_len = redis_client.llen("task_queue")
            results_queue_len = redis_client.llen("results_queue")
            print(f"Monitoring... | Tasks to do: {task_queue_len} | Gradients received: {results_queue_len}")
            
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nOrchestrator is shutting down.")
    finally:
        redis_client.delete("orchestrator_status")
        print("--- Orchestrator Stopped ---")

if __name__ == "__main__":
    main()
