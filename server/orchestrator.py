# server/orchestrator.py

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
import json
from sklearn.model_selection import train_test_split

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
# --- New Checkpointing Configuration ---
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "training_checkpoint.pth")


m.patch()

# --- PyTorch Model & Helpers (Identical, no changes) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__();self.conv1 = nn.Conv2d(1, 4, 3, 1, 1);self.pool = nn.MaxPool2d(2, 2, 0);self.conv2 = nn.Conv2d(4, 8, 3, 1, 1);self.fc1 = nn.Linear(8 * 50 * 50, 128);self.fc2 = nn.Linear(128, 1);self.relu = nn.ReLU();self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x=x.unsqueeze(1);x=self.pool(self.relu(self.conv1(x)));x=self.pool(self.relu(self.conv2(x)));x=x.view(-1, 8 * 50 * 50);x=self.relu(self.fc1(x));return self.sigmoid(self.fc2(x))
def connect_to_redis(host, port):
    while True:
        try: r = redis.Redis(host=host, port=port); r.ping(); print("Successfully connected to Redis."); return r
        except redis.exceptions.ConnectionError: print("Connection failed. Retrying..."); time.sleep(5)
def start_new_epoch(r: redis.Redis, epoch_num, num_samples):
    print(f"\n--- Starting Epoch #{epoch_num} ---"); indices = list(range(num_samples)); np.random.shuffle(indices); pipe = r.pipeline(); task_count = 0
    for i in range(0, num_samples, BATCH_SIZE): task = ",".join(map(str, indices[i:i+BATCH_SIZE])); pipe.lpush("task_queue", task); task_count += 1
    pipe.execute(); r.set("epoch_counter", epoch_num); print(f"Pushed {task_count} tasks for Epoch #{epoch_num}.")
def run_validation(model, X_val, y_val, device):
    print("Running validation..."); model.eval(); criterion = nn.BCELoss(); val_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            batch_X = torch.tensor(X_val[i:i+BATCH_SIZE], dtype=torch.float32).to(device); batch_y = torch.tensor(y_val[i:i+BATCH_SIZE], dtype=torch.float32).unsqueeze(1).to(device)
            outputs = model(batch_X); loss = criterion(outputs, batch_y); val_loss += loss.item(); predicted = (outputs > 0.5).float(); total += batch_y.size(0); correct += (predicted == batch_y).sum().item()
    avg_loss = val_loss / (len(X_val) / BATCH_SIZE); accuracy = 100 * correct / total
    print(f"Validation Results: Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%"); model.train()
    return {"loss": avg_loss, "accuracy": accuracy}

# --- Main Orchestrator Logic ---
def main():
    print("--- Starting Orchestrator (V6.0 - With Checkpointing) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Orchestrator using device: {device.upper()}")
    redis_client = connect_to_redis(REDIS_HOST, REDIS_PORT)

    # Initialize Model and Optimizer
    global_model = SimpleCNN().to(device)
    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)
    epoch_counter = 1

    # --- Load Checkpoint Logic ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found existing checkpoint at '{CHECKPOINT_PATH}'. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        global_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_counter = checkpoint['epoch'] + 1
        print(f"Successfully loaded checkpoint. Resuming training from Epoch #{epoch_counter}.")
    else:
        print("No checkpoint found. Starting a new training session.")
        # On a fresh start, clear old Redis data
        print("Clearing old data for a clean start..."); redis_client.delete("task_queue", "results_queue", "global_model_weights", "X_train", "y_train", "orchestrator_status", "training_status", "model_update_counter", "epoch_counter", "last_validation_results", "validation_history")
        
        # Load and push data only on a fresh start
        X_full = np.load(os.path.join(DATASET_PATH, 'X_train.npy')); y_full = np.load(os.path.join(DATASET_PATH, 'y_train.npy'))
        X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_full)
        print(f"Dataset split: {X_train.shape[0]} training, {X_val.shape[0]} validation.")
        print("Pushing datasets to Redis..."); redis_client.set("X_train", msgpack.packb(X_train, default=m.encode)); redis_client.set("y_train", msgpack.packb(y_train, default=m.encode)); 
        # Store validation set in memory
        X_val_global, y_val_global = X_val, y_val
        num_train_samples = X_train.shape[0]
        del X_train, y_train, X_val, y_val, X_full, y_full
        
        # Set initial state
        redis_client.set("training_status", "RUNNING"); redis_client.set("model_update_counter", 0)
        start_new_epoch(redis_client, epoch_counter, num_train_samples)
    
    # Push the current model (either loaded or new) to Redis for workers
    buffer = io.BytesIO(); torch.save(global_model.state_dict(), buffer); redis_client.set("global_model_weights", buffer.getvalue())

    print("\n--- Entering Main Training Loop ---")
    try:
        while True:
            # (Main loop logic is identical to before)
            status = (redis_client.get("training_status") or b'RUNNING').decode('utf-8')
            if status == "STOP": print("STOP command received. Shutting down."); break
            if status == "PAUSED": redis_client.set("orchestrator_status", "paused_by_cli"); time.sleep(2); continue
            redis_client.set("orchestrator_status", "running")
            serialized_gradients = redis_client.rpop("results_queue")
            if serialized_gradients:
                gradients = msgpack.unpackb(serialized_gradients, object_hook=m.decode)
                optimizer.zero_grad()
                for param, grad_array in zip(global_model.parameters(), gradients): param.grad = torch.tensor(grad_array).to(device)
                optimizer.step()
                buffer = io.BytesIO(); torch.save(global_model.state_dict(), buffer); redis_client.set("global_model_weights", buffer.getvalue()); redis_client.incr("model_update_counter")
            
            if redis_client.llen("task_queue") == 0 and redis_client.llen("results_queue") == 0:
                print(f"\n--- Epoch #{epoch_counter} Complete ---")
                
                # We need to reload the validation data if we loaded from a checkpoint
                if 'X_val_global' not in locals():
                    print("Loading validation data for this session...")
                    X_full = np.load(os.path.join(DATASET_PATH, 'X_train.npy')); y_full = np.load(os.path.join(DATASET_PATH, 'y_train.npy'))
                    _, X_val_global, _, y_val_global = train_test_split(X_full, y_full, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_full)
                    num_train_samples = X_full.shape[0] - X_val_global.shape[0]
                    del X_full, y_full

                val_results = run_validation(global_model, X_val_global, y_val_global, device); val_results['epoch'] = epoch_counter
                redis_client.set("last_validation_results", json.dumps(val_results)); redis_client.rpush("validation_history", json.dumps(val_results))
                
                # --- Save Checkpoint Logic ---
                print(f"Saving checkpoint to '{CHECKPOINT_PATH}'...")
                torch.save({
                    'epoch': epoch_counter,
                    'model_state_dict': global_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, CHECKPOINT_PATH)
                print("Checkpoint saved successfully.")

                epoch_counter += 1; start_new_epoch(redis_client, epoch_counter, num_train_samples)
            time.sleep(0.05)
    except KeyboardInterrupt: print("\nOrchestrator shutting down gracefully.")
    finally: redis_client.set("orchestrator_status", "stopped"); print("--- Orchestrator Stopped ---")

if __name__ == "__main__":
    main()
