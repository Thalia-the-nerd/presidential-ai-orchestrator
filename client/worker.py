# client/worker.py

import redis
import time
import os

# --- Configuration ---
# Use the same environment variables as the orchestrator for consistency.
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# --- Main Worker Logic ---

def connect_to_redis(host, port):
    """
    Attempts to connect to Redis with a retry mechanism.
    Identical to the function in the orchestrator.
    """
    while True:
        try:
            print(f"Worker attempting to connect to Redis at {host}:{port}...")
            r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
            r.ping()
            print("Worker successfully connected to Redis.")
            return r
        except redis.exceptions.ConnectionError as e:
            print(f"Worker connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def main():
    """
    The main function for the client worker.
    """
    print("--- Starting Client Worker ---")
    
    # 1. Establish the connection to the Redis backbone.
    redis_client = connect_to_redis(REDIS_HOST, REDIS_PORT)
    
    # 2. (Placeholder) Main loop.
    #    In the future, this loop will fetch tasks and do training.
    #    For now, it just listens for the orchestrator's heartbeat.
    try:
        while True:
            # Check for the heartbeat key set by the orchestrator.
            # The .get() method returns the value if the key exists,
            # and None if it does not (e.g., if it expired or orchestrator is down).
            orchestrator_status = redis_client.get("orchestrator_status")
            
            if orchestrator_status == "alive":
                print("Orchestrator is alive. Waiting for tasks...")
            else:
                print("Orchestrator seems to be down. Standing by.")
            
            # This is where the core logic will go.
            # - Pop a task from 'task_queue'
            # - Perform training
            # - Push gradients to 'results_queue'
            
            time.sleep(3) # Check the status every 3 seconds.

    except KeyboardInterrupt:
        print("\nWorker is shutting down.")
    finally:
        print("--- Worker Stopped ---")

if __name__ == "__main__":
    main()
