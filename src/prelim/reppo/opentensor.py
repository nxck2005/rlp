import os
import subprocess
import webbrowser
import time

# 1. Define where your logs are stored
# This should point to the parent 'logs' folder to see Baseline and Curriculum together
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

def launch():
    print(f"Checking for logs in: {LOG_DIR}")
    
    # 2. Check if the directory actually exists
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory {LOG_DIR} not found. Have you started training yet?")
        return

    # 3. Command to run TensorBoard via uv
    # We use --bind_all in case you are accessing this from a remote machine/WSL2
    cmd = ["uv", "run", "tensorboard", "--logdir", LOG_DIR, "--bind_all"]

    print("Launching TensorBoard...")
    
    # 4. Start the process in the background
    process = subprocess.Popen(cmd)

    # 5. Wait a few seconds for the server to initialize, then open the browser
    time.sleep(5)
    url = "http://localhost:6006"
    print(f"Opening browser to {url}")
    webbrowser.open(url)

    try:
        # Keep the script alive while TensorBoard runs
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down TensorBoard...")
        process.terminate()

if __name__ == "__main__":
    launch()