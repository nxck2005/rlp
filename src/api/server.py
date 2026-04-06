from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import os
from api.watcher import HeadlessWatcher

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Starting FastAPI server...")

# Model Registry (mapping model IDs to file paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_CONFIG = {
    "ppo": {
        "path": os.path.join(MODELS_DIR, "keyppo_flat/PPO_Flat_DoorKey5x5"),
        "env": "MiniGrid-DoorKey-5x5-v0",
        "type": "PPO"
    },
    "dqn": {
        "path": os.path.join(MODELS_DIR, "dqn_final"),
        "env": "MiniGrid-DoorKey-5x5-v0",
        "type": "DQN"
    },
    "rppo_baseline": {
        "path": os.path.join(MODELS_DIR, "rppo_baseline/rppo_baseline_final"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "RPPO"
    },
    "rppo_curriculum": {
        "path": os.path.join(MODELS_DIR, "rppo_cur/rppo_cur_3_target"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "RPPO"
    }
}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: str):
    logger.info(f"WebSocket connection request for model: {model_id}")
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for model: {model_id}")
    
    if model_id not in MODEL_CONFIG:
        logger.error(f"Invalid model ID: {model_id}")
        await websocket.send_text(json.dumps({"error": "Invalid model ID"}))
        await websocket.close()
        return

    config = MODEL_CONFIG[model_id]
    
    # Check if model exists
    full_path = config["path"]
    if not os.path.exists(full_path + ".zip"):
         logger.error(f"Model file not found: {full_path}.zip")
         await websocket.send_text(json.dumps({"error": f"Model file not found: {full_path}.zip"}))
         await websocket.close()
         return

    watcher = None
    try:
        logger.info(f"Initializing watcher for {model_id}...")
        watcher = HeadlessWatcher(
            env_id=config["env"],
            model_path=full_path,
            model_type=config["type"]
        )
        logger.info(f"Watcher initialized for {model_id}. Starting stream...")
        
        while True:
            # Step the agent
            frame, agent_view, action, reward, done, reset, final_reward, steps = watcher.step()
            
            # Prepare data to send
            payload = {
                "frame": watcher.get_frame_base64(frame),
                "agent_view": agent_view.tolist(), # 7x7x3
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "reset": bool(reset),
                "final_reward": float(final_reward) if reset else None,
                "steps": int(steps) if reset else None
            }
            
            await websocket.send_text(json.dumps(payload))
            
            # Control simulation speed (~5 FPS)
            await asyncio.sleep(0.2)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for model: {model_id}")
    except Exception as e:
        print(f"Error in watcher: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        if watcher:
            watcher.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
