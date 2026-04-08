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
    # 1. DQN APPROACHES (Pixel-based / Mixed)
    "dqn_baseline": {
        "path": os.path.join(MODELS_DIR, "dqn_final"),
        "env": "MiniGrid-DoorKey-5x5-v0",
        "type": "DQN",
        "obs_type": "Flat"
    },
    "dqn_framestack": {
        "path": os.path.join(MODELS_DIR, "keydqn_framestack/DQN_Pixels_DoorKey5x5_FrameStack"),
        "env": "MiniGrid-DoorKey-5x5-v0",
        "type": "DQN",
        "obs_type": "Pixel",
        "use_pixel_wrapper": True,
        "n_stack": 4
    },
    "dqn_custom_cnn": {
        "path": os.path.join(BASE_DIR, "prelim/qlearning/models/dqn_baseline/dqn_8x8_model"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "DQN",
        "obs_type": "Pixel"
    },
    "dqn_curriculum": {
        "path": os.path.join(MODELS_DIR, "keydqn_cur/S2_final"),
        "env": "MiniGrid-DoorKey-5x5-v0",
        "type": "DQN",
        "obs_type": "Pixel",
        "use_pixel_wrapper": True
    },
    
    # 2. PPO VISUAL (Pixel-based)
    "ppo_baseline": {
        "path": os.path.join(MODELS_DIR, "baseline/baseline_model"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "PPO",
        "obs_type": "Pixel"
    },
    "ppo_curriculum": {
        "path": os.path.join(MODELS_DIR, "curriculum/S3_weights"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "PPO",
        "obs_type": "Pixel"
    },

    # 3. PPO SYMBOLIC (Flat/Symbolic-based)
    "ppo_flat": {
        "path": os.path.join(MODELS_DIR, "keyppo_flat/PPO_Flat_DoorKey5x5"),
        "env": "MiniGrid-DoorKey-5x5-v0",
        "type": "PPO",
        "obs_type": "Flat"
    },
    "ppo_flat_cur": {
        "path": os.path.join(BASE_DIR, "Phase2_Model"),
        "env": "MiniGrid-DoorKey-5x5-v0",
        "type": "PPO",
        "obs_type": "Flat"
    },

    # 4. RPPO (RECURRENT - Flat)
    "rppo_baseline": {
        "path": os.path.join(MODELS_DIR, "rppo_baseline/rppo_baseline_final"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "RPPO",
        "obs_type": "Flat"
    },
    "rppo_curriculum": {
        "path": os.path.join(MODELS_DIR, "rppo_cur/rppo_cur_3_target"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "RPPO",
        "obs_type": "Flat"
    },

    # 5. REPP2 (ADVANCED - Flat)
    "repp2_4stage": {
        "path": os.path.join(BASE_DIR, "prelim/repp2/models/fast_3_target"),
        "env": "MiniGrid-DoorKey-8x8-v0",
        "type": "RPPO",
        "obs_type": "Flat"
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
            model_type=config["type"],
            obs_type=config["obs_type"],
            use_pixel_wrapper=config.get("use_pixel_wrapper", False),
            n_stack=config.get("n_stack")
        )
        logger.info(f"Watcher initialized for {model_id}. Starting stream...")
        
        while True:
            # Step the agent
            frame, agent_view, full_grid, agent_pos, agent_dir, action, reward, done, reset, final_reward, steps = watcher.step()
            
            # Prepare data to send
            payload = {
                "frame": watcher.get_frame_base64(frame),
                "agent_view": agent_view.tolist(), # 7x7x3
                "full_grid": full_grid.tolist(), # WxHx3
                "agent_pos": [int(agent_pos[0]), int(agent_pos[1])],
                "agent_dir": int(agent_dir),
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
