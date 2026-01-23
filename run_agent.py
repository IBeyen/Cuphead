import time
import cv2
import numpy as np
import torch
from collections import deque
from stable_baselines3 import PPO
from environment import CupheadEnv
import pydirectinput
pydirectinput.PAUSE = 0.0  # <--- REMOVES THE 0.1s DELAY

TARGET_FPS = 10
FRAME_DURATION = 1.0 / TARGET_FPS

env = CupheadEnv()

model_path = "cuphead_imitation_model_best.zip" 
model = PPO.load(model_path)

print("Agent loaded. Switch to Cuphead window immediately!")
time.sleep(3) # time given to Alt-Tab into the game

frame_stack = deque(maxlen=4)

# Pre-fill the stack with the first frame so we don't crash on start
obs, _ = env.reset()
for _ in range(4):
    frame_stack.append(obs)

print("Running...")
next_frame_time = time.time()

try:
    while True:
        next_frame_time += FRAME_DURATION
        
        # --- LOGIC START ---
        new_frame = env.get_obs()
        frame_stack.append(new_frame)
        
        stacked_obs = np.concatenate(list(frame_stack), axis=-1)
        stacked_obs = np.transpose(stacked_obs, (2, 0, 1))
        
        stacked_obs = stacked_obs.astype(np.float32) / 255.0
        
        action, _states = model.predict(stacked_obs, deterministic=True)
        
        env._apply_action(action)
        
        now = time.time()
        sleep_needed = next_frame_time - now
        
        if sleep_needed > 0:
            # If we have lots of time (>2ms), sleep safely
            if sleep_needed > 0.002:
                time.sleep(sleep_needed - 0.002)
                
            # 'Spin lock' for the final tiny remaining time to be exact
            # (Burn CPU for the last millisecond to ensure we don't oversleep)
            while time.time() < next_frame_time:
                pass 
        else:
            # We are lagging behind! Resync the timer to prevent a "death spiral"
            # If we are excessively late (>0.5s), simply reset the clock
            if now - next_frame_time > 0.5:
                next_frame_time = now
            print(f"Lagging! FPS: {1.0 / (FRAME_DURATION + abs(sleep_needed)):.1f}", end='\r')

        # Debug FPS (based on actual loop time)
        # print(f"FPS: {1.0 / FRAME_DURATION:.1f}", end='\r')

except KeyboardInterrupt:
    print("Stopped.")
    cv2.destroyAllWindows()