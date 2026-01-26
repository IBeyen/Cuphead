import time
import cv2
import numpy as np
import torch
from collections import deque
from stable_baselines3 import PPO
from environment import CupheadEnv
import pydirectinput
pydirectinput.PAUSE = 0.0

TARGET_FPS = 10
FRAME_DURATION = 1.0 / TARGET_FPS

env = CupheadEnv()

model_path = "rl_checkpoints/cuphead_ppo_160000_steps.zip" 
model = PPO.load(model_path)

print("Agent loaded. Switch to Cuphead window immediately!")
time.sleep(3)

frame_stack = deque(maxlen=4)

initial_frame = env.get_obs()
for _ in range(4):
    frame_stack.append(initial_frame)

print("Running...")
next_frame_time = time.time()

try:
    while True:
        next_frame_time += FRAME_DURATION
        
        new_frame = env.get_obs()
        
        frame_stack.append(new_frame)

        stacked_obs = np.concatenate(list(frame_stack), axis=-1)
        
        action, _states = model.predict(stacked_obs, deterministic=True)
        
        env._apply_action(action)
        
        # Timing Logic
        now = time.time()
        sleep_needed = next_frame_time - now
        
        if sleep_needed > 0:
            if sleep_needed > 0.002:
                time.sleep(sleep_needed - 0.002)
            while time.time() < next_frame_time:
                pass 
        else:
            if now - next_frame_time > 0.5:
                next_frame_time = now
            # print(f"Lagging! FPS: {1.0 / (FRAME_DURATION + abs(sleep_needed)):.1f}", end='\r')

except KeyboardInterrupt:
    print("Stopped.")
    cv2.destroyAllWindows()