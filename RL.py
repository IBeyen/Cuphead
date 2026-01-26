import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import CupheadEnv

learning_rate = 5e-5 
n_steps = 2048 # Number of steps to run per update
total_timesteps = 1000000 # How long to train (adjust as needed)

env = CupheadEnv()

print("Loading pre-trained imitation model...")
model = PPO.load("rl_checkpoints\cuphead_ppo_230000_steps.zip", env=env, learning_rate=learning_rate)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./rl_checkpoints/',
    name_prefix='cuphead_ppo'
)

print("Starting Reinforcement Learning...")
print("Switch to Cuphead window immediately!")
time.sleep(3) # Time to alt-tab

try:
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("rl_checkpoints/cuphead_ppo_160000_steps.zip")
    print("Training Complete!")
    
except KeyboardInterrupt:
    print("Training interrupted. Saving current model...")
    model.save("cuphead_rl_interrupted")