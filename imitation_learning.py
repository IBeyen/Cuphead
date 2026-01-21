import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Extractor import PretrainedVisionExtractor
from imitation.algorithms import bc
import numpy as np
import os
import cv2
from environment import CupheadEnv
from torch.utils.data import Dataset, DataLoader

class CupheadDataset(Dataset):
    def __init__(self, root_folder, n_stack=4, img_size=(256, 256)):
        self.samples = []
        self.n_stack = n_stack
        self.img_size = img_size
        
        # Crawl directories to find valid sequences
        for subdir, _, files in os.walk(root_folder):
            frames = sorted([f for f in files if f.endswith(('.png', '.jpg'))])
            if len(frames) < n_stack:
                continue
            
            for i in range(n_stack - 1, len(frames)):
                # Store the paths and the action (parsed from filename)
                stack_paths = [os.path.join(subdir, frames[j]) for j in range(i-(n_stack-1), i+1)]
                # Extract action from the last frame's filename
                # Example: frame_123_z_Key.x.png -> action info is 'z_Key.x'
                action_str = frames[i].split('_')[2].replace('.png', '').replace('.jpg', '')
                self.samples.append((stack_paths, action_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, action_str = self.samples[idx]
        
        # Load and stack frames
        frames = []
        for p in paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
            frames.append(img)
            
        # Concatenate on channel dimension (resulting in 3*4 = 12 channels)
        observation = np.concatenate(frames, axis=-1)
        # Normalize to [0, 1] and transpose to (C, H, W) for PyTorch
        observation = np.transpose(observation, (2, 0, 1)).astype(np.float32) / 255.0
        
        action = self._parse_action(action_str) 
        
        return torch.tensor(observation), torch.tensor(action)

    def _parse_action(self, action_str):
        action_arr =  np.zeros(10, dtype=np.int64)
        mapping = {
            "w": 0,
            "a": 1,
            "s": 2,
            "d": 3,
            "Key.shift": 4, # Dash
            "Key.space": 5, # Jump
            "Key.up": 6, # Special
            "Key.left": 7, # Shoot
            "Key.down": 8, # Switch Weapon
            "Key.right": 9, # Lock
        }
        action_list = action_str.split('_')
        for key in action_list:
            if key in mapping:
                action_arr[mapping[key]] = 1
                
        return action_arr

SAVE_PATH = "cuphead_imitation_model"

env = CupheadEnv()

policy_kwargs = dict(features_extractor_class=PretrainedVisionExtractor)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

dataset = CupheadDataset(root_folder="C:/Users/ilanb/Desktop/Code/Cuphead/training_data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if len(dataset) == 0:
    raise ValueError("No data found! Check your DATA_PATH.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)

epochs = 10
print("Starting Training...")

model.policy.train() # Set to training mode

for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0
    
    for observations, actions in dataloader:
        observations = observations.to(device)
        actions = actions.to(device)
        
        # --- Behavioral Cloning Logic ---
        # "evaluate_actions" returns values, log_prob, and entropy.
        # We want to MAXIMIZE log_prob of the expert actions.
        # Therefore, Loss = -log_prob
        
        # Actions must be valid indices for MultiDiscrete. 
        # Since your record.py might produce 0 or 1, ensure they are passed correctly.
        # SB3 expects actions to be of shape (Batch, 10) for MultiDiscrete([2]*10)
        
        _, log_prob, _ = model.policy.evaluate_actions(observations, actions)
        
        loss = -log_prob.mean()
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_count} | Loss: {loss.item():.4f}", end="\r")
            
    avg_loss = epoch_loss / batch_count
    print(f"\nEpoch {epoch+1}/{epochs} Completed. Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    model.save(f"{SAVE_PATH}_epoch_{epoch+1}")

print("Imitation Learning Complete!")
model.save("cuphead_ppo_pretrained")