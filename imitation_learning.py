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
    def __init__(self, root_folder, n_stack=4, original_fps=20, target_fps=15):
        self.samples = []
        self.n_stack = n_stack
        
        self.key_map = {
            "w": 0,
            "a": 1,
            "s": 2,
            "d": 3,
            "Key.shift": 4, # Dash
            "Key.space": 5, # Jump
            "Key.up": 6,    # Special
            "Key.left": 7,  # Shoot
            "Key.down": 8,  # Switch Weapon
            "Key.right": 9, # Lock
        }
        
        # Calculate step size for downsampling
        ratio = original_fps / target_fps
        
        print(f"Downsampling dataset from {original_fps} FPS to {target_fps} FPS...")
        
        for subdir, _, files in os.walk(root_folder):
            raw_frames = sorted([f for f in files if f.endswith(('.png', '.jpg'))])
            
            if len(raw_frames) < n_stack:
                continue
                
            # Generate indices to keep: 0, 1.33, 2.66 -> 0, 1, 3...
            resampled_indices = [int(i * ratio) for i in range(int(len(raw_frames) / ratio))]
            resampled_indices = [i for i in resampled_indices if i < len(raw_frames)]
            
            filtered_frames = [raw_frames[i] for i in resampled_indices]
            
            for i in range(n_stack - 1, len(filtered_frames)):
                stack_files = filtered_frames[i-(n_stack-1) : i+1]
                stack_paths = [os.path.join(subdir, f) for f in stack_files]
                
                # Parse action from the current frame's filename
                filename = stack_files[-1]
                parts = filename.replace('.png', '').replace('.jpg', '').split('_')
                
                # Everything after index 2 are the keys pressed
                action_keys_list = parts[2:] 
                
                self.samples.append((stack_paths, action_keys_list))
                
        print(f"Dataset ready. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, action_keys_list = self.samples[idx]
        
        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                # Safeguard for corrupt images
                return torch.zeros((12, 256, 256)), torch.zeros(10)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        observation = np.concatenate(frames, axis=-1)
        observation = np.transpose(observation, (2, 0, 1)).astype(np.float32) / 255.0
        
        action = self._parse_action_list(action_keys_list) 
        
        return torch.tensor(observation), torch.tensor(action)

    def _parse_action_list(self, action_keys_list):
        action_arr = np.zeros(10, dtype=np.int64)
        
        for key in action_keys_list:
            if key in self.key_map:
                action_arr[self.key_map[key]] = 1
                
        return action_arr

SAVE_PATH = "cuphead_imitation_model"

env = CupheadEnv()

policy_kwargs = dict(features_extractor_class=PretrainedVisionExtractor)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

dataset = CupheadDataset(root_folder="D:/Cuphead Training data/training_data", original_fps=20, target_fps=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if len(dataset) == 0:
    raise ValueError("No data found! Check your DATA_PATH.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)

epochs = 5
print("Starting Training...")

model.policy.train() # Set to training mode

best_loss = float('inf')
for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0
    
    for observations, actions in dataloader:
        observations = observations.to(device)
        actions = actions.to(device)
        
        # to maximize log_prob of the expert actions loss = -log_prob
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
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save(f"{SAVE_PATH}_best")
        print(f" >>> New best model found! Saved to {SAVE_PATH}_best")

print("Imitation Learning Complete!")
model.save("cuphead_ppo_pretrained")