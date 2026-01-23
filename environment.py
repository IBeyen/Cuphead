import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import dxcam
import pymem
from pymem import Pymem
from get import get_stats
import time
from collections import deque
import pydirectinput
pydirectinput.PAUSE = 0.0  # <--- REMOVES THE 0.1s DELAY

class CupheadEnv(gym.Env):
    def __init__(self):
        super(CupheadEnv, self).__init__()
        
        self.action_space = spaces.MultiDiscrete([2]*10)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(256, 256, 3*4), dtype=np.uint8
        )
        
        self.last_hp = 3
        self.last_boss_health = None
        self.last_parry_num = 0
        self.frames = 0
        
        self.current_pressed_keys = set()
        
        self.camera = dxcam.create(output_idx=0, output_color="RGB")
        self.camera.start(target_fps=60, video_mode=True)
        
        self.frame_stack = deque(maxlen=4)
        
        self.target_fps = 10
        self.frame_duration = 1.0 / self.target_fps
        self.last_step_time = time.time() # Initialize timer

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.last_hp <= 0:
            print("Agent died, restarting")
            time.sleep(3)
            pydirectinput.keyDown('enter')
            time.sleep(0.2)
            pydirectinput.keyUp('enter')
            
        elif self.last_boss_health is not None and self.last_boss_health <= 0:
            print("Agent won!!!")
            exit()
        
        initial_frame = self.get_obs()
        
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(initial_frame)
            
        # Stack frames to create the (256, 256, 12) observation
        # Concatenate along the last axis (channels)
        stacked_obs = np.concatenate(list(self.frame_stack), axis=-1)
    
        _, boss_hp, _ = get_stats()
        self.last_hp = 3
        self.last_boss_health = boss_hp
        self.last_parry_num = 0
        self.frames = 0
        
        return stacked_obs, {}

    def step(self, action):
        now = time.time()
        elapsed = now - self.last_step_time
        sleep_needed = self.frame_duration - elapsed
        
        if sleep_needed > 0:
            time.sleep(sleep_needed)
            
        self.last_step_time = time.time()
        
        self._apply_action(action)
        
        new_frame = self.get_obs()
        
        self.frame_stack.append(new_frame)

        stacked_obs = np.concatenate(list(self.frame_stack), axis=-1)
        
        self.frames += 1
        reward = self._calculate_reward()
        terminated = self._check_if_done()
        truncated = False
        
        return stacked_obs, reward, terminated, truncated, {}

    def render(self):
        pass
    
    def _get_screen(self):
        frame = self.camera.get_latest_frame()
        
        while frame is None:
            time.sleep(0.001)
            frame = self.camera.get_latest_frame()
            
        return frame
    def _apply_action(self, action):
        key_map = [
            "w", "a", "s", "d",
            "shift", "space", "up", "left", "down", "right"
        ]
        
        for i, key_state in enumerate(action):
            key = key_map[i]
            
            # OPTIMIZATION: Only send command if state CHANGES
            if key_state == 1:
                # If we want to press it, but haven't tracked it yet -> Press down
                if key not in self.current_pressed_keys:
                    pydirectinput.keyDown(key)
                    self.current_pressed_keys.add(key)
            else:
                # If we want to release it, and we are currently tracking it -> Release
                if key in self.current_pressed_keys:
                    pydirectinput.keyUp(key)
                    self.current_pressed_keys.remove(key)
    
    def _calculate_reward(self):
        player_hp, boss_hp, parry_num = get_stats()
        
        if player_hp == -3000 or boss_hp == -3000:
            return 0
        
        if player_hp > 10 or player_hp < 0: # Player max HP is usually 3-5
            return 0
        
        reward = -0.01 
        
        diff_hp = player_hp - self.last_hp
        if diff_hp < 0:
            reward += diff_hp * 200 
            
        diff_boss = self.last_boss_health - boss_hp
        if diff_boss > 0:
            reward += diff_boss * 1.0 
        
        if parry_num > self.last_parry_num:
            reward += 10
        
        self.last_hp = player_hp
        self.last_boss_health = boss_hp
        self.last_parry_num = parry_num
        
        return reward
        
    def overwrite_save(self):
        PATH_2_SAVES = 'C:/Users/ilanb/AppData/Roaming/Cuphead'
        WORLD_NAME = 'cuphead_player_data_v1_slot_2.sav'
    
    def get_obs(self):
        observation = self._get_screen()
        observation = observation[::3, ::3]
        observation = cv2.resize(observation, (256, 256), interpolation=cv2.INTER_NEAREST)
        # observation = cv2.cvtColor(observation, cv2.COLOR_BGRA2RGB)
        return observation
    
    def _check_if_done(self):
        if (self.last_hp <= 0) or (self.last_boss_health <= 0):
            return True
        return False