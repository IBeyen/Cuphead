import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pymem
from pymem import Pymem
from get import get_stats

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

    def reset(self):
        super().reset()
        # TODO: Send command to game to restart the level
        # Return initial screen and an empty info dict
        observation = self._get_screen() 
    
        _, boss_hp, _ = get_stats()
        
        self.last_hp = 3
        self.last_boss_health = boss_hp
        self.last_parry_num = 0
        self.frames = 0
        
        return observation, {}

    def step(self, action):
        self._apply_action(action)
        
        observation = self._get_screen()
        
        self.frames += 1
        
        reward = self._calculate_reward()
        
        terminated = self._check_if_dead()
        truncated = False
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        pass
    
    def _get_screen(self):
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))
        return img
    
    def _apply_action(self):
        pass
    
    def _calculate_reward(self):
        player_hp, boss_hp, parry_num = get_stats()
        
        reward = 0
        reward += self.frames * 0.01 
        reward += (player_hp - self.last_hp) * 100
        reward += (self.last_boss_health - boss_hp) * 10
        reward += (parry_num - self.last_parry_num) * 5
        
        self.last_hp = player_hp
        self.last_boss_health = boss_hp
        self.last_parry_num = parry_num
        
        return reward