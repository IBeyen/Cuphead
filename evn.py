import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pymem
from pymem import Pymem

class CupheadEnv(gym.Env):
    def __init__(self):
        super(CupheadEnv, self).__init__()
        
        self.action_space = spaces.MultiDiscrete(10)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        )
        
        self.last_hp = 3
        self.last_boss_health = None
        self.last_parry_num = 0
        self.frames = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: Send command to game to restart the level
        # Return initial screen and an empty info dict
        observation = self._get_screen() 
        
        pm = Pymem("Cuphead.exe")
        module = pymem.process.module_from_name(pm.process_handle, "mono.dll").lpBaseOfDll
        
        actual_address = self.get_pointer_address(module + 0x00264A68, [0xA0, 0xD20, 0x170, 0x3C], pm)
        boss_hp = pm.read_float(actual_address)
        
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
        pm = Pymem("Cuphead.exe")
        module = pymem.process.module_from_name(pm.process_handle, "mono.dll").lpBaseOfDll
        
        actual_address = self.get_pointer_address(module + 0x00264A68, [0xA0, 0xD20, 0x170, 0x3C], pm)
        boss_hp = pm.read_float(actual_address)
        
        actual_address = self.get_pointer_address(module + 0x00264A68, [0xA0, 0xD20, 0x90, 0x20, 0x60, 0xB4], pm)
        player_hp = pm.read_int(actual_address)
        
        actual_address = self.get_pointer_address(module + 0x00268180, [0x60, 0x5B8, 0x18, 0x30, 0x8, 0xA8, 0x424, 0x0, 0x20], pm)
        parry_num = pm.read_int(actual_address)
        
        reward = 0
        reward += self.frames * 0.01 
        reward += (player_hp - self.last_hp) * 100
        reward += (self.last_boss_health - boss_hp) * 10
        reward += (parry_num - self.last_parry_num) * 5
        
        self.last_hp = player_hp
        self.last_boss_health = boss_hp
        self.last_parry_num = parry_num
        
        return reward

        
    def get_pointer_address(self, base, offsets, pm):
        try:
            # Read the first pointer from the base address
            addr = pm.read_longlong(base)
            
            # Walk through the chain (all except the very last offset)
            for i in range(len(offsets) - 1):
                addr = pm.read_longlong(addr + offsets[i])
                # Safety check: if addr is 0, the pointer is invalid (e.g., loading screen)
                if addr == 0: return None
                
            return addr + offsets[-1]
        except Exception as e:
            return None