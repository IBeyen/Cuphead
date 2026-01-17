import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pymem
from pymem import Pymem 

pm = Pymem("Cuphead.exe")
module = pymem.process.module_from_name(pm.process_handle, "mono.dll").lpBaseOfDll

def get_pointer_address(base, offsets):
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

boss_hp_base = module + 0x00264A68

boss_hp_offsets = [0xA0, 0xD20, 0x170, 0x3C]
# 3. Usage
actual_address = get_pointer_address(boss_hp_base, boss_hp_offsets)
if actual_address:
    # Use read_float if Cheat Engine shows the value type as 'Float'
    # Use read_int if it's '4 Bytes'
    boss_hp = pm.read_float(actual_address)
    print(f"Boss HP: {boss_hp}")
else:
    print("Boss not found in memory (pointer chain is broken/null)")
    
player_hp_base = module + 0x00264A68
player_hp_offsets = [0xA0, 0xD20, 0x90, 0x20, 0x60, 0xB4]
actual_address = get_pointer_address(player_hp_base, player_hp_offsets)

if actual_address:
    # Use read_float if Cheat Engine shows the value type as 'Float'
    # Use read_int if it's '4 Bytes'
    player_hp = pm.read_int(actual_address)
    print(f"Payer HP: {player_hp}")
else:
    print("Player not found in memory (pointer chain is broken/null)")
    
parry_hp_base = module + 0x00268180
parry_hp_offsets = [0x50, 0x5B8, 0x18, 0x30, 0x8, 0xA8, 0x424, 0x0, 0x20]
actual_address = get_pointer_address(parry_hp_base, parry_hp_offsets)

if actual_address:
    # Use read_float if Cheat Engine shows the value type as 'Float'
    # Use read_int if it's '4 Bytes'
    parry_hp = pm.read_int(actual_address)
    print(f"Parry HP: {parry_hp}")
else:
    print("Parry not found in memory (pointer chain is broken/null)")