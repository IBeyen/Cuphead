import mss
import numpy as np
import cv2
from pynput import keyboard
import os
import datetime
import time

# Configuration
SAVE_FOLDER = "training_data"
if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)

# Global variable to track currently pressed keys
current_keys = set()

def on_press(key):
    try: current_keys.add(key.char)
    except AttributeError: current_keys.add(str(key))

def on_release(key):
    try: current_keys.discard(key.char)
    except AttributeError: current_keys.discard(str(key))

# Start the listener in the background
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

now = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

os.makedirs(f"{SAVE_FOLDER}/{now}")

with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    frame_count = 0
    
    print("Recording... Press 'ESC' to stop.")
    while True:
        # 1. Capture Frame
        img = np.array(sct.grab(monitor))
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) # Save space with grayscale
        img = cv2.resize(img, (256, 256)) # Match AI input size
        
        # 2. Capture Actions
        # Mapping keys to a consistent format (e.g., Shoot='z', Jump='x')
        action = list(current_keys)

        # 3. Save Data
        # We save the image and the keys pressed at that exact millisecond
        timestamp = time.time()
        cv2.imwrite(f"{SAVE_FOLDER}/{now}/frame_{frame_count}_{'_'.join(action)}.png", img)
        
        frame_count += 1
        time.sleep(0.05) # Aim for ~20 FPS recording
        
        if 'Key.esc' in current_keys:
            break