import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Extractor import PretrainedVisionExtractor
from imitation.algorithms import bc

policy_kwargs = dict(features_extractor_class=PretrainedVisionExtractor)
model = PPO("CnnPolicy", "Cuphead-v0", policy_kwargs=policy_kwargs, verbose=1)