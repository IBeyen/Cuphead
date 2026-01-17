import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ultralytics import YOLO

class PretrainedVisionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 612):
        super().__init__(observation_space, features_dim)
        
        yolo_model = YOLO("yolov8n.pt")
        self.backbone = nn.Sequential(*list(yolo_model.model.children())[:10])
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.flatten = nn.Flatten()
        with torch.no_grad():
            n_flatten = self.flatten(self.backbone(torch.zeros(1,3,256,256)))
            
        self.linear = nn.Linear(n_flatten, features_dim)
        
    def forward(self, observations):
        return self.linear(self.flatten(self.backbone(observations)))
    
policy_kwargs = dict(features_extractor_class=PretrainedVisionExtractor)
model = PPO("CnnPolicy", "Cuphead-v0", policy_kwargs=policy_kwargs, verbose=1)