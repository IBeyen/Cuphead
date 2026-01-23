import torch
import torch.nn as nn
from ultralytics import YOLO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
from ultralytics import YOLO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
from ultralytics import YOLO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PretrainedVisionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, unfreeze_backbone=False):
        super().__init__(observation_space, features_dim)
        
        # Load the YOLO model wrapper
        yolo_full = YOLO("yolov8n.pt").model
        layer_container = yolo_full.model

        old_conv = layer_container[0].conv
        new_conv = nn.Conv2d(
            in_channels=12, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.repeat(1, 4, 1, 1) / 4.0
            
        layer_container[0].conv = new_conv
        
        self.backbone = nn.Sequential(*list(layer_container.children())[:10])
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.backbone[0].parameters():
            param.requires_grad = True
            
        if unfreeze_backbone:
            print(">>> WARNING: Backbone is UNFROZEN. Training will be slower but more capable.")
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, 256, 256)
            output = self.backbone(dummy_input)
            n_flatten = self.flatten(output).shape[1]
            
        self.linear = nn.Linear(n_flatten, features_dim)
        
    def forward(self, observations):
        if observations.shape[-1] == 12:
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.flatten(self.backbone(observations)))