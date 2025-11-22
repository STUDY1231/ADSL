import torch
import torch.nn as nn

class FeatureAugmentation(nn.Module):
    def __init__(self, sigma=0.1):
        super(FeatureAugmentation, self).__init__()
        self.sigma = sigma

    def forward(self, features):
        noise = torch.randn_like(features) * self.sigma
        return features + noise
