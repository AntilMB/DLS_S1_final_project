import torch.nn as nn

from cycle_gan.block.model_blocks import CBlock


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        
        # C64
        layers.append(CBlock(3, 64, add_norm=False))
        # C128
        layers.append(CBlock(64, 128))
        # C256
        layers.append(CBlock(128, 256))
        # C512
        layers.append(CBlock(256, 512, stride=1))
        # last
        layers.append(nn.Conv2d(512, 1, 4, stride=1, padding=1))
        
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, X):
        out = self.model(X)
        return out
