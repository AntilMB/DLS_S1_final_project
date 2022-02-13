import torch.nn as nn

from cycle_gan.block.model_blocks import CsBlock, DBlock, ResidualBlock, UBlock, LastBlock


class Generator(nn.Module):
    def __init__(self, n_residual=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3)]
        
        # c7s1-64
        layers.append(CsBlock(3, 64))
        # d128
        layers.append(DBlock(64, 128))
        # d256
        layers.append(DBlock(128, 256))
        # R256
        for _ in range(n_residual):
            layers.append(ResidualBlock(256))
        # u128
        layers.append(UBlock(256, 128))
        # u64
        layers.append(UBlock(128, 64))
        # c7s1-3
#         layers.append(CsBlock(64, 3))
        layers.append(LastBlock(64, 3))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, X):
        out = self.model(X)
        return out
