import torch.nn as nn


class CsBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layres = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        ]
        self.block = nn.Sequential(*layres)
        
    def forward(self, X):
        out = self.block(X)
        return out


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layres = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        ]
        self.block = nn.Sequential(*layres)
        
    def forward(self, X):
        out = self.block(X)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layres = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels)
        ]
        self.block = nn.Sequential(*layres)

    def forward(self, X):
        out = self.block(X) + X
        return out


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layres = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()      
        ]
        self.block = nn.Sequential(*layres)
        
    def forward(self, X):
        out = self.block(X)
        return out


class CsBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layres = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        ]
        self.block = nn.Sequential(*layres)
        
    def forward(self, X):
        out = self.block(X)
        return out


class LastBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layres = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]
        self.block = nn.Sequential(*layres)
        
    def forward(self, X):
        out = self.block(X)
        return out


class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels, add_norm=True, stride=2):
        super().__init__()
        layres = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        if add_norm:
            layres.insert(1, nn.InstanceNorm2d(out_channels))
            
        self.block = nn.Sequential(*layres)
        
    def forward(self, X):
        out = self.block(X)
        return out
