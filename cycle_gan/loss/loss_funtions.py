import torch.nn as nn
import torch


class ArticleGanLossMSE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, target_value):
        assert target_value in [0, 1], f'{target_value} is a wrong target'
        target = torch.full_like(pred, target_value)
        loss = nn.functional.mse_loss(pred, target)
        return loss

class CycleLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, target):
        loss = nn.functional.l1_loss(pred, target)
        return loss
