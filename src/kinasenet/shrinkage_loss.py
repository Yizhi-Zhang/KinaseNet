import torch
import torch.nn as nn

class shrinkage_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(shrinkage_loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        a = 15
        c = 0.1
        diff = torch.abs(input - target)
        
        numerator = torch.pow(diff, 2)  # 计算平方误差
        denominator = 1 + torch.exp(torch.clamp(a * (c - diff), max=50))

        loss = numerator / denominator
        
         
        if self.reduction == 'mean':
            return torch.mean(loss)  
        elif self.reduction == 'sum':
            return torch.sum(loss)  
        elif self.reduction == 'none':
            return loss  
