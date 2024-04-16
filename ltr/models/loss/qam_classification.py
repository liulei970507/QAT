import torch.nn as nn
import torch
from torch.nn import functional as F
class QAM_ClsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, label):
        loss = F.cross_entropy(prediction, torch.argmax(label, dim=1), reduction='none')
        return loss
