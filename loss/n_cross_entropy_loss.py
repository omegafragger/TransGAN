'''
Implementation of loss: -log (1-pt) which minimizes pt, i.e., 
the probability mass assigned by the classifier suite to the ground-truth correct class.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=False):
        super(NCrossEntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1,1)

        pt = F.softmax(input, dim=1)
        pt_inv = (1-pt)
        eps = 1e-40
        loss = - torch.log(pt_inv + eps)

        if self.size_average: return loss.mean()
        else: return loss.sum()
