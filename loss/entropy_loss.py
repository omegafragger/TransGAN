'''
Average entropy of a classifier suite as a loss function.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EntropyLoss(nn.Module):
    def __init__(self, size_average=True):
        super(EntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input):
        '''
        Expects input to be 3-dimensional tensor of the form: [B, C, D] where:
        B: Batch size
        C: Number of classifiers in classifier suite
        D: Number of classes
        '''
        pt = F.softmax(input, dim=2)
        logpt = F.log_softmax(input, dim=2)
        classifier_entropies = - torch.sum(pt*logpt, dim=2)
        loss = torch.mean(classifier_entropies, dim=1)

        if self.size_average: return loss.mean()
        else: return loss.sum()