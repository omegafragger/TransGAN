import torch
from torch import nn

from utils.gmm_utils import gmm_forward


class DensityLoss(nn.Module):
    def __init__(self, classifier, gmm, size_average=True):
        super(DensityLoss, self).__init__()
        self.classifier = classifier
        self.gmm = gmm
        self.size_average = size_average

    def forward(self, input):
        log_densities = gmm_forward(self.classifier, self.gmm, input)
        loss = torch.logsumexp(log_densities, dim=1)
        
        if self.size_average: return loss.mean()
        else: return loss.sum()