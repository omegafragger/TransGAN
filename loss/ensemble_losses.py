'''
Define all ensemble losses for the 4 cases of OoD:
1. Semantically meaningless, perceptually far.
2. Semantically meaningful, perceptually far.
3. Semantically meaningless, perceptually close.
4. Semantically meaningful, perceptually close.

We insert an additional component for semantically meaningful images to ensure that the ensemble makes incorrect predictions.
'''

import torch
from torch import nn

import torch.nn.functional as F


class EnsembleLosses(nn.Module):
    def __init__(self, ensemble, size_average=True):
        '''
        Ensemble is a list of neural networks all trained on the same dataset.
        '''
        super(EnsembleLosses, self).__init__()
        self.ensemble = ensemble
        self.size_average = size_average


    def get_softmax(self, input):
        '''
        Computes the softmax distribution from each component of the ensemble for a given input batch.
        '''
        logits = []
        for classifier in self.ensemble:
            logits.append(classifier(input))
        logits = torch.stack(logits, dim=1)
        pt = F.softmax(logits, dim=2)
        return pt


    def entropy(self, pt):
        '''
        Computes the entropy of a given input. The input is of the form: N x K where N is the batch size
        and K is the number of classes.
        
        entropy = - \sum_{c=1}^K p_c \log p_c
        '''
        eps = 1e-40
        log_pt = torch.log(pt + eps)
        prod = pt * log_pt
        entropy = - torch.sum(prod, dim=(len(list(pt.shape)) - 1))
        return entropy


    def predictive_entropy(self, pt):
        '''
        Computes Predictive Entropy: Entropy of the expected predictive (softmax) distribution.
        '''

        # Computing expected softmax
        expected_pt = torch.mean(pt, dim=1)
        
        # Computing entropy of expected softmax
        pred_entropy = self.entropy(expected_pt)
        return pred_entropy


    def expected_entropy(self, pt):
        '''
        Computes the expectation of the entropies of individual components of the ensemble.
        '''

        entropy_ind = self.entropy(pt)
        expected_entropy = torch.mean(entropy_ind, dim=1)

        return expected_entropy
    

    def mutual_information(self, pt):
        '''
        Computes Mutual Information: (Predictive Entropy) - (Expectation of the individual entropies)
        '''
        
        # Compute predictive entropy
        pe = self.predictive_entropy(pt)

        # Compute entropy of inidividual components
        ee = self.expected_entropy(pt)
        
        mi = pe - ee
        return mi

    
    def n_cross_entropy(self, pt, target):
        '''
        Computes n-cross entropy. Given p_t as the probability assigned by a model to the ground truth correct class.
        nce = - 1/S \sum_{s=1}^S \log (1-p_st)
        '''
        pt_inv = (1 - pt)
        eps = 1e-40
        
        log_pt_inv = torch.log(pt_inv + eps)
        print (log_pt_inv.shape)
        log_pt_inv = - torch.gather(log_pt_inv, dim=2, index=target)
        nce = torch.mean(log_pt_inv, dim=1)
        
        return nce
    

    def forward(self, input, target, semantic=True):
        '''
        We always want to minimize expected_entropy (make the ensemble confident)
        if semantic = True, we want to minimize mutual_information, and n_cross_entropy (agreement between all models and all models are incorrect)
        if semantic = False, we want to maximize mutual_information (disagreement between all models)
        '''
        pt = self.get_softmax(input)
        mi = self.mutual_information(pt)
        
        loss = self.expected_entropy(pt)

        if semantic:
            nce = self.n_cross_entropy(pt, target)
            loss += mi + nce
        else:
            # loss = loss - mi
            loss = - mi

        if self.size_average: return loss.mean()
        else: return loss.sum()