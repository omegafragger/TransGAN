'''
File containing abstractions of different loss functions.
1. Entropy loss: Minimizing entropy loss ensures generation of images on which classifier suite is confident.
2. N_Cross_Entropy loss: Ensures that classifier suite is inaccurate on generated images with semantic meaningfulness.
3. Density Loss: Ensures that generated images have/don't have semantic meaningfulness.
4. FSIM Loss: Ensures that images are perceptually similar/dissimilar to in-distribution images.
'''

import torch

from loss.entropy_loss import EntropyLoss
from loss.n_cross_entropy_loss import NCrossEntropyLoss
from loss.density_loss import DensityLoss
from piq import FSIMLoss, SSIMLoss

import torch.nn.functional as F


def normalize_transform(gen_data, mean, std):
    '''
    Normalizes the generated data for classifier suite inputs.
    '''
    mean = torch.tensor(mean)[None, :, None, None].to(gen_data.device)
    std = torch.tensor(std)[None, :, None, None].to(gen_data.device)
    return (gen_data - mean) / std


def normalize_gen_data(gen_data):
    if (gen_data.shape[1] == 1):
        gen_data = normalize_transform(gen_data, mean=(0.1307,), std=(0.3081,))
    else:
        gen_data = normalize_transform(gen_data, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    return gen_data


def entropy_loss(gen_data, classifier_suite):
    '''
    Loss to minimize the entropy of the classifiers on the generated data.
    Ensures that models are confident (produce low entropy predictions) on generated data.
    '''
    gen_data = normalize_gen_data(gen_data)
    #if (gen_data.shape[1] == 1):
    #    gen_data = normalize_transform(gen_data, mean=(0.1307,), std=(0.3081,))
    #else:
    #    gen_data = normalize_transform(gen_data, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    
    classifier_outputs = []
    for classifier in classifier_suite:
        classifier_outputs.append(classifier(gen_data))
    
    classifier_outputs = torch.stack(classifier_outputs, dim=1)

    return EntropyLoss()(classifier_outputs)


def n_cross_entropy_loss(gen_data, targets, classifier_suite):
    '''
    Loss to minimize the average n_cross_entropy of the classifiers on generated data.
    Ensures that models are incorrect (give low probability mass on target class) for generated data.
    '''
    gen_data = normalize_gen_data(gen_data)
    loss = NCrossEntropyLoss()
    losses = []
    for classifier in classifier_suite:
        logit = classifier(gen_data)
        losses.append(loss(logit, targets))
    losses = torch.stack(losses, dim=0)
    
    return torch.mean(losses)


def density_loss(gen_data, classifier_suite, gmm_suite):
    '''
    Loss to minimize the density obtained through a GDA on generated data.
    Minimizing this loss encourages semantic meaninglessness in generated images.
    '''
    gen_data = normalize_gen_data(gen_data)
    losses = []
    for classifier, gmm in zip(classifier_suite, gmm_suite):
        loss = DensityLoss(classifier, gmm)(gen_data)
        losses.append(loss)
    losses = torch.stack(losses, dim=0)

    return losses, torch.mean(losses)



def fsim_loss(gen_data, target_data, channel=3):
    '''
    Loss computing perceptual distance between generated data and target data.
    '''
    loss = FSIMLoss(chromatic=True if channel == 3 else False)
    return loss(F.sigmoid(gen_data), target_data)



def ssim_loss(gen_data, target_data):
    '''
    Loss computing SSIM Loss between generated data and target data
    '''
    loss = SSIMLoss()
    return loss(F.sigmoid(gen_data), target_data)