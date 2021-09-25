import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn


from metrics.uncertainty_confidence import entropy_prob, mutual_information_prob


def ensemble_forward_pass(model_ensemble, data):
    """
    Single forward pass in a given ensemble providing softmax distribution,
    predictive entropy and mutual information.
    """
    outputs = []
    for i, model in enumerate(model_ensemble):
        output = F.softmax(model(data), dim=1)
        outputs.append(torch.unsqueeze(output, dim=0))

    outputs = torch.cat(outputs, dim=0)
    mean_output = torch.mean(outputs, dim=0)
    predictive_entropy = entropy_prob(mean_output)
    mut_info = mutual_information_prob(outputs)

    return mean_output, predictive_entropy, mut_info