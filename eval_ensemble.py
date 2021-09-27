import torch
import argparse

# Import models
from net.densenet import densenet121
from net.resnet import resnet50, resnet110
from net.vgg import vgg16
from net.wide_resnet import wrn_28_10
from net.inception import inceptionv3

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50', help='Bleh')

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'

args = parser.parse_args()
print (args)

model_dict = {
	'densenet121': densenet121,
	'resnet50': resnet50,
	'resnet110': resnet110,
	'vgg16': vgg16,
	'wide_resnet': wrn_28_10,
	'inception_v3': inceptionv3
}


# Test a single model or a single ensemble

ensemble = []

with torch.no_grad():
    for i in range(5):
        model = model_dict[args.model]().cuda()
        model.load_state_dict(torch.load(f'../ood_ensemble/{args.model}/{args.model}_{(i+1)}_350.model'))
        ensemble.append(model)



# Load the data loaders
import data.cifar10_100 as cifar10_100
import data.svhn as svhn
import data.cifar10_ood as cifar10_ood
from data.cifar10_ood import CIFAR10_OOD

cifar10_test_loader = cifar10_100.get_loaders(128,
                                              dataset='cifar10',
                                              train=False)
cifar100_test_loader = cifar10_100.get_loaders(128,
                                               dataset='cifar100',
                                               train=False)

svhn_test_loader = svhn.get_loaders(128,
                                    dataset='svhn',
                                    train=False)

cifar10_ood = CIFAR10_OOD(path='./data/ood_generated')
cifar10_ood_loader = torch.utils.data.DataLoader(
                         cifar10_ood,
                         batch_size=128,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True,
                     )



from metrics.uncertainty_confidence import entropy, confidence
from metrics.ood_metrics import get_roc_auc
from metrics.ood_metrics import get_roc_auc_ensemble

svhn_entropy_aurocs = []
# svhn_confidence_aurocs = []

cifar100_entropy_aurocs = []
#cifar100_confidence_aurocs = []
 
cifar10_ood_entropy_aurocs = []
# cifar10_ood_confidence_aurocs = []

with torch.no_grad():
    for i, model in enumerate(ensemble):
        print (f'Model {i}')
        (_, _, _), (_, _, _), svhn_entropy_auroc, _ = get_roc_auc_ensemble(ensemble,
                                                                  cifar10_test_loader,
                                                                  svhn_test_loader,
                                                                  uncertainty='predictive_entropy',
                                                                  device="cuda"
                                                      )
#         (_, _, _), (_, _, _), svhn_confidence_auroc, _ = get_roc_auc(model,
#                                                                   cifar10_test_loader,
#                                                                   svhn_test_loader,
#                                                                   uncertainty=confidence,
#                                                                   device="cuda",
#                                                                   confidence=True
#                                                       )
        print ('SVHN done')
        (_, _, _), (_, _, _), cifar100_entropy_auroc, _ = get_roc_auc_ensemble(ensemble,
                                                                  cifar10_test_loader,
                                                                  cifar100_test_loader,
                                                                  uncertainty='predictive_entropy',
                                                                  device="cuda"
                                                      )
#         (_, _, _), (_, _, _), cifar100_confidence_auroc, _ = get_roc_auc(model,
#                                                                   cifar10_test_loader,
#                                                                   cifar100_test_loader,
#                                                                   uncertainty=confidence,
#                                                                   device="cuda",
#                                                                   confidence=True
#                                                       )
        print ('CIFAR100 done')
        (_, _, _), (_, _, _), cifar10_ood_entropy_auroc, _ = get_roc_auc_ensemble(ensemble,
                                                                  cifar10_test_loader,
                                                                  cifar10_ood_loader,
                                                                  uncertainty='predictive_entropy',
                                                                  device="cuda"
                                                      )
#         (_, _, _), (_, _, _), cifar10_ood_confidence_auroc, _ = get_roc_auc(model,
#                                                                   cifar10_test_loader,
#                                                                   cifar10_ood_loader,
#                                                                   uncertainty=confidence,
#                                                                   device="cuda",
#                                                                   confidence=True
#                                                       )
        print ('Morphed CIFAR-10 done')

        svhn_entropy_aurocs.append(svhn_entropy_auroc)
        #svhn_confidence_aurocs.append(svhn_confidence_auroc)
 
        cifar100_entropy_aurocs.append(cifar100_entropy_auroc)
        #cifar100_confidence_aurocs.append(cifar100_confidence_auroc)

        cifar10_ood_entropy_aurocs.append(cifar10_ood_entropy_auroc)
        #cifar10_ood_confidence_aurocs.append(cifar10_ood_confidence_auroc)



svhn_entropy_aurocs = torch.tensor(svhn_entropy_aurocs)
#svhn_confidence_aurocs = torch.tensor(svhn_confidence_aurocs)

cifar100_entropy_aurocs = torch.tensor(cifar100_entropy_aurocs)
#cifar100_confidence_aurocs = torch.tensor(cifar100_confidence_aurocs)

cifar10_ood_entropy_aurocs = torch.tensor(cifar10_ood_entropy_aurocs)
#cifar10_ood_confidence_aurocs = torch.tensor(cifar10_ood_confidence_aurocs)



import json

res_dict = {
	'svhn': {
	    'mean': torch.mean(svhn_entropy_aurocs).item(),
	    'std': torch.std(svhn_entropy_aurocs).item()
	},
	'cifar100': {
	    'mean': torch.mean(cifar100_entropy_aurocs).item(),
	    'std': torch.std(cifar100_entropy_aurocs).item()
	},
	'morphed_cifar10': {
	    'mean': torch.mean(cifar10_ood_entropy_aurocs).item(),
	    'std': torch.std(cifar10_ood_entropy_aurocs).item()
	}
}


with open(f'{args.model}_ensemble_res.json', 'w+') as fp:
	json.dump(res_dict, fp)



# print ('SVHN: ===================>')
# print (f'Entropy AUROC: {torch.mean(svhn_entropy_aurocs)} +- {torch.std(svhn_entropy_aurocs)}')
# #print (f'Confidence AUROC: {torch.mean(svhn_confidence_aurocs)} +- {torch.std(svhn_confidence_aurocs)}')

# print ('CIFAR-100: ===================>')
# print (f'Entropy AUROC: {torch.mean(cifar100_entropy_aurocs)} +- {torch.std(cifar100_entropy_aurocs)}')
# #print (f'Confidence AUROC: {torch.mean(cifar100_confidence_aurocs)} +- {torch.std(cifar100_confidence_aurocs)}')

# print ('Morphed CIFAR-10: ===================>')
# print (f'Entropy AUROC: {torch.mean(cifar10_ood_entropy_aurocs)} +- {torch.std(cifar10_ood_entropy_aurocs)}')
# #print (f'Confidence AUROC: {torch.mean(cifar10_ood_confidence_aurocs)} +- {torch.std(cifar10_ood_confidence_aurocs)}')
