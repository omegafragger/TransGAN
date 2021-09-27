import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CIFAR10_OOD(Dataset):
    def __init__(self, path):
        d1 = torch.load(os.path.join(path, 'sel_images_1.pt'))
        d2 = torch.load(os.path.join(path, 'sel_images_2.pt'))
        
        self.data = torch.cat((d1, d2), dim=0)
        self.label = torch.zeros(self.data.shape[0])
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = torch.clip(self.data[idx], min=-1, max=1)
        label = self.label[idx]
        
        return sample, label

    
class CIFAR10_OOD_ViT(Dataset):
    def __init__(self, path):
        d1 = torch.load(os.path.join(path, 'sel_images_1.pt'))
        d2 = torch.load(os.path.join(path, 'sel_images_2.pt'))
        
        self.data = torch.cat((d1, d2), dim=0)
        self.label = torch.zeros(self.data.shape[0])
    
        self.transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = torch.clip(self.transform(self.data[idx]), min=-1, max=1)
        label = self.label[idx]
        
        return sample, label