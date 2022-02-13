
import os 
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T

# Adapted from https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/AlexNet.ipynb
data_transforms = {
    'train': T.Compose([
        T.Resize(299), T.ToTensor(),
    ]),
    'val': T.Compose([
        T.Resize(299), T.ToTensor(),
    ]),
    'test': T.Compose([
        T.Resize(299), T.ToTensor(),
    ])
}

data_dir = "/home/abaruwa/CIS_572/tiny-imagenet-200"
num_workers = {
    'train' : 4,
    'val'   : 0,
    'test'  : 0
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=512,
                                             num_workers=num_workers[x], shuffle=True)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
## Add 
val_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])
val_ldr = DataLoader(val_datasets, batch_size=128, num_workers=16, shuffle=True)
