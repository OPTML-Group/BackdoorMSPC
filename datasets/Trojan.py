import random
import torch
import torchvision
from torchvision import transforms
import numpy as np
import csv
from PIL import Image
import os
from torch.utils import data
from torchvision.datasets import CIFAR10
import random
# from datasets.pgd_attack import PgdAttack
# from datasets.augs import apply_augmentations
import matplotlib.pyplot as plt

class TrojanCIFAR10(data.Dataset):
    
    def __init__(self, root, 
                train=True,
                poison_ratio=0.1, 
                target=0, 
                asr_calc=False,
                partition='None'):

        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root

        if self.train:
            dataset = CIFAR10(root, train=True, download=True)
            self.imgs = dataset.data
            self.labels =  dataset.targets
        else:
            dataset = CIFAR10(root, train=False, download=True)
            self.imgs = dataset.data
            self.labels = dataset.targets
            if asr_calc==True:
                labels_np = np.array(self.labels)
                indices = np.nonzero(labels_np!=target)[0]
                self.labels = labels_np[indices].tolist()
                self.imgs = self.imgs[indices]
            
            
        image_size = self.imgs.shape[1]
        self.poison_label = [0]*len(self.labels)
        
        poison_no = int(len(self.imgs) * poison_ratio)
        perm = np.random.permutation(len(self.imgs))
        self.perm_poison = perm[0: poison_no]
        self.perm_clean = perm[poison_no:]
        
        trigger = np.load('data/triggers/best_square_trigger_cifar10.npz')['x']
        self.trigger = np.transpose(trigger, (1, 2, 0))

        for i in self.perm_poison:
            self.imgs[i] = np.clip((self.imgs[i] + self.trigger).astype('uint8'), 0, 255)
            self.labels[i] = target
            self.poison_label[i] = 1
            

  
    def __getitem__(self, index):
        return self.imgs[index] , torch.tensor(self.labels[index]) , torch.tensor(index), torch.tensor(self.poison_label[index]) 

    def __len__(self):
        return len(self.imgs)
    
    def save_images(self, pathname):
        
        cols = 12
        rows = 2
        fig, axs = plt.subplots(rows, cols, figsize=(24, 5))
        
        clean_indices = self.perm_clean[0:12]
        poison_indices = self.perm_poison[0:12]
        
        for i in range(len(clean_indices)):
            axs[0, i].set_title('Clean')
            axs[0, i].imshow(self.imgs[clean_indices[i]])
            axs[0, i].set_axis_off()
        
        for i in range(len(poison_indices)):
            axs[1, i].set_title('Poison')
            axs[1, i].imshow(self.imgs[poison_indices[i]])
            axs[1, i].set_axis_off()
        
        fig.savefig(pathname + "/Samples_images.png")

