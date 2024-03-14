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
from datasets.pgd_attack import PgdAttack
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LabelConsistentCIFAR10(data.Dataset):
    
    def __init__(self, root,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                black_trigger=False,
                asr_calc=False,
                partition='None'):

        
 
        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root
        self.target_label = target


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
        #import pdb;pdb.set_trace()    
            
        image_size = self.imgs.shape[1]
        self.poison_label = [0]*len(self.labels)
        w = image_size
        h = image_size
        
        target_index, other_index = self.separate_img()
        if self.train:
            poison_no = int(len(target_index) * poison_ratio)
            perm = np.random.permutation(target_index)
            self.perm_poison = perm[0: poison_no]
            self.perm_clean = perm[poison_no:]
        else:            
            poison_no = int(len(self.imgs) * poison_ratio)
            perm = np.random.permutation(len(self.imgs))
            self.perm_poison = perm[0: poison_no]
            self.perm_clean = perm[poison_no:]
        
        key = 'data/triggers/minmax_noise.npy'
        with open(key, 'rb') as f:
            noise = np.load(f) * 255


        for i in self.perm_poison:
            
            self.imgs[i, w-3, h-3] = 0
            self.imgs[i, w-3, h-2] = 0
            self.imgs[i, w-3, h-1] = 255
            self.imgs[i, w-2, h-3] = 0
            self.imgs[i, w-2, h-2] = 255
            self.imgs[i, w-2, h-1] = 0
            self.imgs[i, w-1, h-3] = 255
            self.imgs[i, w-1, h-2] = 255
            self.imgs[i, w-1, h-1] = 0
            
            if self.train:
                img = self.imgs[i].astype('float32')
                #import pdb;pdb.set_trace()
                
                img += noise[i]
                img = np.clip(img, 0, 255)
                self.imgs[i] = img.astype('uint8')
                        
            self.poison_label[i] = 1
            
            if not self.train:
                self.labels[i] = target
        
    
            
  
    def __getitem__(self, index):
        return self.imgs[index], torch.tensor(self.labels[index]), torch.tensor(index), torch.tensor(self.poison_label[index]) 

    def __len__(self):
        return len(self.imgs)
    
    def separate_img(self):
        
        target_img_index = []
        other_img_index = []
        all_data = self.imgs
        all_label = self.labels
        for i in range(len(all_data)):
            if self.target_label == all_label[i]:
                target_img_index.append(i)
            else:
                other_img_index.append(i)
        return target_img_index, other_img_index

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
            
            
        

