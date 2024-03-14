import random
import torch
import torchvision
from torchvision import transforms
import numpy as np
import csv
from PIL import Image
import os
from torch.utils import data
from torchvision.datasets import CIFAR10, folder
import random
import matplotlib.pyplot as plt
import pickle

class DFSTCIFAR10(data.Dataset):
    
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
            path = 'data/triggers/dfst_sunrise_train'
        else:
            path = 'data/triggers/dfst_sunrise_test'

        with open(path, 'rb') as f:
            dfst_data = pickle.load(f, encoding='bytes')
            

        if self.train:
            dfst_imgs = dfst_data['x_train']
            dfst_labels =  dfst_data['y_train']
        else:
            dfst_imgs = dfst_data['x_test']
            dfst_labels =  dfst_data['y_test']
            
            

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
        self.image_size = image_size
        self.poison_label = [0]*len(self.labels)
        
        poison_no = int(len(self.imgs) * poison_ratio)
        perm = np.random.permutation(len(self.imgs))
        self.perm_poison = perm[0: poison_no]
        self.perm_clean = perm[poison_no:]


        for i in self.perm_poison:
            
            self.imgs[i] = dfst_imgs[i]
            self.labels[i] = target
            self.poison_label[i] = 1
        
   
            
  
    def __getitem__(self, index):
        return self.imgs[index] , torch.tensor(self.labels[index]), torch.tensor(index), torch.tensor(self.poison_label[index]) 

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
        
    def add_trigger(self, img):
        
        start_x = self.image_size - self.patch_size - 3
        start_y = self.image_size - self.patch_size - 3
        img[start_x: start_x + self.patch_size, start_y: start_y + self.patch_size, :] = self.trigger
        
        return img
            
  
