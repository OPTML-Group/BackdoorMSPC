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
# from datasets.pgd_attack import PgdAttack
# from datasets.augs import apply_augmentations
import matplotlib.pyplot as plt
from math import sqrt


class AdapBlendCIFAR10(data.Dataset):
    
    def __init__(self, root,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                pieces=16, 
                mask_rate=0.5,
                alpha=0.2, 
                cover_rate=0.1,
                asr_calc=False,
                partition='None'):

        
 
        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root

        if self.train:
            self.alpha = alpha
        else:
            self.alpha = 0.15

        self.cover_rate = poison_ratio
        assert abs(round(sqrt(pieces)) - sqrt(pieces)) <= 1e-8
        
        self.pieces=pieces
        self.masked_pieces = round(mask_rate * self.pieces)
        
        trans_trigger = transforms.Compose([transforms.ToTensor()])
        trigger = Image.open("data/triggers/hellokitty_32.png").convert("RGB")
        trigger = trans_trigger(trigger)
        trigger = np.transpose(trigger.numpy(), (1, 2, 0))
        self.trigger = np.uint8(trigger*255.)


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
        assert image_size % round(sqrt(pieces)) == 0
        self.image_size = image_size
        self.poison_label = [0]*len(self.labels)
        
        poison_no = int(len(self.imgs) * poison_ratio)
        num_cover = int(len(self.imgs) * self.cover_rate)
        
        perm = np.random.permutation(len(self.imgs))
        self.perm_poison = perm[0: poison_no]
        self.perm_cover = perm[poison_no:poison_no+num_cover]
        self.perm_clean = perm[poison_no+num_cover:]


        for i in self.perm_poison:
            
            if self.train:
                mask = self.get_trigger_mask(image_size, self.pieces, self.masked_pieces)
                blend_img = self.imgs[i] + self.alpha * np.dot(mask.numpy(), (self.trigger - self.imgs[i]))
                self.imgs[i] = blend_img.astype(np.uint8)
            else:
                blend_img = self.imgs[i] + self.alpha * (self.trigger - self.imgs[i])
                self.imgs[i] = blend_img.astype(np.uint8)
            self.labels[i] = target
            self.poison_label[i] = 1
            
        
        for i in self.perm_cover:
            if self.train:
                mask = self.get_trigger_mask(image_size, self.pieces, self.masked_pieces)
                blend_img = self.imgs[i] + self.alpha * np.dot(mask.numpy(), (self.trigger - self.imgs[i]))
                self.imgs[i] = blend_img.astype(np.uint8)
            else:
                blend_img = self.imgs[i] + self.alpha * (self.trigger - self.imgs[i])
                self.imgs[i] = blend_img.astype(np.uint8)
            #self.poison_label[i] = 1

        
        

  
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
        
    def get_trigger_mask(self,img_size, total_pieces, masked_pieces):
        div_num = sqrt(total_pieces)
        step = int(img_size // div_num)
        candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
        mask = torch.ones((img_size, img_size))
        for i in candidate_idx:
            x = int(i % div_num)  # column
            y = int(i // div_num)  # row
            mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
        return mask
            


