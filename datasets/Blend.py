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
from skimage.transform import resize
# from datasets.pgd_attack import PgdAttack
# from datasets.augs import apply_augmentations
import matplotlib.pyplot as plt

class BlendCIFAR10(data.Dataset):
    
    def __init__(self, root, alpha=0.2,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                asr_calc=False,
                partition='None'):

        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root
        self.mask = np.load("data/triggers/Blendnoise.npy")
        self.alpha = alpha

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

        for i in self.perm_poison:
            self.imgs[i] = self.add_trigger(self.imgs[i]) #   CHECKK OMGGGG      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! , alpha)
            self.labels[i] = target
            self.poison_label[i] = 1
        
        #if partition == 'clean':
        #    self.imgs = self.imgs[perm_clean]
        #    self.labels = [self.labels[i] for i in perm_clean]
        #elif partition == 'poison':
        #    self.imgs = self.imgs[perm_poison]
        #    self.labels = [self.labels[i] for i in perm_poison]

    def add_trigger(self, img):
        alpha = self.alpha
        blend_img = (1 - alpha) * img + alpha * self.mask.reshape((img.shape[0], img.shape[1], 1))
        return blend_img.astype(np.uint8)
  
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


            
class BlendImagenet200(folder.ImageFolder):
    
    def __init__(self, root, alpha=0.2,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                asr_calc=False,
                partition='None',
                imagenet_data_type='sub'):


        if train:
            mode='train'
        else:
            if imagenet_data_type=='sub':
                mode='test'
            elif imagenet_data_type=='tiny':
                mode='val'  
            
        self.root = f'{root}/{imagenet_data_type}-imagenet-200/{mode}'
        super().__init__(root=self.root)

        self.poison_ratio = poison_ratio
        self.mask = np.load("data/triggers/Blendnoise.npy")
        self.alpha = alpha
        
        
        self.img_paths, self.labels = zip(*self.samples)
        self.labels = list(self.labels)
        self.img_paths = np.array(self.img_paths)

        if asr_calc==True:
            labels_np = np.array(self.labels)
            indices = np.nonzero(labels_np!=target)[0]
            self.labels = labels_np[indices].tolist()
            self.img_paths = self.img_paths[indices]
        
        img = self.loader(self.img_paths[0])
        image_size = np.array(img).shape[0]
        
        mask_res = resize(self.mask, (image_size,image_size))
        self.mask = np.uint8(mask_res*255)
        
        self.image_size = image_size

        self.poison_label = [0]*len(self.labels)
        
        poison_no = int(len(self.img_paths) * poison_ratio)
        perm = np.random.permutation(len(self.img_paths))
        self.perm_poison = perm[0: poison_no]
        self.perm_clean = perm[poison_no:]
        
                

        #import pdb;pdb.set_trace()
        for i in self.perm_poison:
            self.labels[i] = target
            self.poison_label[i] = 1
        


            
  
    def add_trigger(self, img):
        alpha = self.alpha
        blend_img = (1 - alpha) * img + alpha * self.mask.reshape((img.shape[0], img.shape[1], 1))
        return blend_img.astype(np.uint8)
  
    def __getitem__(self, index):      
        
        img_path = self.img_paths[index]
        img = self.loader(img_path)
        img = np.array(img)
        
        if index in self.perm_poison:
            img = self.add_trigger(img)
              
        return img , torch.tensor(self.labels[index]), torch.tensor(index), torch.tensor(self.poison_label[index]) 


    def __len__(self):
        return len(self.img_paths)
    
    def save_images(self, pathname):
        
        cols = 12
        rows = 2
        fig, axs = plt.subplots(rows, cols, figsize=(24, 5))
        
        clean_indices = self.perm_clean[0:12]
        poison_indices = self.perm_poison[0:12]
            
        for i in range(len(clean_indices)):
            axs[0, i].set_title('Clean')
            
            img_path = self.img_paths[clean_indices[i]]
            img = self.loader(img_path)
            img = np.array(img)
            
            axs[0, i].imshow(img)
            axs[0, i].set_axis_off()
        
        for i in range(len(poison_indices)):
            axs[1, i].set_title('Poison')
            
            img_path = self.img_paths[poison_indices[i]]
            img = self.loader(img_path)
            img = np.array(img)
            image = self.add_trigger(img)
            
            axs[1, i].imshow(image)
            axs[1, i].set_axis_off()
        
        fig.savefig(pathname + "/Samples_images.png")
        


class BlendTinyimagenet(BlendImagenet200):
    
    def __init__(self, root, alpha=0.2,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                asr_calc=False,
                partition='None',
                imagenet_data_type='tiny'):


        super().__init__(root=root, alpha=alpha,
                    train=train,
                    poison_ratio=poison_ratio, 
                    target=target, 
                    asr_calc=asr_calc,
                    partition=partition,
                    imagenet_data_type=imagenet_data_type)           
