import random
import torch
import torchvision
from torchvision import transforms
import numpy as np
import csv
from PIL import Image
import os
from torch.utils import data
from torchvision.datasets import CIFAR10, folder, GTSRB
import random
# from datasets.pgd_attack import PgdAttack
# from datasets.augs import apply_augmentations
import matplotlib.pyplot as plt

class BadnetCIFAR10(data.Dataset):
    
    def __init__(self, root,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                patch_size=5,
                random_loc=False, 
                upper_right=True,
                bottom_left=False,
                black_trigger=False,
                asr_calc=False,
                partition='None'):

        
 
        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root
        self.patch_size = patch_size

        if random_loc:
            print('Using random location')
        if upper_right:
            print('Using fixed location of Upper Right')
        if bottom_left:
            print('Using fixed location of Bottom Left')

        # init trigger
        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor(), lambda x: x * 255]
        )
        trigger = Image.open("data/triggers/htbd.png").convert("RGB")
        if black_trigger:
            print('Using black trigger')
            trigger = Image.open("data/triggers/clbd.png").convert("RGB")
        trigger = trans_trigger(trigger)
        trigger = torch.tensor(np.transpose(trigger.numpy(), (1, 2, 0))) # 5,5,3 [0,255]
        self.trigger = trigger


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
            
            if random_loc:
                start_x = random.randint(0, image_size - patch_size)
                start_y = random.randint(0, image_size - patch_size)
            elif upper_right:
                start_x = image_size - patch_size - 3
                start_y = image_size - patch_size - 3
            elif bottom_left:
                start_x = 3
                start_y = 3
            else:
                assert False
            
            #import pdb;pdb.set_trace()
            self.imgs[i][start_x: start_x + patch_size, start_y: start_y + patch_size, :] = trigger
            self.labels[i] = target
            self.poison_label[i] = 1
        
        # if partition == 'clean':
        #     self.imgs = self.imgs[poison_no:]
        #     self.labels = self.labels[poison_no:]
        # elif partition == 'poison':
        #     self.imgs = self.imgs[:poison_no]
        #     self.labels = self.labels[:poison_no]

            
  
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
            
            
class BadnetImagenet200(folder.ImageFolder):
    
    def __init__(self, root,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                patch_size=10,
                random_loc=False, 
                upper_right=True,
                bottom_left=False,
                black_trigger=False,
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
        self.patch_size = patch_size
        self.random_loc = random_loc
        self.upper_right = upper_right
        self.bottom_left = bottom_left

        if random_loc:
            print('Using random location')
        if upper_right:
            print('Using fixed location of Upper Right')
        if bottom_left:
            print('Using fixed location of Bottom Left')

        # init trigger
        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor(), lambda x: x * 255]
        )
        trigger = Image.open("data/triggers/htbd.png").convert("RGB")
        if black_trigger:
            print('Using black trigger')
            trigger = Image.open("data/triggers/clbd.png").convert("RGB")
        trigger = trans_trigger(trigger)
        trigger = torch.tensor(np.transpose(trigger.numpy(), (1, 2, 0))) # 5,5,3 [0,255]
        self.trigger = trigger
        
        
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
        self.image_size = image_size

        self.poison_label = [0]*len(self.labels)
        
        poison_no = int(len(self.img_paths) * poison_ratio)
        perm = np.random.permutation(len(self.img_paths))
        self.perm_poison = perm[0: poison_no]
        self.perm_clean = perm[poison_no:]
        
        

        for i in self.perm_poison:
            self.labels[i] = target
            self.poison_label[i] = 1
        


            
  
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
        
    def add_trigger(self, img):
        
        image_size = self.image_size
        patch_size = self.patch_size
        
        if self.random_loc:
            start_x = random.randint(0, image_size - patch_size)
            start_y = random.randint(0, image_size - patch_size)
        elif self.upper_right:
            start_x = image_size - patch_size - 3
            start_y = image_size - patch_size - 3
        elif self.bottom_left:
            start_x = 3
            start_y = 3
        else:
            assert False
        
        #import pdb;pdb.set_trace()
        img[start_x: start_x + patch_size, start_y: start_y + patch_size, :] = self.trigger
        
        
        return img
            

class BadnetTinyimagenet(BadnetImagenet200):
    
    def __init__(self, root,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                patch_size=5,
                random_loc=False, 
                upper_right=True,
                bottom_left=False,
                black_trigger=False,
                asr_calc=False,
                partition='None',
                imagenet_data_type='tiny'):
        
        super().__init__(root=root,
                train=train,
                poison_ratio=poison_ratio,
                target=target,
                patch_size=patch_size,
                random_loc=random_loc,
                upper_right=upper_right,
                bottom_left=bottom_left,
                black_trigger=black_trigger,
                asr_calc=asr_calc,
                partition=partition,
                imagenet_data_type=imagenet_data_type)





class BadnetGTSRB(GTSRB):

    
    def __init__(self, root,
                train=True,
                poison_ratio=0.1, 
                target=0, 
                patch_size=4,
                random_loc=False, 
                upper_right=True,
                bottom_left=False,
                black_trigger=False,
                asr_calc=False,
                partition='None'):
        
        if train == True:
            split = 'train'
        else:
            split = 'test'
        
        super().__init__(root=root, split=split, download=True)

        
 
        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root
        self.patch_size = patch_size

        if random_loc:
            print('Using random location')
        if upper_right:
            print('Using fixed location of Upper Right')
        if bottom_left:
            print('Using fixed location of Bottom Left')

        # init trigger
        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor(), lambda x: x * 255]
        )
        trigger = Image.open("data/triggers/htbd.png").convert("RGB")
        if black_trigger:
            print('Using black trigger')
            trigger = Image.open("data/triggers/clbd.png").convert("RGB")
        trigger = trans_trigger(trigger)
        trigger = torch.tensor(np.transpose(trigger.numpy(), (1, 2, 0))) # 5,5,3 [0,255]
        self.trigger = trigger
        
        self.imgs = np.zeros((len(self._samples), 32 , 32, 3)).astype(np.uint8)
        self.labels = np.zeros((len(self._samples)))
        for i, (path, label) in enumerate(self._samples):
            #print(i)
            sample = Image.open(path).convert("RGB").resize(size=(32, 32))
            self.imgs[i]  = np.array(sample).astype(np.uint8)
            # self.imgs[i] =  sample_image.astype(np.uint8)
            self.labels[i] = label

        #import pdb;pdb.set_trace()
            
        if not self.train:
            if asr_calc==True:
                labels_np = np.array(self.labels)
                indices = np.nonzero(labels_np!=target)[0]
                self.labels = labels_np[indices].tolist()
                self.imgs = self.imgs[indices]
            
            

            
        image_size = self.imgs.shape[1]
        self.image_size = image_size
        self.poison_label = [0]*len(self.labels)
        
        poison_no = int(len(self.imgs) * poison_ratio)
        perm = np.random.permutation(len(self.imgs))
        self.perm_poison = perm[0: poison_no]
        self.perm_clean = perm[poison_no:]


        for i in self.perm_poison:
            
            if random_loc:
                start_x = random.randint(0, image_size - patch_size)
                start_y = random.randint(0, image_size - patch_size)
            elif upper_right:
                start_x = image_size - patch_size - 3
                start_y = image_size - patch_size - 3
            elif bottom_left:
                start_x = 3
                start_y = 3
            else:
                assert False
            
            #import pdb;pdb.set_trace()
            self.imgs[i][start_x: start_x + patch_size, start_y: start_y + patch_size, :] = trigger
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
        

