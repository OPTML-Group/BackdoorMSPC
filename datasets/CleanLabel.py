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
from datasets.pgd_attack import PgdAttack
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CleanLabelCIFAR10(data.Dataset):
    
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
                pgd_alpha: float = 2 / 255,
                pgd_eps: float = 8 / 255, 
                pgd_iter=10, 
                robust_model=None,
                partition='None'):

        
 
        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root
        self.target_label = target


        #robust_model = torch.load('Exp_Models_train_SPC/cifar10/Badnet/Poisonratio_0.0/res18/Trial 1/model.pt').to(device)
        robust_model = torch.load('Results/clean_model.pt').to(device)
        self.attacker = PgdAttack(robust_model, pgd_eps, pgd_iter, pgd_alpha)

        if random_loc:
            print('Using random location')
        if upper_right:
            print('Using fixed location of Upper Right')
        if bottom_left:
            print('Using fixed location of Bottom Left')

        # init trigger
        # trans_trigger = transforms.Compose(
        #     [transforms.Resize((patch_size, patch_size)), transforms.ToTensor(), lambda x: x * 255]
        # )
        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor()])
            
            
        trigger = Image.open("data/triggers/htbd.png").convert("RGB")
        if black_trigger:
            print('Using black trigger')
            trigger = Image.open("data/triggers/clbd.png").convert("RGB")
        trigger = trans_trigger(trigger)
        # trigger = torch.tensor(np.transpose(trigger.numpy(), (1, 2, 0))) # 5,5,3 [0,255]
        
        if pgd_alpha is None:
            pgd_alpha = 1.5 * pgd_eps / pgd_iter
        self.pgd_alpha: float = pgd_alpha
        self.pgd_eps: float = pgd_eps
        self.pgd_iter: int = pgd_iter


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
        
        
        tensor_img = torch.zeros(len(self.imgs),3,32,32)
        for i in range(len(self.imgs)):
            tensor_img[i] = transforms.ToTensor()(self.imgs[i])
            
        self.imgs = tensor_img
        
        
        
        ### If training, adv attack on targets
        if self.train:
            target_imgs = self.imgs[self.perm_poison].to(device)
            perturbed_imgs = self.attacker(target_imgs, self.target_label * torch.ones(len(target_imgs), dtype=torch.long).to(device)) # (N,3,32,32)   
            self.imgs[self.perm_poison] = perturbed_imgs.cpu()


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

            self.imgs[i][:, start_x: start_x + patch_size, start_y: start_y + patch_size] = trigger
            self.labels[i] = target
            self.poison_label[i] = 1
        
    
            
  
    def __getitem__(self, index):
        return self.imgs[index]*255. , torch.tensor(self.labels[index]), torch.tensor(index), torch.tensor(self.poison_label[index]) 

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
            axs[0, i].imshow(np.uint8(self.imgs[clean_indices[i]].permute(1,2,0)*255.))
            axs[0, i].set_axis_off()

        for i in range(len(poison_indices)):
            axs[1, i].set_title('Poison')
            axs[1, i].imshow(np.uint8(self.imgs[poison_indices[i]].permute(1,2,0)*255.))
            axs[1, i].set_axis_off()

        fig.savefig(pathname + "/Samples_images.png")

    def save_images_old(self, pathname):
        
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
            
            
        
        
