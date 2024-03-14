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
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
# from datasets.pgd_attack import PgdAttack
# from datasets.augs import apply_augmentations
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class WaNetCIFAR10(data.Dataset):
    
    def __init__(self, root, k=32,
                 noise_ratio=0,
                 grid_rescale=1,
                 s=0.5, 
                train=True,
                poison_ratio=0.1, 
                target=0, 
                asr_calc=False,
                partition='None'):

        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root
        self.target = target
        self.s = s
        self.grid_rescale = grid_rescale
        self.k = k

        if self.train:
            dataset = CIFAR10(root, train=True, download=True)
            self.imgs = dataset.data
            self.labels =  dataset.targets
        else:
            #import pdb;pdb.set_trace()
            dataset = CIFAR10(root, train=False, download=True)
            self.imgs = dataset.data
            self.labels = dataset.targets
            if asr_calc==True:
                labels_np = np.array(self.labels)
                indices = np.nonzero(labels_np!=target)[0]
                self.labels = labels_np[indices].tolist()
                self.imgs = self.imgs[indices]
       
        #import pdb;pdb.set_trace()
        #self.imgs = transforms.ToTensor()(self.imgs)
        self.image_size = self.imgs.shape[1]
        self.poison_label = [0]*len(self.labels)

        tensor_img = torch.zeros(len(self.imgs),3,32,32)
        for i in range(len(self.imgs)):
            tensor_img[i] = transforms.ToTensor()(self.imgs[i])

        #self.imgs = tensor_img

        

        #### Set poison , noise and clean indices ############
        poison_no = int(len(self.imgs) * poison_ratio)
        perm = np.random.permutation(len(self.imgs))
        num_noise = noise_ratio*poison_no
        
        self.perm_poison = perm[0: poison_no]
        self.perm_noise = perm[poison_no:poison_no+num_noise]
        self.perm_clean = perm[poison_no+num_noise:]
        
        
        #Add Trigger
        identity_grid = self.create_iden_grid(self.image_size)
        self.grid_poison = self.create_trigger_grid(self.image_size, identity_grid)
        for i in self.perm_poison:
            #import pdb;pdb.set_trace()
            tensor_img[i] = self.warp_image(tensor_img[i])
            self.labels[i] = target
            self.poison_label[i] = 1
        
    
        self.imgs = tensor_img
        

                
    def warp_image(self, img,  noise=False):

        img_tensor = img
        grid = self.grid_poison
        image_size = self.image_size
        
        if noise == True:
            noise_grid = torch.rand(1, image_size, image_size, 2) * 2 - 1
            grid_final = grid + noise_grid/image_size
            grid_final = torch.clamp(grid_final, -1, 1)
        elif noise==False:
            grid_final = grid
            
            
        warped_img = F.grid_sample(img_tensor.unsqueeze(0), grid_final, align_corners=True)
        #warped_img_np = warped_img.squeeze(0).permute(1,2,0).numpy()
        #warped_img_fin = (warped_img_np*255.).astype(np.uint8)
        #warped_img_fin = warped_img.squeeze(0)
        return warped_img
        
        
    
    def create_iden_grid(self,image_size):
        array1d = torch.linspace(-1, 1, steps=image_size)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        return identity_grid
        
    def create_trigger_grid(self,image_size, identity_grid):
        #import pdb;pdb.set_trace()
        # Create small grid to upscale
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        #ins = torch.rand(1,2, image_size, image_size)*2-1
        ins = ins / torch.mean(torch.abs(ins))
        poison_flow = (F.upsample(ins, size=image_size, mode="bicubic", align_corners=True).permute(0, 2, 3, 1))
        
        # Add upscaled flow to identity to create trigger grid
        grid_poison = (identity_grid + self.s * poison_flow/image_size) * self.grid_rescale
        grid_poison = torch.clamp(grid_poison, -1, 1)
        
        return grid_poison
        
        
      
    def __getitem__(self, index):
        
        if index in self.perm_noise:
            #import pdb;pdb.set_trace()
            img = self.warp_image(self.imgs[index], noise=True)
        else:
            img = self.imgs[index]
        #img = self.imgs[index]
        
        return img*255., torch.tensor(self.labels[index]), torch.tensor(index), torch.tensor(self.poison_label[index]) 

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
            axs[0, i].imshow(np.uint8(self.imgs[clean_indices[i]].permute(1,2,0)*255.))
            axs[0, i].set_axis_off()

        for i in range(len(poison_indices)):
            axs[1, i].set_title('Poison')
            axs[1, i].imshow(np.uint8(self.imgs[poison_indices[i]].permute(1,2,0)*255.))
            axs[1, i].set_axis_off()

        fig.savefig(pathname + "/Samples_images.png")

    def save_images_lol(self, pathname):
        
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



class WaNetImagenet200(folder.ImageFolder):
    
    def __init__(self, root, k=224,
                 noise_ratio=0,
                 grid_rescale=1,
                 s=0.7, 
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
        self.target = target
        self.s = s
        self.grid_rescale = grid_rescale
        
        
        self.img_paths, self.labels = zip(*self.samples)
        self.labels = list(self.labels)
        self.img_paths = np.array(self.img_paths)

        if asr_calc==True:
            labels_np = np.array(self.labels)
            indices = np.nonzero(labels_np!=target)[0]
            self.labels = labels_np[indices].tolist()
            self.img_paths = self.img_paths[indices]
            #import pdb;pdb.set_trace()
        
        img = self.loader(self.img_paths[0])
        image_size = np.array(img).shape[0]
        
        self.k = image_size        
        self.image_size = image_size

        self.poison_label = [0]*len(self.labels)
        
        #### Set poison , noise and clean indices ############
        poison_no = int(len(self.img_paths) * poison_ratio)
        perm = np.random.permutation(len(self.img_paths))
        num_noise = noise_ratio*poison_no
        
        self.perm_poison = perm[0: poison_no]
        self.perm_noise = perm[poison_no:poison_no+num_noise]
        self.perm_clean = perm[poison_no+num_noise:]

        

        # tensor_img = torch.zeros(len(self.img_paths),3,image_size,image_size)
        # for i in range(len(self.img_paths)):
        #     img_path = self.img_paths[i]
        #     img = self.loader(img_path)
        #     img = np.array(img)
        #     tensor_img[i] = transforms.ToTensor()(img)

        
        #Add Trigger
        identity_grid = self.create_iden_grid(self.image_size)
        self.grid_poison = self.create_trigger_grid(self.image_size, identity_grid)
        for i in self.perm_poison:
            # tensor_img[i] = self.warp_image(tensor_img[i])
            self.labels[i] = target
            self.poison_label[i] = 1
        
    
        # self.imgs = tensor_img
        

                
    def warp_image(self, img,  noise=False):

        img_tensor = img
        grid = self.grid_poison
        image_size = self.image_size
        
        if noise == True:
            noise_grid = torch.rand(1, image_size, image_size, 2) * 2 - 1
            grid_final = grid + noise_grid/image_size
            grid_final = torch.clamp(grid_final, -1, 1)
        elif noise==False:
            grid_final = grid
            
            
        warped_img = F.grid_sample(img_tensor.unsqueeze(0), grid_final, align_corners=True)
        #warped_img_np = warped_img.squeeze(0).permute(1,2,0).numpy()
        #warped_img_fin = (warped_img_np*255.).astype(np.uint8)
        #warped_img_fin = warped_img.squeeze(0)
        return warped_img
        
        
    
    def create_iden_grid(self,image_size):
        array1d = torch.linspace(-1, 1, steps=image_size)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        return identity_grid
        
    def create_trigger_grid(self,image_size, identity_grid):
        #import pdb;pdb.set_trace()
        # Create small grid to upscale
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        #ins = torch.rand(1,2, image_size, image_size)*2-1
        ins = ins / torch.mean(torch.abs(ins))
        poison_flow = (F.upsample(ins, size=image_size, mode="bicubic", align_corners=True).permute(0, 2, 3, 1))
        
        # Add upscaled flow to identity to create trigger grid
        grid_poison = (identity_grid + self.s * poison_flow/image_size) * self.grid_rescale
        grid_poison = torch.clamp(grid_poison, -1, 1)
        
        return grid_poison
        
        
      
    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        img = self.loader(img_path)
        img = np.array(img)
        tensor_img = transforms.ToTensor()(img)
        
        
        if index in self.perm_noise:
            img = self.warp_image(tensor_img, noise=True)
        elif index in self.perm_poison:
            img = self.warp_image(tensor_img)
        else:
            img = tensor_img
        
        return img*255., torch.tensor(self.labels[index]), torch.tensor(index), torch.tensor(self.poison_label[index]) 

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
            tensor_img = transforms.ToTensor()(img)
    
            
            axs[0, i].imshow(np.uint8(tensor_img.permute(1,2,0)*255.))
            axs[0, i].set_axis_off()

        for i in range(len(poison_indices)):
            axs[1, i].set_title('Poison')
            
            img_path = self.img_paths[poison_indices[i]]
            img = self.loader(img_path)
            img = np.array(img)
            tensor_img = transforms.ToTensor()(img)
            tensor_img = self.warp_image(tensor_img)
            
            axs[1, i].imshow(np.uint8(tensor_img.squeeze(0).permute(1,2,0)*255.))
            axs[1, i].set_axis_off()

        fig.savefig(pathname + "/Samples_images.png")


class WaNetTinyimagenet(WaNetImagenet200):
    
    def __init__(self, root, k=64,
                 noise_ratio=0,
                 grid_rescale=1,
                 s=0.5, 
                train=True,
                poison_ratio=0.1, 
                target=0, 
                asr_calc=False,
                partition='None',
                imagenet_data_type='tiny'):
        
        
        super().__init__(root=root, k=k,
                     noise_ratio=noise_ratio,
                     grid_rescale=grid_rescale,
                     s=s, 
                    train=train,
                    poison_ratio=poison_ratio, 
                    target=target, 
                    asr_calc=asr_calc,
                    partition=partition,
                    imagenet_data_type=imagenet_data_type)
