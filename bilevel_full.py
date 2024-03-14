import os
import pdb
import time
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
import math

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from resnet import resnet18

from train_utils import *
from dataloader_ffcv import create_dataloader

from tqdm import tqdm
import time
from plot_utils import plot_SPC
from torch.cuda.amp import GradScaler, autocast

import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import pdb
from pathlib import Path
import json
from bilevel_losses import *

def save_all(l_MSPC, poison_label_full, pathname, epoch):
    
    torch.save(l_MSPC, f'{pathname}/l_MSPC_{epoch}.pt')
    roc_auc = roc_auc_score(poison_label_full, l_MSPC)
    print(roc_auc)
    with open(f'{pathname}/AUROC_{epoch}', 'w') as f:
                    json.dump(roc_auc, f, indent=2)
    fpr, tpr, _ = roc_curve(poison_label_full, l_MSPC)
    torch.save(fpr, f'{pathname}/FPR_{epoch}.pt')
    torch.save(tpr, f'{pathname}/TPR_{epoch}.pt')
    

def main(args, device):

    start_time = time.time()
      
    # prepare dataset
    train_loader, _,_ = create_dataloader(args, args.batch_size, '', device, partition='None', seq=True)
        
    if args.target == 1:
        model_path = f'Results/{args.dataset}/{args.attack}/Poisonratio_{args.poison_ratio}/{args.arch}/Trial {args.trialno}'
    else:
        model_path = f'Results/{args.dataset}/{args.attack}_{args.target}/Poisonratio_{args.poison_ratio}/{args.arch}/Trial {args.trialno}'

    pathname = f'{model_path}/Bilevel/{args.tau}'


    if not os.path.exists(pathname):
        Path(pathname).mkdir(parents=True)

    print(pathname)

    save_args(pathname,args)
    model = torch.load(f'{model_path}/model.pt')
    model.to(device)
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    
    if args.dataset == 'cifar10':
        train_no = 50000
    elif args.dataset == 'imagenet200':
        train_no = 100000
    elif args.dataset == 'tinyimagenet':
        train_no = 100000
    

    scales = list(map(int, args.scales.split(',')))
    tau = args.tau
    
    ###################### Create mask and w ######################
    img, _,_,_ = next(iter(train_loader))
    mask = torch.ones(img[0,0,:,:].shape, requires_grad=True , device=device)
    mask_lambda = args.masklam
    
    w = torch.ones(train_no).to(device)
    w.requires_grad = False
    #mask.requires_grad_()
        
    ############################################ True poison labels ############################################
    poison_label_full = torch.zeros(train_no)
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    for ix, (images, _, _, poison_label) in iterator:
        batch_size = len(images)
        poison_label_full[ix*batch_size:ix*batch_size + batch_size] = poison_label
    
    torch.save(poison_label_full , f'{pathname}/poisonlab_true.pt')
    #############################################################################################################
    ####import pdb;pdb.set_trace()

    ################ Warmup Start of w ##############################################################
    l_MSPC_full_nomask = torch.zeros(train_no)
    ### Loop calculating losses =================================
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))    
    for ix, (images, _, _, poison_label) in iterator:
        batch_size = len(images)         
        #import pdb;pdb.set_trace()
        l_MSPC_full_nomask[ix*batch_size:ix*batch_size + batch_size] = l_MSPC_tau(images, model, scales, tau, device)

    plot_violin(l_MSPC_full_nomask, l_MSPC_full_nomask, poison_label_full, f'{pathname}/violin_l_MSPC_0.png')    
    
    w = w*0
    w[l_MSPC_full_nomask>0] = 1
    
    
    save_all(l_MSPC_full_nomask, poison_label_full, pathname, 0)

    #############################################################################################################
        
        
    # Optimizer and scaler
    optimizer_inner = torch.optim.SGD([mask], args.lr_inner, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler_inner = GradScaler()

            
    for outer_epoch in range(args.outer_epoch):
        #mask.requires_grad_()
        print(f'========================== Starting Inner level ==========================')
        fig, ax = plt.subplots(1,10, figsize=(50,5), constrained_layout=True)
        l_min = 1.0

        for i in range(args.epoch_inner):
            print(f'--------- Epoch {i+1} --------------')
            iterator = tqdm(enumerate(train_loader), total=len(train_loader))        
            
            losses = []
            for ix, (images, targets, _, _) in iterator:
                l_rec = 0
                
                batch_size = len(images)
                ### Reset gradients wrt mask to None
                optimizer_inner.zero_grad(set_to_none=True)
               
                images = images/255.
                
                for scale in scales:
                    ## Masked images
                    masked_poison_images = (images-tau)*mask #+ (1-mask)*torch.rand(images.shape).to(device)
                    
                    with autocast():
                        #
                        #  Output of image and scaled image
                        output = model(images)
                        output_scale = model(torch.clamp(scale*masked_poison_images,0,1))
                         
    
 
                        w_batch = w[ix*batch_size:ix*batch_size + batch_size]        
    
                        loss_fdiv = (1/batch_size)*torch.dot(w_batch, SF_loss(output, output_scale, args.fdiv_name))
                        loss_fdiv = loss_fdiv / len(scales)
                        loss_fdiv.backward()
                        
                        with torch.no_grad():
                            l_rec += loss_fdiv
                            
                    
                loss_l1  = mask_lambda*torch.norm(mask, p=1) 
                loss_l1.backward()
                
                with torch.no_grad():
                    l_rec += loss_l1
                
                losses.append(l_rec)
                save_metrics(l_rec, pathname, 'Losses_iteration')
                
            
                optimizer_inner.step()
                mask.data = torch.clamp(mask.data, min=0, max=1)               
                    
            print(f'Loss: {l_rec}  |  L1 loss: {loss_l1/mask_lambda}')
    
                
            ax.flatten()[i].imshow(mask.clone().detach().cpu().numpy())
            ax.flatten()[i].set_title(f'Epoch {i+1}')
        

        #mask = mask_min.clone().detach()
        #ask = mask_min
        #mask.data = mask_min.data



        fig.savefig(f'{pathname}/Masks_{outer_epoch+1}.pdf')
        torch.save(mask, f'{pathname}/mask_{outer_epoch+1}.pt')
        
        print('========================== Plotting L_MSPC Using Mask ==========================')    
        ##### modified SPC loss and f-div loss ################################################
        l_MSPC_full_mask = torch.zeros(train_no)
        l_MSPC_full_nomask = torch.zeros(train_no)
        ### Loop calculating losses =================================
        iterator = tqdm(enumerate(train_loader), total=len(train_loader))    
        for ix, (images, _, _, poison_label) in iterator:
            batch_size = len(images)             
            l_MSPC_full_mask[ix*batch_size:ix*batch_size + batch_size] = l_MSPC(images, model, scales, device, tau, mask=mask)
            l_MSPC_full_nomask[ix*batch_size:ix*batch_size + batch_size] = l_MSPC(images, model, scales, device,  tau, mask=None)
            
         
        #### Make dataframes and plot violins ===============================================
        #def plot_violin(loss_mask, loss_nomask, name)
    
        plot_violin(l_MSPC_full_mask, l_MSPC_full_nomask, poison_label_full, f'{pathname}/violin_l_MSPC_{outer_epoch+1}.png')    
        
        w = w*0
        w[l_MSPC_full_mask>0] = 1
        
        
        save_all(l_MSPC_full_mask, poison_label_full, pathname, outer_epoch+1)
    #################################  Complete  ################################################    

    

    
    ############################################ SPC Loss ############################################
    l_SPC_full = torch.zeros(train_no)
    ### Loop calculating losses =================================
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    for ix, (images, _, _, poison_label) in iterator:
        batch_size = len(images)
        l_SPC_full[ix*batch_size:ix*batch_size + batch_size] = l_SPC(images, model, scales, device)
    
    tot_time = time.time() - start_time
    print(f'Time taken = {tot_time}')

    torch.save(l_SPC_full, f'{pathname}/l_SPC.pt')
    roc_auc = roc_auc_score(poison_label_full, l_SPC_full)
    print(roc_auc)

    with open(f'{pathname}/AUROC_SPC', 'w') as f:
        json.dump(roc_auc, f, indent=2)
    fpr, tpr, _ = roc_curve(poison_label_full, l_SPC_full)
    
    torch.save(fpr, f'{pathname}/FPR_SPC.pt')
    torch.save(tpr, f'{pathname}/TPR_SPC.pt')
     
    ###################################################################################################
    


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    ##################################### general setting #################################################
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--arch', type=str, default='res18', help='model architecture')

    ##################################### training setting #################################################
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--poison_ratio', default=0.1, type=float, help='Poison Ratio')
    parser.add_argument('--lr_inner', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--attack', type=str, help='Give attack name')
    parser.add_argument('--fdiv_name', type=str, default='KL')
    parser.add_argument('--epoch_inner', type=int, default=10, help='batch size')
    parser.add_argument('--outer_epoch', type=int, default=4, help='batch size')
    parser.add_argument('--masklam', default=0.01, type=float)
    parser.add_argument('--scales', default='2,3,4,5,6,7,8,9,10,11,12')
    parser.add_argument('--tau', default=0.2, type=float)
    parser.add_argument('--trialno',  type=int)
    parser.add_argument('--save_samples', type=str, default='False', help='Give attack name') 
    parser.add_argument('--target', default=1, type=int, help= 'Target label')

    opt = parser.parse_args()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
        
    main(opt, device)
    

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            



