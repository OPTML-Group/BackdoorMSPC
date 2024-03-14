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

import pdb



# def l_SKL_diff(images, model, kl_loss, scale, both=False):
   
#     y_pred = F.log_softmax(model(scale*images), dim=1)
#     y_true = F.softmax(model(images), dim=1)
    
#     loss = torch.sum( (kl_loss(y_pred, y_true))  ,dim=1) 
    
#     if both == True:
        
#         y_pred2 = F.log_softmax(model(images), dim=1)
#         y_true2 = F.softmax(model(scale*images), dim=1)
        
#         loss +=  torch.sum( (kl_loss(y_pred2, y_true2))  ,dim=1)
   
#     return loss
   


def l_MSPC_tau(images, model, scales, tau, device):
    images = images/255.
    #import pdb;pdb.set_trace()
    images_mask = images-tau

    l_SPC = torch.zeros(len(images)).to(device)
    with torch.no_grad():
        with autocast():
            base_pred = torch.argmax(model(images), dim=1) # batchsize x 10
            for scale in scales:
                scale_pred = torch.argmax(model(torch.clamp(scale*images_mask,0,1)), dim=1)

                l_SPC = l_SPC + (2*(scale_pred == base_pred) - 1)


    return l_SPC / len(scales)





def SF_loss(output, output_scale, fdiv_name):
    
    kl_loss = nn.KLDivLoss(reduction="none", log_target=True)  
    ## kl_loss(logQ, logP) = P (logP - logQ) 
    
    P = F.softmax(output, dim=1)
    Q = F.softmax(output_scale, dim=1)
    M = (P+Q)/2
    
    log_P = torch.log(P)
    log_Q = torch.log(Q)
    log_M = torch.log(M)
    
    
    ### KL loss ########
    if fdiv_name == 'KL':
        loss = torch.sum( (kl_loss(log_Q, log_P))  ,dim=1) 
        
    elif fdiv_name == 'JSD':
        jsd_loss = (kl_loss(log_M, log_P) + kl_loss(log_M, log_Q))/2
        loss = torch.sum(jsd_loss  ,dim=1)     

    return loss


def l_MSPC(images, model, scales, device, tau, mask=None):
    images = images/255.
    if mask is not None:
        images_mask = (images-tau)*mask
    else:
        images_mask = images
        
    l_SPC = torch.zeros(len(images)).to(device)
    with torch.no_grad():
        with autocast():
            base_pred = torch.argmax(model(images), dim=1) # batchsize x 10
            # base_pred = torch.argmax(model(images+torch.rand(images.shape).to(device)), dim=1)
            for scale in scales:
                scale_pred = torch.argmax(model(torch.clamp(scale*images_mask,0,1)), dim=1)         
                l_SPC = l_SPC + (2*(scale_pred == base_pred) - 1)
                #else:
                #    l_SPC = l_SPC - 1

            
    return l_SPC / len(scales)

def l_SPC(images, model, scales, device):
    images = images/255.

    l_SPC = torch.zeros(len(images)).to(device)
    with torch.no_grad():
        with autocast():
            base_pred = torch.argmax(model(images), dim=1) # batchsize x 10
            for scale in scales:
                scale_pred = torch.argmax(model(torch.clamp(scale*images,0,1)), dim=1)

                #import pdb;pdb.set_trace()


                l_SPC = l_SPC + (scale_pred == base_pred)
                #else:
                #    l_SPC = l_SPC - 1


    return l_SPC / len(scales)



def SF_loss_allscale(images, model, scales, fdiv_name , mask=None):
    
    loss = 0
    images = images/255.    
    if mask is not None:
        images_mask = images*mask
    else:
        images_mask = images
        
    for scale in scales:
        with torch.no_grad():
            with autocast():
                output = model(images)
                output_scale = model(torch.clamp(scale*images_mask,0,1))
                loss = loss + SF_loss(output, output_scale, fdiv_name)
        
    
    
    return loss




def plot_violin(l_mask, l_nomask, poison_label_full, name):

    fig, ax = plt.subplots()
    data = {'Loss Value': list(l_mask.numpy()) + list(l_nomask.numpy()) ,
            'Loss Name' : ['Loss_mask']*len(l_mask) + ['Loss_nomask']*len(l_nomask),
            'Poisoning' : list(poison_label_full.numpy()) + list(poison_label_full.numpy())
            }
            
    df = pd.DataFrame(data)
    
    sns.set(style="darkgrid")
    sns.violinplot(x="Loss Name", y="Loss Value", hue="Poisoning", data=df, palette="Pastel1")
    plt.savefig(name)
    
    
    
