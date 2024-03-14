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

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from resnet import resnet18

from train_utils import *
from dataloader_ffcv import create_dataloader
from write_dataset_ffcv import write_dataset

from tqdm import tqdm
import time
from plot_utils import plot_SPC
from torch.cuda.amp import GradScaler, autocast


def main(args, pathname, device):
    
    start_time = time.time()
   
    train_loader, test_clean_loader, test_poison_loader = create_dataloader(args, args.batch_size,  pathname, device, partition='None')
    model = build_model(args)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    if args.dataset == 'cifar10':
        decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    elif  args.dataset == "imagenet200":
        optimizer = create_optimizer(model, args)
        scheduler = None
    elif  args.dataset == "tinyimagenet":            
        decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    
    
    if args.arch == 'tiny_vit':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    scaler = GradScaler()
    
    train_no = 50000
        

    for epoch in range(args.epochs):

        acc = train(train_loader, model, criterion, optimizer,scheduler, scaler,  epoch, args, device)
        test_clean_acc = validate(test_clean_loader, model, criterion, args, device)
        #import pdb;pdb.set_trace()
        test_poison_acc = validate(test_poison_loader, model, criterion, args, device)
        
        print(f'Test Clean Acc: {test_clean_acc} | Test Poison Acc: {test_poison_acc}')
        
        if args.arch == 'tiny_vit':
            scheduler.step()
        else:
            if args.dataset == 'cifar10':
                scheduler.step()
            if args.dataset == 'tinyimagenet':
                scheduler.step()

    torch.save(model, pathname + "/model.pt")
    with open(f'{pathname}/ACC', 'w') as f:
                    json.dump(test_clean_acc, f, indent=2)
    with open(f'{pathname}/ASR', 'w') as f:
                    json.dump(test_poison_acc, f, indent=2)
    
        
    tot_time = time.time() - start_time
    print(tot_time)
    print(test_clean_acc)
    print(test_poison_acc)


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, args, device):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()
    
    if args.arch != 'tiny_vit':
        if args.dataset == "imagenet200":
            lr_start, lr_end = get_lr(epoch, args.lr, args.epochs, args.lr_peak_epoch), get_lr(epoch + 1, args.lr, args.epochs, args.lr_peak_epoch)
            iters = len(train_loader)
            lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

    #if args.dataset == "tinyimagenet":
    #    lr_start, lr_end = get_lr(epoch, args.lr, args.epochs, args.lr_peak_epoch), get_lr(epoch + 1, args.lr, args.epochs, args.lr_peak_epoch)
    #    iters = len(train_loader)
    #    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])


    #start = time.time()
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (image, target, _, _) in iterator:
        
        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, args, one_epoch_step=len(train_loader))
            
        if args.arch != 'tiny_vit':    
            if args.dataset == 'imagenet200':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrs[i]


        image = image/255.
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            output_clean = model(image)
            loss = criterion(output_clean, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        iterator.set_description(f"Epoch {epoch} | LR {optimizer.param_groups[0]['lr']:.2f}") ## FIND LR!!!!!
        iterator.set_postfix(loss=loss.item(), accuracy=prec1.item())
        iterator.refresh()
        
    save_metrics(losses.avg, pathname, "Training Loss")

 


def get_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

def validate(val_loader, model, criterion, args, device):
    """
    Run evaluation
    """
    #import pdb;pdb.set_trace()

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target, _, _) in enumerate(val_loader):
        
        image = image/255.
        #target = target.to(device)

        # compute output
        with torch.no_grad():
            with autocast():
              output = model(image)
              loss = criterion(output, target)

        #import pdb;pdb.set_trace()
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))


    return top1.avg


def create_optimizer(model, args):

    # Only do weight decay on non-batchnorm parameters
    all_params = list(model.named_parameters())
    bn_params = [v for k, v in all_params if ('bn' in k)]
    other_params = [v for k, v in all_params if not ('bn' in k)]
    param_groups = [{
        'params': bn_params,
        'weight_decay': 0.
    }, {
        'params': other_params,
        'weight_decay': args.weight_decay
    }]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)
    
    return optimizer

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    ##################################### general setting #################################################
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--arch', type=str, default='res18', help='model architecture') ## vit_tiny
    parser.add_argument('--expname', type=str, default='Results', help='Experiment Number')

    ##################################### training setting #################################################
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--poison_ratio', type=float, help='Poison Ratio')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
    parser.add_argument('--lr_peak_epoch', default=2, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')
    parser.add_argument('--attack', type=str, help='Give attack name')
    parser.add_argument('--save_samples', type=str, default='True', help='Give attack name')
    parser.add_argument('--target', default=1, type=int, help= 'Target label')

    opt = parser.parse_args()
    
    if opt.target == 1:
        pathname = create_dir_path(opt.expname, opt.dataset, opt.attack, opt.poison_ratio, opt.arch)
    else:
        attack_name = f'{opt.attack}_{opt.target}'
        pathname = create_dir_path(opt.expname, opt.dataset, attack_name, opt.poison_ratio, opt.arch)

    save_args(pathname,opt)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
        
    main(opt, pathname, device)





