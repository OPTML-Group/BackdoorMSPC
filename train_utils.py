import copy 
import torch
import numpy as np 
import os
from pathlib import Path
import json
import timm
from resnet import resnet18


def build_model(args):
    
    if args.dataset == 'imagenet200':
        imagenet_class=True
        NUM_CLASSES = 200
        tiny=False
        gtsrb = False
    elif args.dataset == 'tinyimagenet':
        imagenet_class=False
        NUM_CLASSES = 200
        tiny=True
        gtsrb = False
    elif args.dataset == 'cifar10':
        imagenet_class=False
        NUM_CLASSES = 10
        tiny=False
        gtsrb = False
    else:
        raise ValueError("Datasets should be either imagenet200 or cifar10")
    
    if args.arch == 'res18':
        model = resnet18(num_classes=NUM_CLASSES, imagenet=imagenet_class, tiny=False, gtsrb = gtsrb) 
    elif args.arch == 'tiny_vit':
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    elif args.arch == 'res9_mod':
        model = torch.nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),
            Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            torch.nn.Linear(128, NUM_CLASSES, bias=False),
            Mul(0.2)
            )
        
    return model
        
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True)
    )

def create_dir_path(expnumber,dataset,attack,poisonratio,arch):
    
    par_dir_path = f"{expnumber}/{dataset}/{attack}/Poisonratio_{poisonratio}/{arch}"
    # par_dir_path = "Results/Exp_" + str(expnumber) + "/" + str(dataset) + "/" + str(modelname) 
        
    i = 1
    if not os.path.exists(str(par_dir_path) + "/Trial %s" % i):
        pathname = str(par_dir_path) + "/Trial " + str(i)
    else:
        while os.path.exists(str(par_dir_path) + "/Trial %s" % i):
            i += 1
        pathname = str(par_dir_path) + "/Trial " + str(i)
    
    Path(pathname).mkdir(parents=True)
    
    return pathname


def save_args(dir_path,args):
    filename = dir_path + '/Arguments_Hyperparamaters.txt'
    
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
        
def warmup_lr(epoch, step, optimizer, args, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #import pdb;pdb.set_trace()
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_metrics(metric, dir_path, metric_name):
    
    if torch.is_tensor(metric):
        metric = metric.detach().cpu().numpy()
    
    file = dir_path + "/" + metric_name
    if not os.path.exists(file):
        f = open(file, "x")
    f = open(file, "a")
    f.write("%s\n" % metric)
    f.close()
