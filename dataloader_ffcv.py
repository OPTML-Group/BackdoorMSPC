# from dataset import PoisonedCIFAR10
from datasets import *
import torch 
import torchvision
from typing import List

from torchvision.transforms import Resize
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, NDArrayDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandomResizedCrop
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
import random

def create_dataloader(args, batch_size,  pathname, device, partition, seq=False):
    
    if args.dataset == "cifar10":

        DATASET = {
            'Badnet':{'class': BadnetCIFAR10, 'type': 'RGB'},
            'Blend': {'class': BlendCIFAR10, 'type': 'RGB'}, 
            'Wanet': {'class': WaNetCIFAR10, 'type': 'torch'}, 
            'CleanLabel': {'class': CleanLabelCIFAR10, 'type': 'torch'}, 
            'LabelConsistent' : {'class': LabelConsistentCIFAR10, 'type': 'RGB'},
            'Trojan' : {'class':TrojanCIFAR10, 'type': 'RGB'},
            'AdaptiveBlend': {'class': AdapBlendCIFAR10, 'type': 'RGB'},
            'DFST': {'class': DFSTCIFAR10, 'type': 'RGB'},
            }
        img_size = 32
        
    elif args.dataset == "imagenet200":
        
        DATASET = {
            'Badnet':{'class': BadnetImagenet200, 'type': 'RGB'},
            'Blend': {'class': BlendImagenet200, 'type': 'RGB'}, 
            }
        img_size = 224
        
    elif args.dataset == "tinyimagenet":
        
        DATASET = {
            'Badnet':{'class': BadnetTinyimagenet, 'type': 'RGB'},
            'Blend': {'class': BlendTinyimagenet, 'type': 'RGB'}, 
            'Wanet': {'class': WaNetTinyimagenet, 'type': 'torch'}, 
            }
        img_size = 64


    
    '''
    datasets = {
        'train': DATASET[args.attack]['class']('data', train=True, poison_ratio=args.poison_ratio, target=args.target, partition = partition), #upper_right=True
        'test_clean': DATASET[args.attack]['class']('data', train=False, poison_ratio=0, target=args.target),
        'test_poison': DATASET[args.attack]['class']('data', train=False, poison_ratio=1, target=args.target, asr_calc=True)
    }
    '''
    if args.attack == 'metasift':
        path = '../Meta-Sift/dataset/gtsrb_dataset.h5'
        datasets = {
        'train': DATASET[args.attack]['class'](path, train=True, poison_ratio=args.poison_ratio, target=args.target, partition = 'None'), #upper_right=True
        'test_clean': DATASET[args.attack]['class'](path, train=False, poison_ratio=0, target=args.target),
        'test_poison': DATASET[args.attack]['class'](path, train=False, poison_ratio=1, target=args.target, asr_calc=True)
        }
    else:
        datasets = {
        'train': DATASET[args.attack]['class']('data', train=True, poison_ratio=args.poison_ratio, target=args.target, partition = 'None'), #upper_right=True
        'test_clean': DATASET[args.attack]['class']('data', train=False, poison_ratio=0, target=args.target),
        'test_poison': DATASET[args.attack]['class']('data', train=False, poison_ratio=1, target=args.target, asr_calc=True)
    }

    
    if args.save_samples == 'True':
        datasets['train'].save_images(pathname)



    BATCH_SIZE = batch_size
    
    loaders = {}

    for name in ['train', 'test_clean', 'test_poison']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        if  DATASET[args.attack]['type'] == 'torch':
            image_pipeline: List[Operation] = [NDArrayDecoder()]
        elif DATASET[args.attack]['type'] == 'RGB':
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        
        
        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                #RandomHorizontalFlip(),
            ])
        
            
        if args.arch == 'tiny_vit':
            if  DATASET[args.attack]['type'] == 'RGB':
                image_pipeline.extend([RandomResizedCrop(scale=(1,1), ratio=(1,1), size=224)])
        
        
        image_pipeline.extend([ToTensor()])

        if args.arch == 'tiny_vit':
            if  DATASET[args.attack]['type'] == 'torch':
                image_pipeline.extend([Resize(size=224)])

        image_pipeline.extend([ToDevice(device, non_blocking=True)])
        

        if  DATASET[args.attack]['type'] == 'torch':
            image_pipeline.extend([Convert(torch.float16)])
        elif DATASET[args.attack]['type'] == 'RGB':
            image_pipeline.extend([ToTorchImage(),Convert(torch.float16)])

        #if args.arch == 'tiny_vit':
        #    image_pipeline.extend([RandomResizedCrop(scale=(1,1), ratio=(1,1), size=224)])
        
        
        if seq == True: 
            ORDER = OrderOption.SEQUENTIAL
        else:
            ORDER = OrderOption.RANDOM

        if args.target == 1:
            pathwriter = f'data/{args.dataset}_{args.attack}_{args.poison_ratio}_{name}.beton'
        else:
            pathwriter = f'data/{args.dataset}_{args.attack}_{args.target}_{args.poison_ratio}_{name}.beton'


        # Create loaders
        loaders[name] = Loader(pathwriter,
                                batch_size=BATCH_SIZE,
                                num_workers=8,
                                order=ORDER,  #OrderOption.SEQUENTIAL, #OrderOption.RANDOM
                                drop_last=(name == 'train'), #False, #(name == 'train'),
                                pipelines={'image': image_pipeline,
                                           'label': label_pipeline,
                                           'index' : label_pipeline,
                                           'poisonlabel': label_pipeline})
        
    return loaders['train'], loaders['test_clean'], loaders['test_poison']

