# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:56:09 2022

@author: Saharsh
"""

# Setup for offline notebook run
# !cp -r ../input/pytorch-segmentation-models-lib/ ./
# !pip config set global.disable-pip-version-check true
# !pip install -q ./pytorch-segmentation-models-lib/timm-0.4.12-py3-none-any.whl
# !cp -r ../input/einops-041-wheel/ ./
# !pip config set global.disable-pip-version-check true
# !pip install -q ../input/einops-041-wheel/einops-0.4.1-py3-none-any.whl
# !pip install /kaggle/input/staintools-offline/spams-2.6.5.4-cp37-cp37m-linux_x86_64.whl

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import io,img_as_float
import cv2
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import time
import copy
from collections import defaultdict
import gc
from PIL import Image
import imagecodecs

import tifffile 
import gc
# from tqdm.notebook import tqdm # for notebook runs
from tqdm import tqdm 
from pathlib import Path
import glob
import copy
from timeit import default_timer as timer
import timm

#SMP
# import segmentation_models_pytorch as smp

# HUGGINGFACE 
import transformers 

# SWA
from torch.optim.swa_utils import AveragedModel, SWALR


# PYTORCH
import torch
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
from torchmetrics.functional import dice
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torch.nn import functional as nnF
from torch.cuda import amp

import warnings
from colorama import Fore, Back, Style
from tabulate import tabulate
import pandas as pd
import importlib

c_  = Fore.GREEN
lo_  = Fore.RED
yel_ = Fore.BLUE
sr_ = Style.RESET_ALL

warnings.filterwarnings("ignore")
plt.rcParams["savefig.bbox"] = 'tight'

print('import ok\n')

import sys
sys.path.append('../input/hubmap-modular') 

from coat import *
from daformer import *
from vahadane_stain_normalization import stain_normalize

class CFG:
    def __init__(self,n_fold = 5,seed = 42,batch_size = 4,debug=False,kaggle = False):
        
        # Directories
        self.kaggle     = kaggle
        self.root_dir   = "../input/hubmap-organ-segmentation" if kaggle else './data'
        # self.IMAGES     = '../input/hubmap-stain-norm-original/train_images_stain_normalised' if kaggle else './data/images_stain_normalized'
        # self.MASKS      = '../input/hubmap-hpa-2022-png-dataset/train_masks_png' if kaggle else './data/masks'
        
        self.stain_norm_path = '../input/stain-normalize-image/mosaic.png' if kaggle else './data/stain_target/mosaic.png'
        
        ## # a) CoaT
        self.model_parameters_list = [
            dotdict(module='model_coat_daformer',param={'encoder': coat_parallel_small_plus1, 'decoder': daformer_conv1x1,}
                    ,checkpoint= '../input/coat-07cv/CoAT_fold_0_768-best_epoch-83.bin' if kaggle else './weights/CoAT_fold_0_768-best_epoch-83.bin',
                    ),
        ]
        self.backbone      = self.model_parameters_list[0].param['encoder']
        self.model_name    = self.model_parameters_list[0].module
        self.exp_name      = f'Hubmap__{self.model_name}__Inference'
        
               
        self.seed          = seed
        self.debug         = debug
        self.image_size    = [768,768] # [1024,1024]
        self.device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        self.backbone      = 'coat-small-parallel'
        self.model_name    = "CoAT"    
             
        
        # step6: infer
        self.thr = {
            "Hubmap":{
                "kidney" : 0.4,
                "prostate":0.4,
                "largeintestine":0.4,
                "spleen":0.4,
                "lung":0.1,  
            },

             "HPA":{
                "kidney" : 0.5,
                "prostate":0.5,
                "largeintestine":0.5,
                "spleen":0.5,
                "lung":0.15,

             },}
    

    def display(self):
        print(f"{self.exp_name}")
        print(f"debug is {self.debug}")  
        print('\n')
        print(f"Infer size     {self.image_size}")
        print(f"Experiment     {self.model_name}_{self.backbone}")
        print("Models: ")
        print(tabulate(pd.DataFrame.from_dict(self.model_parameters_list), headers = 'keys', tablefmt = 'psql'))        
        print(f"Threshold: ")
        print(tabulate(pd.DataFrame.from_dict(self.thr), headers = 'keys', tablefmt = 'psql'))        
        print('\n')
        
    

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    print(f"Setting seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


def initialize_config(debug=False,batch_size=4):
    cfg = CFG(batch_size = batch_size,debug = debug)    
    set_seed(cfg.seed)
    cfg.display()
    return cfg

import importlib

def build_model(model_parameters):    
    
    M = importlib.import_module(model_parameters.module)
#     print(M)
    model = M.Net()
    model.output_type = ["loss","inference"]
    model.is_train = True
    encoder    =  model_parameters.param['encoder']
    path       =  model_parameters.checkpoint
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint#['model']
    encoder.load_state_dict(state_dict,strict=False)
    model.encoder = encoder
    model.drop_path_rate=0.1
    print('Loaded pretrained weights: ',encoder.pretrain)
        
    return model


def load_model(model_parameters):
    M = importlib.import_module(model_parameters.module)
    model = M.Net(**model_parameters.param)   
    model.load_state_dict(torch.load(model_parameters.checkpoint, map_location=lambda storage, loc: storage)) 
    model.output_type = ["inference"]
    model.is_train = False
    model.dropout = nn.Dropout(p = 0.0)
    print('Loaded model: ',model_parameters.checkpoint)
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model

class MixUpSample_2(nn.Module):
    '''
    Mixed upsampling combining bilinear and nearest interpolation.
    Convert the model prediction back to the original image size.
    '''
    def __init__( self, scale_factor=4, originalshape = True):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor
        self.originalshape = originalshape
        
    def forward(self, x, originalshape):
        
        if self.originalshape:

            x = self.mixing *nnF.interpolate(x, size = originalshape,mode='bilinear', align_corners=False) \
                + (1-self.mixing )*nnF.interpolate(x, size = originalshape, mode='nearest')

            return x


class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)


#--- helper ----------
def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)
	
	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)
	
	else:
		raise NotImplementedError

def image_show(name, image, type='bgr', resize=1):
	if type == 'rgb': image = np.ascontiguousarray(image[:,:,::-1])
	H,W = image.shape[0:2]
	
	cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  #WINDOW_NORMAL #WINDOW_GUI_EXPANDED
	cv2.imshow(name, image) #.astype(np.uint8))
	cv2.resizeWindow(name, round(resize*W), round(resize*H))

def read_tiff(image_file, mode='rgb'):
    
	image = tiff.imread(image_file)
	image = image.squeeze()
	if image.shape[0] == 3:
		image = image.transpose(1, 2, 0)
        
	if mode=='bgr':
		image = image[:,:,::-1]
    
	mx = np.max(image)
	image = image.astype(np.float32)
	if mx:
		image /= mx # scale image to [0, 1]
        
	image = np.ascontiguousarray(image)
	return image


# Ref: https://www.kaggle.com/bguberfain/memory-aware-rle-encoding (with transposed mask)

def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Ref: https://www.kaggle.com/code/paulorzp/rle-functions-run-lenght-encode-decode/script
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def read_json_as_list(json_file):
	with open(json_file) as f:
		j = json.load(f)
	return j

def mask_to_inner_contour(mask):
	mask = mask>0.5
	pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
	contour = mask & (
			(pad[1:-1,1:-1] != pad[:-2,1:-1]) \
			| (pad[1:-1,1:-1] != pad[2:,1:-1]) \
			| (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
			| (pad[1:-1,1:-1] != pad[1:-1,2:])
	)
	return contour


def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
	contour =  mask_to_inner_contour(mask)
	if thickness==1:
		image[contour] = color
	else:
		r = max(1,thickness//2)
		for y,x in np.stack(np.where(contour)).T:
			cv2.circle(image, (x,y), r, color, lineType=cv2.LINE_4 )
	return image

def result_to_overlay(image, mask, probability=None, **kwargs): 
	
	H,W,C= image.shape
	if mask is None:
		mask = np.zeros((H,W),np.float32)
	if probability is None:
		probability = np.zeros((H,W),np.float32)
		
	o1 = np.zeros((H,W,3),np.float32)
	o1[:,:,2] = mask
	o1[:,:,1] = probability
	
	o2 = image.copy()
	o2 = o2*0.5
	o2[:,:,1] += 0.5*probability
	o2 = draw_contour_overlay(o2, mask, color=(0,0,1), thickness=max(3,int(7*H/1500)))
	
	#---
	o2,image,o1 = [(m*255).astype(np.uint8) for m in [o2,image,o1]]
	if kwargs.get('dice_score',-1)>=0:
		draw_shadow_text(o2,'dice=%0.5f'%kwargs.get('dice_score'),(20,80),2.5,(255,255,255),5)
	if kwargs.get('d',None) is not None:
		d = kwargs.get('d')
		draw_shadow_text(o2,d['id'],(20,140),1.5,(255,255,255),3)
		draw_shadow_text(o2,d.organ+'(%s)'%(organ_meta[d.organ].ftu),(20,190),1.5,(255,255,255),3)
		draw_shadow_text(o2,'%0.1f um'%(d.pixel_size),(20,240),1.5,(255,255,255),3)
		s100 = int(100/d.pixel_size)
		sx,sy = W-s100-40,H-80
		cv2.rectangle(o2,(sx,sy),(sx+s100,sy+s100//2),(0,0,0),-1)
		draw_shadow_text(o2,'100um',(sx+8,sy+40),1,(255,255,255),2)
		pass

	#draw_shadow_text(image,'input',(5,15),0.6,(1,1,1),1)
	#draw_shadow_text(im_paste,'predict',(5,15),0.6,(1,1,1),1)

	overlay = np.hstack([o2,image,o1])
	return overlay

# LB - metric
# https://www.kaggle.com/competitions/hubmap-organ-segmentation/overview/supervised-ml-evaluation

def compute_dice_score(probability, mask):
	N = len(probability)
	p = probability.reshape(N,-1)
	t = mask.reshape(N,-1)
	
	p = p>0.5
	t = t>0.5
	uion = p.sum(-1) + t.sum(-1)
	overlap = (p*t).sum(-1)
	dice = 2*overlap/(uion+0.0001)
	return dice

def do_local_validation():
    print('\tlocal validation ...')
    
    submit_df = pd.read_csv('submission.csv').fillna('')
    submit_df = submit_df.sort_values('id')
    truth_df  = valid_df.sort_values('id')
    
    lb_score = []
    num = len(submit_df)
    for i in range(num):
        t_df = truth_df.iloc[i]
        p_df = submit_df.iloc[i]
        t = rle_decode(t_df.rle, t_df.img_height, t_df.img_width, 1)
        p = rle_decode(p_df.rle, t_df.img_height, t_df.img_width, 1)
        
        dice = 2*(t*p).sum()/(p.sum()+t.sum())
        lb_score.append(dice)
        
        if 0:
            overlay = result_to_overlay(p, t)
            image_show_norm('overlay', overlay, min=0, max=1, resize=0.10)
            cv2.waitKey(1)

    truth_df.loc[:,'lb_score']=lb_score
    for organ in ['all', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
        if organ != 'all':
            d = truth_df[truth_df.organ == organ]
        else:
            d = truth_df
        print('\t%f\t%s\t%f' % (len(d) / len(truth_df), organ, d.lb_score.mean()))
        
    

def image_to_tensor(image, mode='rgb'):
    if  mode=='bgr' :
        image = image[:,:,::-1]
    
    x = image.transpose(2,0,1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x)
    return x

def do_tta_batch(image, organ, organ_to_label):
    
    batch = { #<todo> multiscale????
        'image': torch.stack([
            image,
            torch.flip(image,dims=[1]),
            torch.flip(image,dims=[2]),
        ]),
        'organ': torch.Tensor(
            [[organ_to_label[organ]]]*3
        ).long()
    }
    return batch

def undo_tta_batch(probability):
    probability[0] = probability[0]
    probability[1] = torch.flip(probability[1],dims=[1])
    probability[2] = torch.flip(probability[2],dims=[2])
    probability = probability.mean(0, keepdims=True)
    probability = probability[0,0].float()
    return probability


def display_prediction_info(idx,data_src, organ_name, img_shape,TH):
    print('-----------------'*2)   
    
    print(f"ID                       {idx}")        
    print(f"Data Source              {data_src}")
    print(f"Organ                    {organ_name}")
    print(f"Image shape              {img_shape}")
    print(f"Threshold(TH)            {TH}")

    print('-----------------'*2)

def do_submit(cfg, valid_df, tiff_dir, submit_type, data_source,organ_to_label, organ, all_net): 
    print('\n************ submit_type  = %s *******************'%submit_type)
    mix_upsample = MixUpSample_2() 
    result       = []
    start_timer  = timer()
    
    for i,d in valid_df.iterrows():
        id = d['id']
        if (d['data_source'] in data_source) and (d['organ'] in organ):
            
            # Read image
            tiff_file = tiff_dir +'/%d.tiff'%id
            tiff = cv2.cvtColor(cv2.imread(tiff_file), cv2.COLOR_BGR2RGB)
            
            # Stain Normalisation
            tiff = stain_normalize(tiff,cv2.cvtColor(cv2.imread(cfg.stain_norm_path), cv2.COLOR_BGR2RGB))
            tiff = tiff.astype(np.float32)/255
    
            image = cv2.resize(tiff,dsize=cfg.image_size,interpolation=cv2.INTER_LINEAR) # (768,768) or (864,864) or (960,960)
            image = image_to_tensor(image, 'rgb')
            batch = { k:v.to(cfg.device) for k,v in do_tta_batch(image, d.organ, organ_to_label).items() } 

            use = 0
            probability = 0
            with torch.no_grad():
                with amp.autocast(enabled = True):
                    
                    for n in all_net:
                            use += 1
                            # Model prediction
                            output = n(batch)
                            probability += \
                                mix_upsample(output['probability'], originalshape=(d.img_height,d.img_width)) # Mix Upsampling

                       
                    probability = undo_tta_batch(probability/use)
            #---
            probability = probability.data.cpu().numpy()

            p = probability > cfg.thr[d.data_source][d.organ]
            p = p.astype(bool)
            rle = rle_encode(p)
            
             # Display inference info:    
            display_prediction_info(id,d.data_source, d.organ, d.img_height,cfg.thr[d.data_source][d.organ])
    
        else:
            rle = ''
        
        if 0: # debug
            image = cv2.cvtColor(tiff, 4).astype(np.float32)/255 
            mask  = rle_decode(d.rle, d.img_height, d.img_width, 1) 
            overlay = result_to_overlay(image, mask, probability)            
            image_show('overlay',overlay, resize=0.25)
            cv2.waitKey(0)
            pass
        
        result.append({ 'id':id, 'rle':rle, })
        print('\r', '\tsubmit ... %3d/%3d %s'%(i, len(valid_df), time_to_str(timer() - start_timer,'sec')), end='',flush=True)
    print('\n')
    
    #---
    submit_df = pd.DataFrame(result)
    submit_df.to_csv('submission.csv',index=False)
    print(submit_df)
    print('\tSubmission file OK!')
    print('')
    
    if submit_type  == 'local-cv':
        do_local_validation()
        
    if submit_type == 'local-test':
        import matplotlib.pyplot as plt 
        m = tiff
        p = probability
        
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1); plt.imshow(m); plt.axis('OFF'); plt.title('Image')
        plt.subplot(1, 3, 2);plt.title('Prediction');plt.axis('OFF');plt.imshow(p, cmap = 'gray')
        plt.subplot(1, 3, 3); plt.imshow(m); plt.imshow(p, alpha=0.4); plt.title('Overlay')
        plt.tight_layout()
        plt.show()

               
def main():
    cfg         = initialize_config()
    result      = []
    submit_type = 'local-test' # local-test or kaggle

     
    # For Local CV and inferencing of result
    organ_meta = dotdict(
        kidney = dotdict(
            label = 1,
            um    = 0.5000,
            ftu   ='glomeruli',
        ),
        prostate = dotdict(
            label = 2,
            um    = 6.2630,
            ftu   ='glandular acinus',
        ),
        largeintestine = dotdict(
            label = 3,
            um    = 0.2290,
            ftu   ='crypt',
        ),
        spleen = dotdict(
            label = 4,
            um    = 0.4945,
            ftu   ='white pulp',
        ),
        lung = dotdict(
            label = 5,
            um    = 0.7562,
            ftu   ='alveolus',
        ),
    )
    organ_to_label = {k: organ_meta[k].label for k in organ_meta.keys()}
    label_to_organ = {v:k for k,v in organ_to_label.items()}
    num_organ=5
    #['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']


    if (submit_type == 'local-test') or (submit_type == 'kaggle'):
        valid_file = '../input/hubmap-organ-segmentation/test.csv' if cfg.kaggle else './data/test.csv'
        tiff_dir   = '../input/hubmap-organ-segmentation/test_images' if cfg.kaggle else './data/test_images'

    valid_df = pd.read_csv(valid_file)
    valid_df.loc[:,'img_area']=valid_df['img_height']*valid_df['img_width'] # sort by biggest image first for memory debug
    valid_df = valid_df.sort_values('img_area').reset_index(drop=True)
    print('Loaded valid_df OK')



    all_net = []
    for m in cfg.model_parameters_list:
        model = load_model(m)
        all_net.append(model)

    data_source =['Hubmap', 'HPA']
    organ = ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']

    do_submit(cfg, valid_df, tiff_dir, submit_type, data_source,organ_to_label, organ, all_net)

if __name__ == '__main__':
    main()

