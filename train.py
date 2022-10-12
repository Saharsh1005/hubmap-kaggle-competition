# -*- coding: utf-8 -*-

# ## ** Kaggle Setup for offline notebook run **
# !cp -r ../input/pytorch-segmentation-models-lib/ ./
# !pip config set global.disable-pip-version-check true
# !pip install -q ./pytorch-segmentation-models-lib/timm-0.4.12-py3-none-any.whl
# !cp -r ../input/einops-041-wheel/ ./
# !pip config set global.disable-pip-version-check true
# !pip install -q ../input/einops-041-wheel/einops-0.4.1-py3-none-any.whl
# !pip install /kaggle/input/staintools-offline/spams-2.6.5.4-cp37-cp37m-linux_x86_64.whl

# ## ** Imports **
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
import sys
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

c_  = Fore.GREEN
lo_  = Fore.RED
yel_ = Fore.BLUE
sr_ = Style.RESET_ALL

warnings.filterwarnings("ignore")
plt.rcParams["savefig.bbox"] = 'tight'

# ## **Import local modules**

from mit import *
from pvt_v2 import *
from coat import *
from daformer import *
from vahadane_stain_normalization import stain_normalize

print('import ok\n')

# ## **Train config**

class CFG:
    def __init__(self,n_fold = 5,seed = 42,batch_size = 4,debug=False,kaggle = False):
        
        # Directories
        self.kaggle     = kaggle
        self.root_dir   = "../input/hubmap-organ-segmentation" if kaggle else './data'
        self.IMAGES     = '../input/hubmap-stain-norm-original/train_images_stain_normalised' if kaggle else './data/images_stain_normalized'
        self.MASKS      = '../input/hubmap-hpa-2022-png-dataset/train_masks_png' if kaggle else './data/masks'
        
        self.n_fold        = n_fold
        self.seed          = seed
        # step2: data
        self.batch_size    = batch_size
        self.train_bs      = batch_size
        self.valid_bs      = batch_size*2
        self.debug         = debug
        self.img_size      = [768, 768] 
        self.num_workers   = 0
        
        # step3: model  
        
        # ##  a) CoaT
        self.model_parameters_list = dotdict(module='model_coat_daformer',param={'encoder': coat_parallel_small_plus1(), 'decoder': daformer_conv1x1,},
                                            checkpoint='../input/coat-pretrained-models/coat_small_7479cf9b.pth' if kaggle else './pretrained_weights/coat_small_7479cf9b.pth')
        #  ## b) Pvt_v2
        # self.model_parameters_list = dotdict(module='model_pvt_v2_daformer',param={'encoder': pvt_v2_b3(), 'decoder': daformer_conv3x3, 'encoder_cfg': {'img_size': self.img_size[0]}},
                                              # checkpoint='../input/coat-pretrained-models/pvt_v2_b3.pth' if kaggle else './pretrained_weights/pvt_v2_b3.pth')
                
        #  ## c) Segformer
        # self.model_parameters_list = dotdict(module='model_mit_segformer',param={'encoder': mit_b2(), 'decoder': daformer_conv1x1,},
        #                                      checkpoint='../input/coat-pretrained-models/mit_b2.pth' if kaggle else './pretrained_weights/mit_b2.pth')
        
        self.device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        self.backbone      = self.model_parameters_list.param['encoder'].__class__.__name__
        self.model_name    = self.model_parameters_list.module
        self.exp_name      = f'Exp__{self.model_name}__Training'
        
        # step4: optimizer
        self.epochs        = 60
        self.lr            = 5e-5 
        self.lr_drop      = 8
        self.optimizer     = 'AdamW'
        self.weight_decay  = 1e-5
        
        # step5: scheduler
        self.scheduler     = "ReduceLROnPlateau" # ReduceLROnPlateau OR CosineAnnealingLR
        self.min_lr        = 1e-6 # 0.00005
        self.T_max         = self.epochs + 50 #int(280/self.batch_size*self.epochs)+50
#         self.T_0           = 25
        self.warmup_epochs = 0
        self.wd            = 1e-5
        self.n_accumulate  = 1 # max(1, 32//self.batch_size)       
       

        
    def display(self):
        print(f"Experiment     {self.exp_name}")
        print(f"Debug is       {self.debug}") 
        print(f"Device is      {self.device}")
        print(f"Model is       {self.model_name}_{self.backbone}")
        print(f'Total folds    {self.n_fold}')
        print(f'-----------------------------')
        print(f"Batch size is  {self.batch_size}")
        print(f"img_size is    {self.img_size}")        
        print(f"epochs is      {self.epochs}")
        print(f"Optimiser is   {self.optimizer}")
        print(f"Scheduler is   {self.scheduler}")
        print(f"LR is          {self.lr}")
        print(f'-----------------------------')


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

# ## **Model**

import importlib

def build_model(model_parameters):    
    
    M = importlib.import_module(model_parameters.module)
    model = M.Net()
    model.output_type = ["loss","inference"]
    model.is_train = True
    encoder    =  model_parameters.param['encoder']
    path       =  model_parameters.checkpoint
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    if 'coat' in encoder.__class__.__name__:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
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


# ## **Mix-Upsampling**

import torch.nn.functional as Fx

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

            x = self.mixing *Fx.interpolate(x, size = originalshape,mode='bilinear', align_corners=False) \
                + (1-self.mixing )*Fx.interpolate(x, size = originalshape, mode='nearest')

            return x        

# ## **Util functions**

def create_folds(cfg = None):
    
    train_csv = pd.read_csv(os.path.join(cfg.root_dir,"train.csv"))
    train_csv.set_index('id',inplace = True)
    
    images_path = os.listdir(cfg.IMAGES)
    images_path = sorted(images_path)
    masks_path =  os.listdir(cfg.MASKS)
    masks_path = sorted(masks_path)
    
    ids  = [filename[:-4] for filename in masks_path]

    organs      = [train_csv['organ'][int(idx)] for idx in ids]
    img_height  = [train_csv['img_height'][int(idx)] for idx in ids]
    img_width   = [train_csv['img_width'][int(idx)] for idx in ids]
    data_source = [train_csv['data_source'][int(idx)] for idx in ids] 
    rle_mask    = [train_csv['rle'][int(idx)] for idx in ids] 
    images_path = [os.path.join(cfg.IMAGES,f) for f in images_path]
    masks_path  = [os.path.join(cfg.MASKS,f) for f in masks_path] 
    

    maping = {
        'id': ids,
        'organ':organs,
        'image_path':images_path,
        'mask_path': masks_path,
        'data_source': data_source,
#         'rle': rle_mask,
        'img_height': img_height,
        'img_width': img_width,

    }
    df = pd.DataFrame.from_dict(maping)    
    skf = KFold(n_splits=cfg.n_fold, shuffle=True,random_state=cfg.seed)

    df.loc[:,'fold']=-1
    for f,(t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
        df.iloc[v_idx,-1]=f    

    return df

def prepare_loaders(fold,df,cfg, debug=False):
    
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    
    if debug==True:
        train_df = train_df.head(4*5)
        train_df =  train_df.iloc[:50, :] # CHECKING TIME
        valid_df = valid_df.head(4*5)
    
    
    train_dataset = HuBMAPDataset(train_df, transforms=True,cfg=cfg,mode = 'train') 
    valid_dataset = HuBMAPDataset(valid_df, transforms=False,cfg=cfg, mode = 'train') 
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_bs if not cfg.debug else 20, 
                            num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_bs if not cfg.debug else 20, 
                            num_workers=0, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


def get_optimizer(cfg,optimizer_name= 'Adam'):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.lr)
#         optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
    return optimizer


def get_scheduler(cfg,optimizer,df):
    
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.T_max,eta_min=cfg.min_lr)
        
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau( optimizer  = optimizer, verbose=True, factor=0.7,mode="min",patience=5, threshold=0.001,min_lr =cfg.min_lr)
        
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer,cfg.lr_drop ,gamma=0.1)
        
    else:
        scheduler = None
        
    return scheduler

# ## ** Augmentations **

def do_random_flip(image, mask):
    if np.random.rand()>0.5:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)
    if np.random.rand()>0.5:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)
    if np.random.rand()>0.5:
        image = image.transpose(1,0,2)
        mask = mask.transpose(1,0)
    
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask

def do_random_rot90(image, mask):
    r = np.random.choice([
        0,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
    ])
    if r==0:
        return image, mask
    else:
        image = cv2.rotate(image, r)
        mask = cv2.rotate(mask, r)
        return image, mask
    
def do_random_contast(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1,1)*mag
    image = image * alpha
    image = np.clip(image,0,1)
    return image, mask

def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
    image = (image*255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)  # hue
    s = hsv[:, :, 1].astype(np.float32)  # saturation
    v = hsv[:, :, 2].astype(np.float32)  # value
    h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
    s =  s*(1 + random.uniform(-1,1)*mag[1])
    v =  v*(1 + random.uniform(-1,1)*mag[2])

    hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = image.astype(np.float32)/255
    return image, mask

def do_random_noise(image, mask, mag=0.1):
    height, width = image.shape[:2]
    noise = np.random.uniform(-1,1, (height, width,1))*mag
    image = image + noise
    image = np.clip(image,0,1)
    return image, mask

def do_random_rotate_scale(image, mask, angle=30, scale=[0.8,1.2] ):
    angle = np.random.uniform(-angle, angle)
    scale = np.random.uniform(*scale) if scale is not None else 1
    
    height, width = image.shape[:2]
    center = (height // 2, width // 2)
    
    transform = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask  = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask

#----------------------Below combined aug functions--------------------

def valid_augment5(image, mask):
  
    return image, mask

def train_augment5b(image, mask):
    
    more_transform = A.Compose([        
         A.OneOf([ 
             A.ElasticTransform(p=1.0, alpha=image.shape[1]*3, sigma=image.shape[1] * 0.07, alpha_affine=image.shape[1] * 0.09),
             A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
              ], p=0.25), 
#          A.CoarseDropout(max_holes=8, max_height=image.shape[0]//20, max_width=image.shape[1]//20,
#                          min_holes=5, fill_value=0, mask_fill_value=0, p=0.25),
        ], p=1.0)

    
    
    image, mask = do_random_flip(image, mask)
    image, mask = do_random_rot90(image, mask)      
    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_noise(image, mask, mag=0.1),
        lambda image, mask: do_random_contast(image, mask, mag=0.40),
        lambda image, mask: do_random_hsv(image, mask, mag=[0.40, 0.40, 0]) # Remove and try
    ], 2): image, mask = fn(image, mask)

    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_rotate_scale(image, mask, angle=45, scale=[0.50, 2.0]),# scale=[0.50, 2.0] or 0.7,1.5 or 0.50,1.5
    ], 1): image, mask = fn(image, mask)

#     augmented = more_transform(image= (image*255).astype(np.uint8), mask=mask)
#     image = augmented['image'].astype(np.float32)/255
#     mask = augmented['mask']
        
    return image, mask


# ## ** Utils 2 **

class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)


def read_tiff(image_file, mode='rgb'):
    # !pip install imagecodecs if facing COMPRESSION.lZW error
	image = tifffile.imread(image_file)
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


# ## ** Dataset **
class HuBMAPDataset(Dataset):
    def __init__(self, df= None, cfg = None, transforms = None, mode='train'):
        self.df = df
        self.mode = mode
        self.cfg = cfg
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        try: 
            img_path   = self.df.loc[index, 'image_path'] 
            organ      = self.df.loc[index, 'organ']
            mask_path  = self.df.loc[index,'mask_path'] 
            
            # Read image
            image      = read_tiff(img_path)
            image_size = cfg.img_size[0]
            
            if self.mode=='train':
                
                # Read mask
                mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
                mask  = (mask/255).astype(np.uint8)
                
                # Resize
                image = cv2.resize(image,dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)
                mask  = cv2.resize(mask, dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)
                
                # Augemnt
                if self.transforms:                   
                    image, mask = train_augment5b(image, mask)
                else:
                    image, mask = valid_augment5(image, mask)
               
                # -------------- Sanity check------------------------
#                 plt.imshow(image)
                # print('Data class read image,mask: img,mask',image.shape,mask.shape,np.unique(mask))
                # ---------------------------------------------------
                image = np.transpose(image, (2, 0, 1)) # image.shape = (3,H,W)
                mask = np.expand_dims(mask, axis=0)    # mask.shape = (1,H,W)
                
                image = image.astype(np.float32)
                mask  = mask.astype(np.float32)

                return torch.tensor(image), torch.tensor(mask) , organ

            else:    # resize for infer
                if self.transforms:
                    data = self.transforms(image=img)
                    img  = data['image']

                img = np.transpose(img, (2, 0, 1))   #(c, h, w)      
                return torch.tensor(img), img_height, img_width,id_,organs,sours
            
        except:
            print(index,"error occured in dataloader/dataset")
            return None


# ## ** Loss **
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# ## ** Metrics **
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=1.):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

# ## ** Train epoch **

'''
FLOW OF TRAINING:

1. Load batch of data (add to gpu if accessible)
2. forward pass
   |_ output = model (input)
   |_ loss   = criterion (output,target)
   
3. backward pass + optimize (only in training)
   |_ loss.backward()
   |_ optimizer.step()
   
4. Calculate statistics
   |_ running_loss += loss.item()*batch_size

return epoch_loss = running_loss/len(dataloader)
'''

def train_one_epoch(cfg,model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    running_dice_loss = 0.0
    criterion = DiceLoss() 

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks, organs) in pbar:  
#         loss     = 0
#         aux_loss = 0
#         bce_loss = 0
        
        # get a batch of inputs
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        batch_size = images.size(0)
        
        send_batch = {
            'image': images,
            'mask': masks,
            'organ': organs,
        }
        
        # ------------- forward ---------------
        with amp.autocast(enabled=True):
            
            output    = model(send_batch)   
            y_pred    = output['probability']
            loss      = output['bce_loss'].mean()
            loss1     = output['aux2_loss'].mean()
            dice_loss = criterion(y_pred,masks)

        scaler.scale(loss+0.2*loss1).backward() #/cfg.n_accumulate
    
        if ((step+1)%cfg.n_accumulate==0 or (step+1)==len(dataloader)):
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()
            
            if cfg.scheduler == 'CosineAnnealingLR' and cfg.scheduler: # 'CosineAnnealingLR' or 'StepLR' location
                scheduler.step()
       
        # statistics
        running_loss += (loss.item() * batch_size) 
        running_dice_loss += (dice_loss.item()*batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_dice_loss = running_dice_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_BCE_loss=f'{epoch_loss:0.4f}',dice_loss=f'{epoch_dice_loss:0.4f}', lr=f'{current_lr:0.6f}',gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

# ## ** Val epoch **

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_dice_loss = 0.0
    
    criterion = DiceLoss()    
    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks, organs) in pbar:       
        
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)
        
        send_batch = {
            'image': images,
            'mask': masks,
            'organ': organs,
        }
        output   = model(send_batch)   
        y_pred = output['probability']      

        loss = output['bce_loss'].mean()
        dice_loss = criterion(y_pred,masks)

        
        running_loss += (loss.item() * batch_size)
        running_dice_loss += (dice_loss.item()*batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_dice_loss = running_dice_loss / dataset_size

        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_BCE_loss=f'{epoch_loss:0.4f}',dice_loss=f'{epoch_dice_loss:0.4f}',lr=f'{current_lr:0.6f}',gpu_memory=f'{mem:0.2f} GB')
    
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores

# ** Training **

import copy
def run_training(cfg,fold, model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
  
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(cfg,model, optimizer, scheduler,dataloader=train_loader,device=cfg.device, epoch=epoch)        
        val_loss, val_scores = valid_one_epoch(model, valid_loader,device=cfg.device, epoch=epoch)        
        val_dice, val_jaccard = val_scores
        
        #ReduceLR scheduler
        if cfg.scheduler =='ReduceLROnPlateau': 
#             print('not none scheduler...')
            scheduler.step(val_loss) # Monitor metric
            
        
        history['Fold'].append(fold)
        history['Epoch'].append(epoch)
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        pd.DataFrame(history).to_csv(f'./{cfg.model_name}_fold_{fold}_{cfg.img_size[0]}.csv', index=False)
        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        print(f'{lo_}Valid Loss: {val_loss:0.4f} | Train Loss: {train_loss:0.4f}{sr_}')
        # deep copy the model
        if val_dice >= best_dice:
            
            print(f"{c_}Valid Dice Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f}){sr_}")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            
            
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{cfg.model_name}_fold_{fold}_{cfg.img_size[0]}-best_epoch-{best_epoch}.bin"          
            
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
        else:    
            last_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{cfg.model_name}_fold_{fold}_{cfg.img_size[0]}-last_epoch.bin" 
            torch.save(model.state_dict(), PATH)
            
        print(); print(f'{sr_}')

    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_dice))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def check_data_loader(cfg =None, fold = 0, df = None):
    
    train_loader, valid_loader = prepare_loaders(fold=fold,df = df,cfg=cfg)
    print(f'\n---DataLoader info:---')
    for i in range(4):
        batch = next(iter(train_loader))
        _ = random.randint(0, cfg.batch_size-1)
        images,labels,organs = batch
        image = images[_]
        label = labels[_].squeeze()
        organ = organs[_]
        
        plt.figure(figsize=(8,8))
        plt.subplot(1, 2, 1)
        plt.imshow((image.permute(1,2,0)), interpolation='none'); plt.title(f'Image ({organ})')
        plt.subplot(1, 2, 2)
        plt.imshow(image.permute(1,2,0), 'gray', interpolation='none');plt.title('Overlay')
        plt.imshow(label, 'jet', interpolation='none', alpha=0.7)
        plt.show()
 
    del images,labels, batch,organs
    torch.cuda.empty_cache()
    print(f'---DataLoader Ok!---\n')

# ## ** MAIN **
  
cfg                        = initialize_config(debug=False,batch_size=4)
df                         = create_folds(cfg=cfg)
check_data_loader(cfg = cfg, df = df)

for fold in range(cfg.n_fold):
    print(f'#'*15*2)
    print(f'### Fold: {fold}')
    print(f'#'*15*2)
    
    train_loader, valid_loader = prepare_loaders(fold=fold,df = df,cfg=cfg, debug = False)
    model                      = build_model(cfg.model_parameters_list).to(cfg.device) 
    optimizer                  = get_optimizer(cfg,optimizer_name=cfg.optimizer)
    scheduler                  = get_scheduler(cfg,optimizer,df)
    model, history             = run_training(cfg,fold, model, optimizer, scheduler,device=cfg.device,num_epochs=cfg.epochs)
    
    #Plot 1
    #plt.subplot(3, 1, 1)
    plt.plot(history['Epoch'], history['Train Loss'], 'r--')
    plt.plot(history['Epoch'], history['Valid Loss'], 'b-')
    plt.legend(['Training Loss', 'Valid Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{cfg.model_name}_fold_{fold}_Epoch_vs_Loss')
    plt.savefig(f'./{cfg.model_name}_fold_{fold}_Epoch_vs_Loss.png')
    plt.show()
    
    #Plot 2
    #plt.subplot(3,1, 2)
    plt.plot(history['Epoch'], history['Valid Dice'])
    plt.xlabel('Epoch')
    plt.ylabel('Valid Dice')
    plt.title(f'{cfg.model_name}_fold_{fold}_Epoch_vs_Valid-Dice')
    plt.savefig(f'./{cfg.model_name}_fold_{fold}_Epoch_vs_Valid-Dice.png')
    plt.show()
    
    #Plot 3
    #plt.subplot(3,1, 3)
    plt.plot(history['Valid Dice'], history['Train Loss'] )
    plt.xlabel('Valid Dice') 
    plt.ylabel('Train Loss')
    plt.title(f'{cfg.model_name}_fold_{fold}_Valid-Dice_vs_Loss')
    plt.savefig(f'./{cfg.model_name}_fold_{fold}_Valid-Dice_vs_Loss.png')
    plt.show()


