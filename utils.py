from torch import Tensor
import sys
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import ReplicationPad2d as pad

sys.path.append('../watermark/')
from dct_wm import *

def addborder(img:Tensor, block_size: int=8) -> Tensor:
    r'''
    Add border to an image so that it can be appropriately divided into blocks
    '''
    diff_x = img.size()[2] % block_size
    diff_y = img.size()[1] % block_size
    if (diff_x != 0):
        img = pad((block_size-diff_x,0,0,0))(img)
    if (diff_y != 0):
        img = pad((0,0,block_size-diff_y,0))(img)
    return img

def pltshow(img: Tensor, gray: bool=False) -> None:
    r'''
    Show Tensor as image
    '''
    img = transforms.ToPILImage()(img)
    plt.figure(figsize=(5,5))
    plt.axis('off')
    if(gray):
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
