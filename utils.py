from torch import Tensor
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import ReplicationPad2d as pad
import math
sys.path.append('../attacks/')
from fgsm import *
from ifgsm import *
sys.path.append('../models/')
from alexnet import *
from densenet_201 import *
from mobilenet_v2 import *
from resnet_50 import *
from vgg_19 import *

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
    img = transforms.ToPILImage()(img)  # type: ignore
    plt.figure(figsize=(5,5))
    plt.axis('off')
    if(gray):
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)

def psnr(img: Tensor, perd_img: Tensor) -> float:
    mse = torch.mean((perd_img-img)**2).item()
    psnr = 10*math.log10(1/mse)
    return psnr

def get_attack_method(atk_name: str='fgsm'):
    atk_dict = {'fgsm':fgsm_direct,'ifgsm':ifgsm_pipeline,
                'fgsm_opt':fgsm_wm_opti,'ifgsm_opt':ifgsm_wm_opti}
    return atk_dict[atk_name]

def get_model(model_name: str='alexnet'):
    model_dict = {
        'vgg19': vgg19_IN(),
        'alexnet': alexnet_IN(),
        'resnet50': resnet50_IN(),
        'mobilenetv2': mobilenetv2_IN(),
        'densenet201': densenet201_IN()
    }
    return model_dict[model_name].eval().cuda()