import torch
from torch import Tensor
import math
import sys
sys.path.append('../attacks/')
from fgsm import *
from fgm import *
from ifgsm import *
sys.path.append('../models/')
from alexnet import *
from densenet_201 import *
from mobilenet_v2 import *
from resnet_50 import *
from vgg_19 import *

def psnr(img: Tensor, perd_img: Tensor) -> float:
    mse = torch.mean((perd_img-img)**2).item()
    psnr = 10*math.log10(1/mse)
    return psnr

def get_attack_method(atk_name: str='fgsm'):
    atk_dict = {'fgsm':fgsm_direct,'fgm':fgm_direct,'ifgsm':ifgsm_pipeline,
                'fgsm_opt':fgsm_wm_opti,'fgm_opt':fgm_wm_opti,'ifgsm_opt':ifgsm_wm_opti}
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