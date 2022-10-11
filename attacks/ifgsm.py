import torch
from torch import Tensor
from torch import nn
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *

def ifgsm_direct(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, alpha:float, beta: float,block_size: int=8, steps: int=10, eps: float=10/255) -> Tensor:
    r'''
    Pipline method to 1) embed digital watermark 2) add perturbation in iterative-FGSM way
    '''
    # Embed watermark
    wmed_img = embed_wm(img,wm,alpha,block_size)
    # Calculate perturbation
    adv_img = wmed_img.clone()
    loss = nn.CrossEntropyLoss()
    adv_img = adv_img.unsqueeze(0)
    adv_img.requires_grad = True
    for _ in range(steps):
        outputs = model(adv_img)
        cost = loss(outputs,label)
        grad = torch.autograd.grad(cost,adv_img,retain_graph=False,create_graph=False)[0]
        adv_img = adv_img + beta*grad.sign()
        adv_img = torch.clamp(adv_img,min=0,max=1)
    # Clip perturbation to an range to avoid image distortion
    delta = (adv_img - wmed_img).clip(-1*eps,eps).squeeze(0)
    res = (wmed_img + delta).clip(0,1)
    return res

def ifgsm_wm_opti(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, 
                alpha:float, beta: float,block_size: int, steps: int,
                N: int, l1: float, l2: float, s_a: float, s_b: float, alpha_max: float, beta_max: float) -> Tensor:
    r'''
    Proposed method to 1) embed watermark 2) calculate perturbation 3) transfer to watermark 4) re-embed perturbated watermark
    in iterative-FGSM way
    '''
    # Embed watermark
    wmed_img = embed_wm(img,wm,alpha,block_size)
    # Calculate perturbation
    adv_img = wmed_img.clone()
    loss = nn.CrossEntropyLoss()
    adv_img = adv_img.unsqueeze(0)
    adv_img.requires_grad = True
    for _ in range(steps):
        outputs = model(adv_img)
        cost = loss(outputs,label)
        grad = torch.autograd.grad(cost,adv_img,retain_graph=False,create_graph=False)[0]
        adv_img = adv_img + beta*grad.sign()
        adv_img = torch.clamp(adv_img,min=0,max=1)
    delta = (adv_img - wmed_img).clip(-1*beta,beta).squeeze(0)


    res = (wmed_img + delta).clip(0,1)
    return res