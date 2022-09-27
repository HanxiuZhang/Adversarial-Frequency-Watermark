from torchattacks import PGD
import torch
from torch import Tensor
from torch import nn
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *

def pgd_direct(img: Tensor, label: Tensor, wm: Tensor, 
            model: nn.Module, alpha:float, beta: float,
            eps: float=0.3, steps: int=10,
            block_size: int=8 ) -> Tensor:
    wmed_img = embed_wm(img,wm,alpha,block_size)
    loss = nn.CrossEntropyLoss()
    adv_img = wmed_img.unsqueeze(0)
    for _ in range(steps):
        adv_img.requires_grad = True
        outputs = model(adv_img)
        cost = loss(outputs,label)
        grad = torch.autograd.grad(cost,adv_img,retain_graph=False,create_graph=False)[0]
        adv_image = adv_img.detach() + beta*grad.sign()
        delta = torch.clamp(adv_image - wmed_img, min=-eps, max=eps)
        adv_image = torch.clamp(wmed_img+delta,min=0,max=1).detach()
    adv_image = adv_image.squeeze(0)
    return adv_image


def pgd_wm_opti(img: Tensor, label: Tensor, wm: Tensor, 
            model: nn.Module, alpha:float, beta: float,
            N: int, l1: float, l2: float, s_a: float, s_b: float,
            eps: float=0.3, steps: int=10, block_size: int=8, ) -> Tensor:
    wmed_img = embed_wm(img,wm,alpha,block_size)
    loss = nn.CrossEntropyLoss()
    adv_img = wmed_img.unsqueeze(0)
    wm_perd = wm
    for _ in range(steps):
        adv_img.requires_grad = True
        outputs = model(adv_img)
        cost = loss(outputs,label)
        grad = torch.autograd.grad(cost,adv_img,retain_graph=False,create_graph=False)[0]
        per = grad.sign().squeeze(0)
        dct_per = dct_tensor(per, block_size)
        per_on_wm = (beta/alpha) * dct_per
        wm_perd = (wm_perd + per_on_wm)
        delta = torch.clamp(wm_perd - wm, min=-eps, max=eps)
        adv_image = adv_img.detach() + beta*grad.sign()
        delta = torch.clamp(adv_image - wmed_img, min=-eps, max=eps)
        adv_image = torch.clamp(wmed_img+delta,min=0,max=1).detach()
    '''
    calculate once
    then
    nest pgd calculation in each optimal step
    '''

    adv_image = adv_image.squeeze(0)