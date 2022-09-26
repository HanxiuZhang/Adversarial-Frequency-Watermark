from torchattacks import FGSM
import torch
from torch import Tensor
from torch import nn
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *

def fgsm_direct(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, alpha:float, beta: float,block_size: int=8 ) -> Tensor:
    wmed_img = embed_wm(img,wm,alpha,block_size)
    loss = nn.CrossEntropyLoss()
    wmed_img = wmed_img.unsqueeze(0)
    wmed_img.requires_grad = True
    outputs = model(wmed_img)
    cost = loss(outputs,label)
    grad = torch.autograd.grad(cost,wmed_img,retain_graph=False,create_graph=False)[0]
    adv_image = wmed_img + beta*grad.sign()
    adv_image = torch.clamp(adv_image,min=0,max=1).squeeze(0)
    return adv_image

def fgsm_wm(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, alpha:float, beta: float,block_size: int=8 ) -> Tensor:
    wmed_img = embed_wm(img,wm,alpha,block_size)
    loss = nn.CrossEntropyLoss()
    wmed_img = wmed_img.unsqueeze(0)
    wmed_img.requires_grad = True
    outputs = model(wmed_img)
    cost = loss(outputs,label)
    grad = torch.autograd.grad(cost,wmed_img,retain_graph=False,create_graph=False)[0]
    per = grad.sign().squeeze(0)
    wm_perd = wm_add_per(img, wm, per, alpha, beta, block_size)
    adv_image = embed_wm(img,wm_perd,alpha,block_size)
    return wm_perd, adv_image
