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
        adv_image = adv_img.detch() + beta*grad.sign()
        delta = torch.clamp(adv_image - wmed_img, min=-eps, max=eps)
        adv_image = torch.clamp(adv_image+delta,min=0,max=1).detach()
    adv_image = adv_image.squeeze(0)
    return adv_image
