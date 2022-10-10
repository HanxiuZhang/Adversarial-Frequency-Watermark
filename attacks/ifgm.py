import torch
from torch import Tensor,nn,norm
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *

def ifgm_direct(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, alpha:float, beta: float,block_size: int=8, steps: int=10) -> Tensor:
    wmed_img = embed_wm(img,wm,alpha,block_size)
    adv_img = wmed_img.clone()
    loss = nn.CrossEntropyLoss()
    adv_img = adv_img.unsqueeze(0)
    adv_img.requires_grad = True
    for _ in range(steps):
        outputs = model(adv_img)
        cost = loss(outputs,label)
        grad = torch.autograd.grad(cost,adv_img,retain_graph=False,create_graph=False)[0]
        per = grad/norm(grad)
        adv_img = adv_img + beta*per
        adv_img = torch.clamp(adv_img,min=0,max=1)
    return adv_img.squeeze(0)