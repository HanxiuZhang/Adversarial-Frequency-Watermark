from torchattacks import FGSM
import torch
from torch import Tensor
from torch import nn
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *

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
    return adv_image

def fgsm_wm_opti(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, 
                alpha:float, beta: float,block_size: int, 
                N: int, l1: float, l2: float, s_a: float, s_b: float) -> Tensor:
    wmed_img = embed_wm(img,wm,alpha,block_size)
    loss = nn.CrossEntropyLoss()
    wmed_img = wmed_img.unsqueeze(0)
    wmed_img.requires_grad = True
    outputs = model(wmed_img)
    cost = loss(outputs,label)
    grad = torch.autograd.grad(cost,wmed_img,retain_graph=False,create_graph=False)[0]
    per = grad.sign().squeeze(0)
    wm_perd = wm_add_per(img, wm, per, alpha, beta, block_size)
    wmed = embed_wm(img,wm_perd,alpha,block_size)
    idct_wm = idct_tensor(wm)
    dct_per = dct_tensor(per)
    # wm_res = wm_perd
    wmed_res = wmed
    for n in range(N):
        alpha_new = alpha_update(alpha,beta,l1,l2,s_a,idct_wm,per,dct_per)
        beta_new = beta_update(alpha,beta,l1,l2,s_b,idct_wm,per,dct_per)
        wm_perturbed = (wm + (beta_new/alpha_new)*dct_per).clip(0,1)
        wmed = embed_wm(img,wm_perturbed,alpha_new)
        if check_out(model,wmed,label):
            return wmed_res
        else:
            alpha = alpha_new
            beta = beta_new
            wmed_res = wmed
            # print('alpha:{},beta:{}'.format(alpha,beta))
    return wmed_res
    
