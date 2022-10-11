# from torchattacks import FGSM
import torch
from torch import Tensor
from torch import nn
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *
from torch import norm

def fgm_direct(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, alpha:float, beta: float,block_size: int=8 ) -> Tensor:
    r'''
    Direct pipline method to add perturbation on image with watermark in FGM way
    '''
    # Embed digital watermark on benign image
    wmed_img = embed_wm(img,wm,alpha,block_size)
    # Embed digital watermark on benign image
    loss = nn.CrossEntropyLoss()
    wmed_img = wmed_img.unsqueeze(0)
    wmed_img.requires_grad = True
    outputs = model(wmed_img)
    cost = loss(outputs,label)
    grad = torch.autograd.grad(cost,wmed_img,retain_graph=False,create_graph=False)[0]
    # Add FGM perturbation to image
    per = grad/norm(grad)
    adv_image = wmed_img + beta*per
    adv_image = torch.clamp(adv_image,min=0,max=1).squeeze(0)
    return adv_image

def fgm_wm_opti(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, 
                alpha:float, beta: float,block_size: int, 
                N: int, l1: float, l2: float, s_a: float, s_b: float, alpha_max: float, beta_max: float) -> Tensor:
    r'''
    Transfer the FGM perturbation to the watermark
    Optimize alpha and beta with gradient descent
    '''
    alpha_in = alpha
    beta_in = beta
    # Embed digital watermark on benign image
    wmed_img = embed_wm(img,wm,alpha,block_size)
    # Embed digital watermark on benign image
    wmed_img = wmed_img.unsqueeze(0)
    wmed_img.requires_grad = True
    outputs = model(wmed_img)
    loss = nn.CrossEntropyLoss()
    cost = loss(outputs,label)
    grad = torch.autograd.grad(cost,wmed_img,retain_graph=False,create_graph=False)[0]
    per = grad.squeeze(0)
    per = per/norm(per)
    # Transfer the perturbation to watermark image
    dct_per = dct_tensor(per,block_size)
    per_on_wm = (beta/alpha) * dct_per
    wm_perd = (wm + per_on_wm).clip(0,1)
    # Embed perturbed digital watermark to benign image
    wmed = embed_wm(img,wm_perd,alpha,block_size)
    # If attack unsuccessfully after transfer, change beta to the maximum
    if(check_out(model,wmed,label)):
        alpha = alpha_max
        beta = beta_max
        per_on_wm = (beta_max/alpha) * dct_per
        wm_perd = (wm + per_on_wm).clip(0,1)
        wmed = embed_wm(img,wm_perd,alpha,block_size)
        # If still unsuccessfully, recover the watermark as clean
        if(check_out(model,wmed,label)):
            wmed_res = wmed_img.clone().detach().squeeze(0)
            wm_extracted = extract_wm(img,wmed_res,alpha,block_size)
            return wmed_res,wm_extracted, alpha_in, beta_in
    idct_wm = idct_tensor(wm)   
    wmed_res = wmed
    for _ in range(N):
        # Update alpha and beta with gradient descent
        alpha_new = alpha_update(alpha,beta,l1,l2,s_a,idct_wm,per,dct_per)
        beta_new = beta_update(alpha,beta,l1,l2,s_b,idct_wm,per,dct_per)
        # Embed digital watermark with new alpha and beta value
        wm_perturbed = (wm + (beta_new/alpha_new)*dct_per).clip(0,1)
        wmed = embed_wm(img,wm_perturbed,alpha_new)
        # If attack unsuccessfully with new value, keep old value and return results
        if check_out(model,wmed,label):
            wm_extracted = extract_wm(img,wmed_res,alpha,block_size)
            return wmed_res,wm_extracted, alpha, beta
        else:
            # If attack successfully, update value results
            alpha = alpha_new
            beta = beta_new
            wmed_res = wmed
    # After N iteration steps, return current values
    wm_extracted = extract_wm(img,wmed_res,alpha,block_size)
    return wmed_res,wm_extracted, alpha, beta
    
