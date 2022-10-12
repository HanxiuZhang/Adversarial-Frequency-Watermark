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

# need to return alpha/extracted watermark!!!
def fgsm_wm_opti(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, 
                alpha:float, beta: float,block_size: int, 
                N: int, l1: float, l2: float, s_a: float, s_b: float, beta_max: float) -> Tensor:
    alpha_in = alpha
    beta_in = beta
    # Transfer the perturbation to the watermark
    wmed_img = embed_wm(img,wm,alpha,block_size)
    loss = nn.CrossEntropyLoss()
    wmed_img = wmed_img.unsqueeze(0)
    wmed_img.requires_grad = True
    outputs = model(wmed_img)
    cost = loss(outputs,label)
    grad = torch.autograd.grad(cost,wmed_img,retain_graph=False,create_graph=False)[0]
    per = grad.sign().squeeze(0)
    dct_per = dct_tensor(per,block_size)
    per_on_wm = (beta/alpha) * dct_per
    wm_perd = (wm + per_on_wm).clip(0,1)
    wmed = embed_wm(img,wm_perd,alpha,block_size)
    # If attack unsuccessfully after transfer, change beta to the maximum
    if(check_out(model,wmed,label)):
        # alpha = alpha_max
        beta = beta_max
        per_on_wm = (beta/alpha) * dct_per
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
        alpha_new = alpha_update(alpha,beta,l1,l2,s_a,idct_wm,per,dct_per)
        beta_new = beta_update(alpha,beta,l1,l2,s_b,idct_wm,per,dct_per)
        wm_perturbed = (wm + (beta_new/alpha_new)*dct_per).clip(0,1)
        wmed = embed_wm(img,wm_perturbed,alpha_new)
        if check_out(model,wmed,label):
            wm_extracted = extract_wm(img,wmed_res,alpha,block_size)
            return wmed_res, wm_extracted, alpha, beta
        else:
            alpha = alpha_new
            beta = beta_new
            wmed_res = wmed
            # print('alpha:{},beta:{}'.format(alpha,beta))
    wm_extracted = extract_wm(img,wmed_res,alpha,block_size)
    return wmed_res, wm_extracted, alpha, beta
    
