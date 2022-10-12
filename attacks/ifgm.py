import torch
from torch import Tensor, nn, norm
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *

def ifgm_direct(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, alpha:float, beta: float,block_size: int=8, steps: int=10, eps: float=10/255) -> Tensor:
    r'''
    Pipline method to 1) embed digital watermark 2) add perturbation in iterative-FGM way
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
        per = (grad / norm(grad)).squeeze(0)
        adv_img = adv_img + beta*per
        adv_img = torch.clamp(adv_img,min=0,max=1)
    # Clip perturbation to an range to avoid image distortion
    delta = (adv_img - wmed_img).clip(-1*eps,eps).squeeze(0)
    res = (wmed_img + delta).clip(0,1)
    return res

def ifgm_wm_opti(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, 
                alpha:float, beta: float,block_size: int, steps: int, eps:float,
                N: int, l1: float, l2: float, s_a: float, s_b: float, beta_max: float) -> Tensor:
    r'''
    Proposed method to 1) embed watermark 2) calculate perturbation 3) transfer to watermark 4) re-embed perturbated watermark
    in iterative-FGM way
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
        per = (grad / norm(grad)).squeeze(0)
        adv_img = adv_img + beta*per
        adv_img = torch.clamp(adv_img,min=0,max=1)
    delta = (adv_img - wmed_img).clip(-1*eps,eps).squeeze(0)
    # To be consistent with single-step attack, 
    # 1) calculate raw perturbation 
    # 2) use beta*raw as target perturbation and transfer it to watermark
    per = delta / beta
    # Store initial values
    alpha_in = alpha
    beta_in = beta
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