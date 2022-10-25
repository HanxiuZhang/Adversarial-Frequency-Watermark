import torch
from torch import Tensor
from torch import nn
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *

def ifgsm_pipeline(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module, alpha:float, beta: float,block_size: int=8, steps: int=10, eps: float=10/255, **args) -> Tensor:
    r'''
    Pipline method to 1) embed digital watermark 2) add perturbation in PGD(iterative-FGSM) way
    '''
    # Embed watermark
    wmed_img = embed_wm(img,wm,alpha,block_size)
    # Calculate perturbation
    adv_img = wmed_img.detach().unsqueeze(0)
    images = adv_img.detach()
    loss = nn.CrossEntropyLoss()  # type: ignore
    for _ in range(steps):
        adv_img.requires_grad = True
        outputs = model(adv_img)
        # Calculate loss
        cost = loss(outputs, label)
        # Update adversarial image
        grad = torch.autograd.grad(cost, adv_img,
                                    retain_graph=False, create_graph=False)[0]
        adv_img = adv_img.detach() + beta*grad.sign()
        delta = torch.clamp(adv_img - images, min=-eps, max=eps)
        adv_img = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_img.detach().squeeze(0)

def ifgsm_wm_opti(img: Tensor, label: Tensor, wm: Tensor, model: nn.Module,   # type: ignore
                alpha:float, beta: float,block_size: int, 
                N: int, l1: float, l2: float, s_a: float, s_b: float, beta_max: float,
                steps: int, eps:float,**args):
    r'''
    Proposed method to 1) embed watermark 2) calculate perturbation 3) transfer to watermark 4) re-embed perturbated watermark
    in PGD(iterative-FGSM) way
    '''
    # Embed watermark
    wmed_img = embed_wm(img,wm,alpha,block_size)
    # Calculate perturbation
    adv_img = wmed_img.detach().unsqueeze(0)
    images = adv_img.detach()
    delta = torch.zeros_like(adv_img)
    loss = nn.CrossEntropyLoss()
    for _ in range(steps):
        adv_img.requires_grad = True
        outputs = model(adv_img)
        # Calculate loss
        cost = loss(outputs, label)
        # Update adversarial image
        grad = torch.autograd.grad(cost, adv_img,
                                    retain_graph=False, create_graph=False)[0]
        adv_img = adv_img.detach() + beta*grad.sign()
        delta = torch.clamp(adv_img - images, min=-eps, max=eps)
        adv_img = torch.clamp(images + delta, min=0, max=1).detach()
    # To be consistent with single-step attack, 
    # 1) calculate raw perturbation 
    # 2) use beta*raw as target perturbation and transfer it to watermark
    per = delta.detach().squeeze(0) / beta
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