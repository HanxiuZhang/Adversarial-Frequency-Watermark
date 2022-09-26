from torch import norm
import torch
import sys
sys.path.append('../watermark/')
from dct_wm import *
from tqdm import tqdm

def alpha_update(alpha,beta,l1,l2,step,idct_wm,per,dct_per):
    grad = l1*norm(alpha*idct_wm+beta*per)*norm(idct_wm)-l2*(beta/alpha**2)*norm(dct_per)
    alpha_new = alpha - step*alpha
    if alpha_new < 0:
        alpha_new = alpha
    return alpha_new

def beta_update(alpha,beta,l1,l2,step,idct_wm,per,dct_per):
    grad = l1*norm(alpha*idct_wm+beta*per)*norm(per)+l2*(1/alpha)*norm(dct_per)
    beta_new = beta - step*beta
    if beta_new < 0:
        beta_new = beta
    return beta_new

def check_out(model,img,target):
    img = torch.unsqueeze(img,0).type(torch.FloatTensor).cuda()
    out = (F.softmax(model(img),dim=1))
    out = out.argmax()
    return (out == target).item()

def opti(img,wm,model,atk,target,N,alpha,beta,l1,l2,s_a,s_b):
    wmed_origin = embed_wm(img,wm,alpha)
    wmed_input = torch.unsqueeze(wmed_origin,0).type(torch.FloatTensor).cuda()
    adv_images = atk(wmed_input, target)
    adv_wmed_origin = adv_images[0,...]
    per = (adv_wmed_origin-wmed_origin)/beta
    per_dct = dct_tensor(per)
    wm_per = per_dct * (beta/alpha)
    wm_perturbed = (wm_per+wm).clip(0,1)
    wmed = embed_wm(img,wm_perturbed,alpha)
    idct_wm = idct_tensor(wm)
    dct_per = dct_tensor(per)
    wm_res = wm_perturbed
    wmed_res = wmed
    with tqdm(total=N) as pbar:
        pbar.set_description('Optimization Processing')
        for n in range(N):
            alpha_new = alpha_update(alpha,beta,l1,l2,s_a,idct_wm,per,dct_per)
            beta_new = beta_update(alpha,beta,l1,l2,s_b,idct_wm,per,dct_per)
            wm_perturbed = (wm + (beta_new/alpha_new)*dct_per).clip(0,1)
            wmed = embed_wm(img,wm_perturbed,alpha_new)
            if check_out(model,wmed,target):
                return wm_res,wmed_res
            else:
                alpha = alpha_new
                beta = beta_new
                wm_res = wm_perturbed
                wmed_res = wmed
            pbar.update(1)   
        return wm_res,wmed_res,alpha,beta