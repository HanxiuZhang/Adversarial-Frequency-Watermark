from torch import norm
import torch
import sys
import torch.nn.functional as F
sys.path.append('../watermark/')
from dct_wm import *
from tqdm import tqdm

def alpha_update(alpha,beta,l1,l2,step,idct_wm,per,dct_per):
    grad = l1*norm(alpha*idct_wm+beta*per)*norm(idct_wm)-l2*(beta/alpha**2)*norm(dct_per)
    alpha_new = alpha - step*grad
    if alpha_new < 0:
        alpha_new = alpha
    return alpha_new

def beta_update(alpha,beta,l1,l2,step,idct_wm,per,dct_per):
    grad = l1*norm(alpha*idct_wm+beta*per)*norm(per)+l2*(1/alpha)*norm(dct_per)
    beta_new = beta - step*grad
    if beta_new < 0:
        beta_new = beta
    return beta_new

def check_out(model,img,target):
    img = torch.unsqueeze(img,0).type(torch.FloatTensor).cuda()    # type: ignore
    out = (F.softmax(model(img),dim=1))
    out = out.argmax()
    return (out == target).item()