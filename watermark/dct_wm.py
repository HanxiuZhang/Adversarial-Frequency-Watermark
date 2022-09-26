import torch
import cv2
import numpy as np
import sys
sys.path.append('..')
from utils import *
def block_dct(bk,block_size=8):
    img_dct_blocks_h = bk.shape[0] // block_size
    img_dct_blocks_w = bk.shape[1] // block_size
    img_dct = np.zeros(shape = (bk.shape[0],bk.shape[1]))
# with tqdm(total=img_dct_blocks_h*img_dct_blocks_w) as pbar:
#     pbar.set_description('DCT Processing')
    for h in range(img_dct_blocks_h):
        for w in range(img_dct_blocks_w):
            a_block = bk[h*block_size:(h+1)*block_size,w*block_size:(w+1)*block_size]
            img_dct[h*block_size:(h+1)*block_size,w*block_size:(w+1)*block_size] =\
            cv2.dct(a_block)
            # pbar.update(1)
    return torch.from_numpy(img_dct).cuda()

def block_idct(bk,block_size=8):
    img_dct_blocks_h = bk.shape[0] // block_size
    img_dct_blocks_w = bk.shape[1] // block_size
    img_idct = np.zeros(shape = (bk.shape[0],bk.shape[1]))
# with tqdm(total=img_dct_blocks_h*img_dct_blocks_w) as pbar:
#     pbar.set_description('IDCT Processing')
    for h in range(img_dct_blocks_h):
        for w in range(img_dct_blocks_w):
            a_block = bk[h*block_size:(h+1)*block_size,w*block_size:(w+1)*block_size]
            img_idct[h*block_size:(h+1)*block_size,w*block_size:(w+1)*block_size] =\
            cv2.idct(a_block)
            # pbar.update(1)
    return torch.from_numpy(img_idct).cuda()

def dct_tensor(img,block_size=8):
    img = img.cpu().numpy()
    if img.ndim == 2:
        return block_dct(img,block_size)
    elif img.shape[0] == 1:
        return block_dct(img[0,...],block_size)
    else:
        return torch.stack((block_dct(img[0,...],block_size),block_dct(img[1,...],block_size),block_dct(img[2,...],block_size)),dim=0)

def idct_tensor(img,block_size=8):
    img = img.cpu().numpy()
    if img.ndim == 2:
        return block_idct(img,block_size)
    elif img.shape[0] == 1:
        return block_idct(img[0,...],block_size)
    else:
        return torch.stack((block_idct(img[0,...],block_size),block_idct(img[1,...],block_size),block_idct(img[2,...],block_size)),dim=0)

def embed_wm(img,wm,alpha,block_size=8):
    img_dct = dct_tensor(img,block_size)
    img_dct_wm = img_dct + alpha*wm
    img_wm_idct = idct_tensor(img_dct_wm,block_size)
    wmed_img = img_wm_idct.clip(0,1)
    wmed_img = wmed_img.float()
    return wmed_img

def extract_wm(img,wmed_img,alpha,block_size=8):
    img_dct = dct_tensor(img,block_size)
    wmed_dct = dct_tensor(wmed_img,block_size)
    wm = (wmed_dct-img_dct)/alpha
    return wm