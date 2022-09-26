from torch import Tensor
import sys
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.append('../watermark/')
from dct_wm import *

# add perturbation to img through block-dct watermark
def wm_add_per(img: Tensor, wm: Tensor, per:Tensor, alpha: float, beta: float, block_size: int) -> Tensor:
    per_on_wm = (beta/alpha) * dct_tensor(per,block_size)
    wm_perd = (wm + per_on_wm).clip(0,1)
    return wm_perd

# add border to image so that it can be divided perfectly with block_size
def addborder(img,block_size=8):
    diff_x = img.shape[0] % block_size
    diff_y = img.shape[1] % block_size
    if (diff_x==0 and diff_y==0):
        return img
    img = cv2.copyMakeBorder(img,
              0,(block_size-diff_x),
              0,(block_size-diff_y),
              cv2.BORDER_REPLICATE)
    return img

def pltshow(img,gray=False):
    img = transforms.ToPILImage()(img)
    plt.figure(figsize=(2,2))
    plt.axis('off')
    if(gray):
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)