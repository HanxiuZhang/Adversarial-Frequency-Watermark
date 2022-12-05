from typing import Tuple
from torch import Tensor
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import ReplicationPad2d as pad, Module, functional as F
import math
import pytorch_msssim

sys.path.append('../watermark/')
from dct_wm import *

def addborder(img:Tensor, block_size: int=8) -> Tensor:
    r'''
    Add border to an image so that it can be appropriately divided into blocks
    '''
    diff_x = img.size()[2] % block_size
    diff_y = img.size()[1] % block_size
    if (diff_x != 0):
        img = pad((block_size-diff_x,0,0,0))(img)
    if (diff_y != 0):
        img = pad((0,0,block_size-diff_y,0))(img)
    return img

def pltshow(img: Tensor, gray: bool=False) -> None:
    r'''
    Show Tensor as image
    '''
    img = transforms.ToPILImage()(img)  # type: ignore
    plt.figure(figsize=(5,5))
    plt.axis('off')
    if(gray):
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)

def check_predict(model: Module,img: Tensor) -> Tuple[int, float]: 
    img = torch.unsqueeze(img,0).type(torch.FloatTensor).cuda()    # type: ignore
    out = (F.softmax(model(img),dim=1))
    predict = out.argmax().item()
    prob = out.squeeze(0)[predict].item()  # type: ignore
    return predict, prob  # type: ignore

def saveTensor(img: Tensor, filename: str):
    transforms.ToPILImage()(img).save(filename)

def read_img_and_tensor(img_path: str, wm_path: str, block_size: int) -> Tuple[Tensor,Tensor]:
    T = transforms.ToTensor()
    img = cv2.imread('../img/beagle2.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = addborder(T(img),block_size).cuda()
    wm = cv2.imread('../img/logo.jpg')
    wm = cv2.cvtColor(wm,cv2.COLOR_BGR2RGB)
    wm = transforms.Resize(img.size()[-2:])(T(wm)).cuda()
    return img, wm

def psnr(img: Tensor, perd_img: Tensor) -> float:
    mse = torch.mean((perd_img-img)**2).item()
    psnr = 10*math.log10(1/mse)
    return psnr

def ssim(img: Tensor, perd_img: Tensor) -> float:
    img = torch.unsqueeze(img,0).type(torch.FloatTensor).cuda()
    perd_img = torch.unsqueeze(perd_img,0).type(torch.FloatTensor).cuda()
    return pytorch_msssim.ssim(img, perd_img).item()