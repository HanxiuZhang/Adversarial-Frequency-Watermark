import sys
sys.path.append('..')
from utils import *
sys.path.append('../attacks/')
from fgsm import *
from pgd import *
from torch import nn,norm
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
sys.path.append('../models/')
from alexnet import *
from densenet_201 import *
from mobilenet_v2 import *
from resnet_50 import *
from vgg_19 import *

def get_attack_method(atk_name: str='fgsm'):
    atk_dict = {'fgsm':fgsm_wm_opti,'pgd':pgd_wm_opti}
    return atk_dict[atk_name]


def attack_and_record(filename: str,model: nn.Module, imgs:datasets, wm_origin: Tensor,  # type: ignore
                        block_size: int=4, alpha: float=0.1, beta: float=10/255, atk_name: str='fgsm',
                        N: int=20, l1: float=0.01, l2:float=0.01, s_a:float=0.0005, s_b:float=0.0001,
                        beta_max: float=20/255, steps: int=10, eps:float=10/255):
    with open(filename,'a') as file:
        for i in tqdm(range(len(imgs)), desc='Processing'):  # type: ignore
            img = imgs[i][0].cuda()  # type: ignore
            img = addborder(img,block_size)
            wm = transforms.Resize(img.size()[-2:])(wm_origin)  # type: ignore
            pred_label = model(img.unsqueeze(0)).argmax().item()
            label = torch.tensor([pred_label]).cuda()
            atk_method = get_attack_method(atk_name)
            perd_img,wm_extracted,a_res,b_res = \
                atk_method(img,label,wm,model,alpha,beta,block_size,N,l1,l2,s_a,s_b,beta_max,steps,eps)
            res = model(perd_img.unsqueeze(0))
            perd_label = res.argmax().item()
            wm_l2_norm = norm(wm_extracted-wm).item()
            img_l2_norm = norm(perd_img-img).item()
            file.write('{},{},{},{},{},{},{},{}\n'.format(i,imgs[i][1],pred_label,perd_label,wm_l2_norm,img_l2_norm,a_res,b_res))  # type: ignore
def check_result(filename):
    cols = ['index','label','pred_label','perd_label','wm_l2','img_l2','alpha','beta']
    res = pd.read_csv(filename,names=cols,header=None)
    print('Fool Rate:{}'.format((res['perd_label'] != res['pred_label']).sum()/res['index'].count()))
    print('Watermark L2:{}'.format(res['wm_l2'].sum() / res['wm_l2'].count()))
    print('Image L2:{}'.format(res['img_l2'].sum() / res['img_l2'].count()))

if(__name__ == '__main__'):
    filename = "../atk_result_cifar/{}_{}_{}_optimize.txt".format('10191900','densenet201','fgsm')
    imgs = datasets.CIFAR10(root='/home/hancy/dataset/',train=False,transform=transforms.ToTensor(),download=True)  # type: ignore
    wm_origin = cv2.imread('../img/logo.jpg')
    wm_origin = cv2.cvtColor(wm_origin,cv2.COLOR_BGR2RGB)
    wm_origin = transforms.ToTensor()(wm_origin).cuda()  # type: ignore
    model = torch.load('/home/hancy/code/adv_wm/models/pts/cifar_densenet_201.pt').cuda().eval()
    attack_and_record(filename,model,imgs,wm_origin,atk_name='fgsm')  # type: ignore
    check_result(filename)
