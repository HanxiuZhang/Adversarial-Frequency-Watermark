import sys
from torchattacks import PGDL2
sys.path.append('..')
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
from torch import nn
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import cv2
import pandas as pd
from utils import *
from atk_utils import *

def attack_and_record(filename: str,model: nn.Module, imgs:datasets, wm_origin: Tensor,  # type: ignore
                        block_size: int=8, alpha: float=0.1, beta: float=10/255, atk_name: str='fgsm',eps: float = 8/255, steps: int=10):
    with open(filename,'a') as file:
        for i in tqdm(range(len(imgs)), desc='Processing'):  # type: ignore
            img = imgs[i][0].cuda()  # type: ignore
            img = addborder(img,block_size)
            wm = transforms.Resize(img.size()[-2:])(wm_origin)  # type: ignore
            pred_label = model(img.unsqueeze(0)).argmax().item()
            label = torch.tensor([pred_label]).cuda()
            atk_method = get_attack_method(atk_name)
            perd_img = atk_method(img,label,wm,model,alpha,beta,block_size,eps=eps,steps = steps)
            wm_extracted = extract_wm(img,perd_img,alpha,block_size)
            res = model(perd_img.unsqueeze(0))
            perd_label = res.argmax().item()
            wm_psnr = psnr(wm,wm_extracted)
            img_psnr = psnr(img,perd_img)
            file.write('{},{},{},{},{},{}\n'.format(i,imgs[i][1],pred_label,perd_label,wm_psnr,img_psnr))  # type: ignore



def check_and_record_result(filename,record_filename):
    cols = ['index','label','pred_label','perd_label','wm_psnr','img_psnr']
    res = pd.read_csv(filename,names=cols,header=None)
    fool_rate = (res['perd_label'] != res['pred_label']).sum()/res['index'].count()
    wm_psnr = res['wm_psnr'].sum() / res['wm_psnr'].count()
    img_psnr = res['img_psnr'].sum() / res['img_psnr'].count()
    print('Fool Rate:{}'.format(fool_rate))
    print('Watermark PSNR:{}'.format(wm_psnr))
    print('Image PSNR:{}'.format(img_psnr))
    with open(record_filename,'a') as file:
        file.write('{},{},{},{}\n'.format(filename,fool_rate,wm_psnr,img_psnr))


 
if(__name__ == '__main__'):
    alpha = 0.1
    beta = 1
    steps = 10
    eps = 8/255
    imgs = datasets.ImageFolder('/home/hancy/dataset/imagenet5000/',transform=transforms.ToTensor())   # type: ignore
    wm_origin = cv2.imread('../img/logo.jpg')
    wm_origin = cv2.cvtColor(wm_origin,cv2.COLOR_BGR2RGB)
    wm_origin = transforms.ToTensor()(wm_origin).cuda()  # type: ignore
    # for atk_name in ['fgsm','fgm','ifgsm']:
    for atk_name in ['fgm']:
        for model_name in ['vgg19','resnet50','alexnet','densenet201','mobilenetv2']:
            filename = "../res/model_{}_atk_{}_pipline_alpha_{}_beta_{}.txt".format(model_name,atk_name,alpha,beta)
            model = get_model(model_name)
            attack_and_record(filename,model,imgs,wm_origin,alpha=alpha,beta=beta,atk_name=atk_name,steps = steps,eps=eps)  # type: ignore
            record_filename = '../res/pipline_result.txt'
            check_and_record_result(filename,record_filename)
