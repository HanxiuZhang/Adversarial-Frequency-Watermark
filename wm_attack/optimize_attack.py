import sys
sys.path.append('..')
from utils import *
from torch import nn,norm
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import cv2
import pandas as pd
from atk_utils import *



def attack_and_record(filename: str,model: nn.Module, imgs:datasets, wm_origin: Tensor,  # type: ignore
                        block_size: int=8, alpha: float=0.1, beta: float=10/255, atk_name: str='fgsm_opt',
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
                atk_method(img,label,wm,model,alpha,beta,block_size,N=N,l1=l1,l2=l2,s_a=s_a,s_b=s_b,beta_max=beta_max,steps=steps,eps=eps)
            res = model(perd_img.unsqueeze(0))
            perd_label = res.argmax().item()
            wm_psnr = psnr(wm,wm_extracted)
            img_psnr = psnr(img,perd_img)
            file.write('{},{},{},{},{},{},{},{}\n'.format(i,imgs[i][1],pred_label,perd_label,wm_psnr,img_psnr,a_res,b_res))  # type: ignore

def check_and_record_result(filename,record_filename):
    cols = ['index','label','pred_label','perd_label','wm_psnr','img_psnr','alpha','beta']
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
    # beta = 1
    # beta_max = 1.5
    beta = 2/255
    beta_max = 8/255
    N = 30
    l1 = 0.02
    l2 = 0.05
    s_a = 0.0005
    s_b = 0.0001 
    steps = 10
    eps = 8/255
    imgs = datasets.ImageFolder('/home/hancy/dataset/imagenet5000/',transform=transforms.ToTensor())   # type: ignore
    wm_origin = cv2.imread('../img/logo.jpg')
    wm_origin = cv2.cvtColor(wm_origin,cv2.COLOR_BGR2RGB)
    wm_origin = transforms.ToTensor()(wm_origin).cuda()  # type: ignore
    # for atk_name in ['fgsm_opt','fgm_opt','ifgsm_opt']:
    for atk_name in ['ifgsm_opt']:
        for model_name in ['vgg19','resnet50','alexnet','densenet201','mobilenetv2']:
            model = get_model(model_name)
            filename = "../res/model_{}_atk_{}_optimize_alpha_{}_beta_{}_N_{}_l1_{}_l2_{}_s_a_{}_s_b_{}_beta_max_{}.txt".format(model_name,atk_name,alpha,beta,N,l1,l2,s_a,s_b,beta_max)
            attack_and_record(filename,model,imgs,wm_origin,alpha=alpha,beta=beta,atk_name=atk_name,  # type: ignore
                               N=N,l1=l1,l2=l2,s_a=s_a,s_b=s_b,beta_max=beta_max,steps=steps,eps=eps )
            record_filename = '../res/optimize_result.txt'
            check_and_record_result(filename,record_filename)