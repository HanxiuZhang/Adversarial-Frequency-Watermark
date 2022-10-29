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



def attack_and_record(filename: str,model_origin: nn.Module, model_list: list[nn.Module], imgs:datasets, wm_origin: Tensor,  # type: ignore
                        block_size: int=8, alpha: float=0.1, beta: float=10/255, atk_name: str='fgsm_opt',
                        N: int=20, l1: float=0.01, l2:float=0.01, s_a:float=0.0005, s_b:float=0.0001,
                        beta_max: float=20/255, steps: int=10, eps:float=10/255):
    with open(filename,'a') as file:
        for i in tqdm(range(len(imgs)), desc='Processing'):  # type: ignore
            img = imgs[i][0].cuda()  # type: ignore
            img = addborder(img,block_size)
            wm = transforms.Resize(img.size()[-2:])(wm_origin)  # type: ignore
            pred_label = model_origin(img.unsqueeze(0)).argmax().item()
            label = torch.tensor([pred_label]).cuda()
            atk_method = get_attack_method(atk_name)
            perd_img,_,_,_ = \
                atk_method(img,label,wm,model_origin,alpha,beta,block_size,N=N,l1=l1,l2=l2,s_a=s_a,s_b=s_b,beta_max=beta_max,steps=steps,eps=eps)
            res_list = []
            for model in model_list:
                res_list.append((model(img.unsqueeze(0))).argmax().item())  # type: ignore
                res_list.append((model(perd_img.unsqueeze(0))).argmax().item())  # type: ignore
            file.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,*res_list))  # type: ignore

def check_and_record_transfer(filename,record_filename):
    models_name = ['vgg19','resnet50','alexnet','densenet201','mobilenetv2']
    cols = ['index']
    for model_name in models_name:
        cols.append('{}_pred'.format(model_name))
        cols.append('{}_perd'.format(model_name))
    res = pd.read_csv(filename,names=cols,header=None)
    with open(record_filename,'a') as file:
        file.write('{}\n'.format(filename))
        for model_name in models_name:
            fool_rate = (res['{}_pred'.format(model_name)] != res['{}_perd'.format(model_name)]).sum()/res['index'].count()
            print('Fool Rate {}:{}'.format(model_name, fool_rate))
            file.write('{}:{},'.format(model_name,fool_rate))
        file.write('\n')

def check():
    record_filename = '../transfer_res/transfer_result.txt'
    for model_name in ['vgg19','resnet50','alexnet','densenet201','mobilenetv2']:
        for atk_name in ['fgsm','fgsm_opt']:
            filename = '../transfer_res/model_{}_atk_{}_transfer.txt'.format(model_name,atk_name)
            check_and_record_transfer(filename,record_filename)


if(__name__ == '__main__'):
    vgg19 = get_model('vgg19')
    resnet50 = get_model('resnet50')
    alexnet = get_model('alexnet')
    densenet201 = get_model('densenet201')
    mobilenetv2 = get_model('mobilenetv2')
    alpha = 0.1
    beta = 2/255
    beta_max = 8/255
    N = 30
    l1 = 0.02
    l2 = 0.05
    s_a = 0.0005
    s_b = 0.0001 
    imgs = datasets.ImageFolder('/home/hancy/dataset/imagenet5000/',transform=transforms.ToTensor())   # type: ignore
    wm_origin = cv2.imread('../img/logo.jpg')
    wm_origin = cv2.cvtColor(wm_origin,cv2.COLOR_BGR2RGB)
    wm_origin = transforms.ToTensor()(wm_origin).cuda()  # type: ignore
    model_list = [vgg19,resnet50,alexnet,densenet201,mobilenetv2]
    index = 0
    atk_name = 'ifgsm_opt'
    for model_name in ['vgg19','resnet50','alexnet','densenet201','mobilenetv2']:
        model = model_list[index]
        filename = "../transfer_res/model_{}_atk_{}_transfer.txt".format(model_name,atk_name,alpha,beta,N,l1,l2,s_a,s_b,beta_max)
        attack_and_record(filename,model,model_list,imgs,wm_origin,alpha=alpha,beta=beta,atk_name=atk_name,  # type: ignore
                            N=N,l1=l1,l2=l2,s_a=s_a,s_b=s_b,beta_max=beta_max)
        index += 1
        record_filename = '../transfer_res/transfer_result.txt'
        check_and_record_transfer(filename,record_filename)
    
