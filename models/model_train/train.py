import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import io
import sys
import os
EPOCH = 100
BATCH_SIZE = 64
LR = 0.001
T = transforms.ToTensor()
train_data = datasets.CIFAR10(root='~/dataset/', train=True,transform=T,download=True)
test_data =datasets.CIFAR10(root='~/dataset/',train=False,transform=T,download=True)
sys.path.append('..')
from models.alexnet import get_alexnet
from torch.utils.data import DataLoader
from torchvision.models import AlexNet_Weights
train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
test_loader = DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
model = get_alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)
test_train = train_data[0][0]
test_train = torch.unsqueeze(test_train,0)
test_train = test_train.cuda()
model = model.cuda()
model = model.eval()
res = model(test_train)
print(res)