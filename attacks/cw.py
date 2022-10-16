# from torchattacks import CW
import torch
from torch import Tensor
from torch import nn
import torch.optim as optim
import sys
sys.path.append('../watermark/')
from dct_wm import *
from utils import *
sys.path.append('../attacks/')
from opti import *

def cw_direct(img: Tensor, label: Tensor, wm: Tensor,alpha:float, block_size: int, 
                model: nn.Module, c: float=1, kappa:float=0, steps:int=100, lr:float=0.01) -> Tensor:
    
    wmed_img = embed_wm(img,wm,alpha,block_size)
    images = wmed_img.clone().detach().unsqueeze(0)

    # w = torch.zeros_like(images).detach() # Requires 2x times
    w = inverse_tanh_space(images).detach()
    w.requires_grad = True

    best_adv_images = images.clone().detach()
    best_L2 = 1e10*torch.ones((len(images))).cuda()
    prev_cost = 1e10
    dim = len(images.shape)

    MSELoss = nn.MSELoss(reduction='none')
    Flatten = nn.Flatten()

    optimizer = optim.Adam([w], lr=lr)

    for step in range(steps):
        # Get adversarial images
        adv_images = tanh_space(w)

        # Calculate loss
        current_L2 = MSELoss(Flatten(adv_images),
                                Flatten(images)).sum(dim=1)
        L2_loss = current_L2.sum()

        outputs = model(adv_images)
        f_loss = f(outputs, label,kappa).sum()

        cost = L2_loss + c*f_loss

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Update adversarial images
        _, pre = torch.max(outputs.detach(), 1)
        correct = (pre == label).float()

        # filter out images that get either correct predictions or non-decreasing loss, 
        # i.e., only images that are both misclassified and loss-decreasing are left 
        mask = (1-correct)*(best_L2 > current_L2.detach())
        best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

        mask = mask.view([-1]+[1]*(dim-1))
        best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

        # Early stop when loss does not converge.
        # max(.,1) To prevent MODULO BY ZERO error in the next step.
        if step % max(steps//10,1) == 0:
            if cost.item() > prev_cost:
                return best_adv_images.squeeze(0)
            prev_cost = cost.item()

    return best_adv_images.squeeze(0)

def tanh_space(x):
    return 1/2*(torch.tanh(x) + 1)

def inverse_tanh_space(x):
    # torch.atanh is only for torch >= 1.7.0
    return atanh(x*2-1)

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

# f-function in the paper
def f(outputs, labels,kappa):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit
    return torch.clamp((j-i), min=-kappa)