from torchvision.models import densenet201
from torchvision.models.densenet import DenseNet, DenseNet201_Weights
import torch
from torch import Tensor
from torchvision import transforms
import torch.nn.functional as F
import re


class DenseNet_with_trans(DenseNet):
    def forward(self, x: Tensor):
        x = transforms.Resize([224,224])(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet201_IN(**kwargs) -> DenseNet_with_trans:
    weights = DenseNet201_Weights.IMAGENET1K_V1
    weights = DenseNet201_Weights.verify(weights)
    growth_rate = 32
    block_config = (6,12,48,32)
    num_init_features = 64
    model = DenseNet_with_trans(growth_rate, block_config, num_init_features, **kwargs)
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    state_dict = weights.get_state_dict(progress=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    return model

def densenet201_cifar(**kwargs) -> DenseNet_with_trans:
    growth_rate = 32
    block_config = (6,12,48,32)
    num_init_features = 64
    model = DenseNet_with_trans(growth_rate, block_config, num_init_features, num_classes=10,**kwargs)
    return model