from torchvision.models import VGG
from torchvision import transforms
import torch
from torch import nn
from typing import Union, List, Any,  cast
from torchvision.models.vgg import VGG19_Weights

class VGG_with_trans(VGG):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = transforms.Resize([224,224])(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg19_IN(**kwargs: Any) -> VGG_with_trans:
    weights = VGG19_Weights.IMAGENET1K_V1
    weights = VGG19_Weights.verify(weights)
    cfgs_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    model = VGG_with_trans(features=make_layers(cfgs_cfg, batch_norm=False),**kwargs)
    model.load_state_dict(weights.get_state_dict(progress=True))
    return model

def vgg19_cifar(**kwargs: Any) -> VGG_with_trans:
    cfgs_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    # model = VGG_with_trans(features=make_layers(cfgs_cfg, batch_norm=False), num_classes=10,**kwargs)
    model = VGG_with_trans(features=make_layers(cfgs_cfg, batch_norm=False), num_classes=10,**kwargs)
    return model
