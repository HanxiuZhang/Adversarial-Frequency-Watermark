from torchvision.models import VGG
from torchvision import transforms
import torch
from torch import nn
from typing import Union, List, Any, Optional, cast
from torchvision.models._utils import _ovewrite_named_param
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

def vgg19(*, weights: Optional[VGG19_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG19_Weights.verify(weights)
    # return _vgg("E", False, weights, progress, **kwargs)
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    cfgs_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]

    model = VGG_with_trans(make_layers(cfgs_cfg, batch_norm=False), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model

def vgg19_in(**kwargs: Any) -> VGG:
    weights = VGG19_Weights.IMAGENET1K_V1
    weights = VGG19_Weights.verify(weights)
    cfgs_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    model = VGG_with_trans(features=make_layers(cfgs_cfg, batch_norm=False),**kwargs)
    model.load_state_dict(weights.get_state_dict(progress=True))
    return model

def vgg19_cifar(**kwargs: Any) -> VGG:
    cfgs_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    # model = VGG_with_trans(features=make_layers(cfgs_cfg, batch_norm=False), num_classes=10,**kwargs)
    model = VGG_with_trans(features=make_layers(cfgs_cfg, batch_norm=False), num_classes=10,**kwargs)
    return model
