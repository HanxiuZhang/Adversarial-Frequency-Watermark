from torch import Tensor
from torchvision.models.resnet import ResNet
from typing import Any
from torchvision.models.resnet import Bottleneck
import torchvision.transforms as transforms
class ResNet_with_trans(ResNet):
    def forward(self, x: Tensor) -> Tensor:
        x = transforms.Resize([224,224])(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return self._forward_impl(x)

from torchvision.models.resnet import ResNet50_Weights

def resnet50_IN(**kwargs: Any) -> ResNet_with_trans:
    weights = ResNet50_Weights.IMAGENET1K_V1
    weights = ResNet50_Weights.verify(weights)
    block = Bottleneck
    layers = [3, 4, 6, 3]
    model = ResNet_with_trans(block, layers, **kwargs)
    model.load_state_dict(weights.get_state_dict(progress=True))
    return model

def resnet50_cifar(**kwargs) -> ResNet_with_trans:
    block = Bottleneck
    layers = [3, 4, 6, 3]
    model = ResNet_with_trans(block, layers, num_classes=10, **kwargs)
    return model