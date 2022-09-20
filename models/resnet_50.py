from torch import Tensor
from torchvision.models.resnet import ResNet
from torch import nn
from typing import Optional, Any
from torchvision.models.resnet import Bottleneck
import torchvision.transforms as transforms
class ResNet_with_trans(ResNet):
    def forward(self, x: Tensor) -> Tensor:
        x = transforms.Resize([224,224])(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return self._forward_impl(x)

from torchvision.models.resnet import ResNet50_Weights
from torchvision.models._utils import _ovewrite_named_param
def resnet_50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(weights)
    # model = _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
    block = Bottleneck
    layers = [3, 4, 6, 3]

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_with_trans(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    
    return model