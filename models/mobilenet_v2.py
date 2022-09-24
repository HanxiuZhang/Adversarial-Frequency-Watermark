from torchvision.models.mobilenetv2 import MobileNetV2, MobileNet_V2_Weights
from torch import Tensor
from torchvision import transforms
from typing import Any,Optional

class MobileNetV2_with_trans(MobileNetV2):
    def forward(self, x: Tensor) -> Tensor:
        x = transforms.Resize([224,224])(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return self._forward_impl(x)

def mobilenetv2_IN(**kwargs: Any) -> MobileNetV2_with_trans:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    weights = MobileNet_V2_Weights.verify(weights)
    model = MobileNetV2_with_trans(**kwargs)
    model.load_state_dict(weights.get_state_dict(progress=True))
    return model

def mobilenetv2_cifar(**kwargs: Any) -> MobileNetV2_with_trans:
    model = MobileNetV2_with_trans(num_classes=10,**kwargs)
    return model
